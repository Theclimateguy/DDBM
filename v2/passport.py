"""Signal passport: calibrated multi-kernel DDBM.

Builds on DDBM (github.com/Theclimateguy/DDBM) with three upgrades:

1. Exact Monte-Carlo calibration of p-values. Rank normalization makes the
   test distribution-free under H0 (iid, any continuous marginal), so the
   null distribution of the scan statistic depends only on (n, kernel, K)
   and can be simulated exactly — including the max-over-(kernel,K)
   selection, so no Bonferroni and no iid-KS approximation is needed.
   This fixes the anti-conservative p-values of the original ks_2samp use.

2. Kernel ensemble. Power kernels E = N^p + dN^p mod ((N+1)^p - N^p) for
   p in {2,3,5} act as resonant probes with different phase lattices
   ({1/4,3/4}, {0,1/3,2/3}, fifths); a pseudo-random hash kernel provides a
   pure-atomicity baseline. The scan statistic is the max standardized
   deviation across all cells; the per-kernel profile is kept as a
   qualitative signature of the dynamics.

3. Passport. Alongside the structure test it reports the pair-count
   scaling slope (a box-counting dimension estimate of the 2D delay
   embedding), the atomicity ratio, and the regularity gate of the
   original package.
"""

import sys as _sys, pathlib as _pathlib

# Use the ddbm package from this repository's src/ tree.
_SRC = _pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))


from pathlib import Path

import numpy as np

from ddbm.config import DEFAULT_CONFIG
from ddbm.ddbm_core import rank_normalize_01, quantize_timeseries
from ddbm.preprocessing import make_residual
from ddbm.regularity import regularity_gate


# ---------------------------------------------------------------- kernels
def _power_kernel(p):
    def f(Nc, dN):
        S = (Nc + 1) ** p - Nc ** p
        E = Nc ** p + dN ** p
        return np.mod((E % S) / S, 1.0)

    f.__name__ = f"p{p}"
    return f


def _hash_kernel(Nc, dN):
    h = (Nc.astype(np.uint64) * np.uint64(2654435761)
         ^ (dN.astype(np.int64).view(np.uint64) * np.uint64(40503)))
    h *= np.uint64(2246822519)
    h ^= h >> np.uint64(13)
    h *= np.uint64(3266489917)
    h ^= h >> np.uint64(16)
    return (h % np.uint64(2 ** 53)).astype(float) / float(2 ** 53)


def _cyclotomic_kernel(coeffs, deg, name):
    """Kernel from the homogeneous cyclotomic form S = Phi_n(N+1, N).

    Modulo S the ratio rho = (N+1)/N is an exact n-th root of unity, which is
    what pins the phases to a lattice with denominator Phi_n(1). The power
    kernels are the special case S_p = prod_{d|p, d>1} Phi_d(N+1, N), i.e.
    they carry redundant cyclotomic factors; the irreducible forms below are
    both cheaper and sharper.
    """
    def f(Nc, dN):
        Nc = Nc.astype(object)
        S = np.zeros_like(Nc)
        for c in coeffs:
            S = S * Nc + c
        E = Nc ** deg + dN.astype(object) ** deg
        return np.array([float(int(e) % int(s)) / float(s) if s else 0.0
                         for e, s in zip(E, S)], dtype=float)
    f.__name__ = name
    return f


# Phi_n(N+1, N) coefficient lists (highest power of N first), see
# cyclotomic_kernels.py for the symbolic derivation.
_CYC_FORMS = {
    "cyc3": ([3, 3, 1], 2),            # 3N^2+3N+1   lattice denom 3
    "cyc7": ([7, 21, 35, 35, 21, 7, 1], 6),   # lattice denom 7
    "cyc8": ([2, 4, 6, 4, 1], 4),      # 2N^4+4N^3+6N^2+4N+1  lattice denom 2
    "cyc12": ([1, 2, 5, 4, 1], 4),     # N^4+2N^3+5N^2+4N+1   lattice denom 1
}

KERNELS = {name: _cyclotomic_kernel(c, d, name) for name, (c, d) in _CYC_FORMS.items()}
KERNELS["hash"] = _hash_kernel

# the original power family, kept for the ablation comparison
POWER_KERNELS = {"p2": _power_kernel(2), "p3": _power_kernel(3),
                 "p5": _power_kernel(5), "hash": _hash_kernel}
K_GRID = (100, 200, 400, 800)
K_DIM_GRID = (25, 50, 100, 200, 400, 800)


def _pairs(x, K):
    u = rank_normalize_01(np.asarray(x, float))
    N = quantize_timeseries(u, K)
    return N[:-1], np.diff(N)


def _phases(x, K, kernel):
    Nc, dN = _pairs(x, K)
    return kernel(Nc, dN)


def _ks_D(data, pool_sorted):
    """Two-sample KS statistic, pool pre-sorted."""
    d = np.sort(np.asarray(data, float))
    n, m = d.size, pool_sorted.size
    F2 = np.searchsorted(pool_sorted, d, side="right") / m
    i = np.arange(1, n + 1) / n
    return float(max(np.max(np.abs(i - F2)), np.max(np.abs(i - 1.0 / n - F2))))


def _residual(x, config):
    """Detrend + AR(1) prewhiten (+ vol standardize). Falls back to the
    prewhitened series if volatility standardization degenerates, which it
    does for near-constant or strongly oscillatory inputs."""
    x = np.asarray(x, float)
    x_resid, *_ = make_residual(x, config)
    if x_resid.size < max(100, x.size // 4) or np.std(x_resid) < 1e-12:
        cfg = dict(config)
        cfg["ROLLVOL_POLICY"] = "off"
        x_resid, *_ = make_residual(x, cfg)
    return x_resid


# ------------------------------------------------------------ calibration
class Calibration:
    """Null model for the full pipeline at a given series length."""

    def __init__(self, n, b_pool=40, b_cal=300, seed=2026, config=None,
                 residualize=True, verbose=False):
        self.n = int(n)
        self.config = dict(DEFAULT_CONFIG if config is None else config)
        self.residualize = residualize
        rng = np.random.default_rng(seed)

        # null phase pools (iid -> same pipeline as data)
        self.pools = {}
        for kname, kern in KERNELS.items():
            for K in K_GRID:
                parts = []
                for _ in range(b_pool):
                    z = rng.normal(size=self.n)
                    if residualize:
                        z = _residual(z, self.config)
                    parts.append(_phases(z, K, kern))
                self.pools[(kname, K)] = np.sort(np.concatenate(parts))

        # calibration replicates: D per cell, pair counts per K
        cells = [(kn, K) for kn in KERNELS for K in K_GRID]
        self.cells = cells
        D = np.empty((b_cal, len(cells)))
        pair_counts = np.empty((b_cal, len(K_DIM_GRID)))
        for b in range(b_cal):
            z = rng.normal(size=self.n)
            if residualize:
                z = _residual(z, self.config)
            for j, (kn, K) in enumerate(cells):
                D[b, j] = _ks_D(_phases(z, K, KERNELS[kn]), self.pools[(kn, K)])
            for j, K in enumerate(K_DIM_GRID):
                Nc, dN = _pairs(z, K)
                pair_counts[b, j] = np.unique(
                    Nc.astype(np.int64) * 10_000_019 + dN).size
            if verbose and (b + 1) % 50 == 0:
                print(f"  calibration {b + 1}/{b_cal}")

        self.D_mean = D.mean(axis=0)
        # floor the std: a few cells have near-degenerate null D distributions
        # (atoms of the phase lattice pin the statistic), which would blow up
        # z-scores without changing MC p-values
        std_raw = D.std(axis=0, ddof=1)
        self.D_std = np.maximum(std_raw, 0.25 * np.median(std_raw))
        self.D_null = D
        self.Z_null = (D - self.D_mean) / self.D_std
        self.pairs_iid_mean = pair_counts.mean(axis=0)
        # null distribution of the ensemble scan statistic
        self.T_null = self.Z_null.max(axis=1)
        # and of single-kernel scans (for comparisons)
        self.T_null_by_kernel = {
            kn: self.Z_null[:, [j for j, c in enumerate(cells) if c[0] == kn]].max(axis=1)
            for kn in KERNELS
        }

    # ------------------------------------------------------------------
    def _mc_p(self, obs, null):
        return float((1 + np.sum(null >= obs)) / (1 + null.size))

    def _scan_T(self, series):
        """Max standardized deviation over all (kernel, K) cells."""
        z = np.empty(len(self.cells))
        for j, (kn, K) in enumerate(self.cells):
            Dv = _ks_D(_phases(series, K, KERNELS[kn]), self.pools[(kn, K)])
            z[j] = (Dv - self.D_mean[j]) / self.D_std[j]
        return z

    def iaaft_p(self, x, b_boot=99, seed=7, n_iter=100):
        """Confirmation stage: IAAFT surrogate test (Schreiber-Schmitz).

        Null hypothesis: the series is a monotone static transform of a
        linear Gaussian process — i.e. all structure lives in the power
        spectrum and the marginal distribution. IAAFT surrogates preserve
        both *exactly*, so under rank normalization a surrogate carries the
        identical set of lattice values as the data and differs only in
        temporal ordering. The test therefore isolates ordering structure,
        which is what a deterministic map or flow has and a linear process
        does not.

        This is the right null for oscillatory flows, where a parametric AR
        fit (ar_bootstrap_p) distorts the spectrum and loses power.
        """
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        rng = np.random.default_rng(seed)

        xr = _residual(x, self.config) if self.residualize else x
        T_obs = self._scan_T(xr).max()

        amp = np.abs(np.fft.rfft(x))
        xs = np.sort(x)
        T_boot = []
        for _ in range(b_boot):
            y = rng.permutation(x)
            for _ in range(n_iter):
                Y = np.fft.rfft(y)
                y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=x.size)
                y = xs[np.argsort(np.argsort(y))]
            try:
                ys = _residual(y, self.config) if self.residualize else y
                if ys.size < self.n // 2:
                    continue
                T_boot.append(self._scan_T(ys).max())
            except (ValueError, FloatingPointError):
                continue
        T_boot = np.asarray(T_boot)
        if T_boot.size < b_boot // 2:
            return np.nan
        return self._mc_p(T_obs, T_boot)

    def ar_bootstrap_p(self, x, b_boot=99, seed=7, max_order=5):
        """Confirmation stage: semi-parametric AR sieve bootstrap.

        Null hypothesis: the series is a linear AR(p) process (order by AIC,
        innovations resampled from empirical residuals). Kills detections
        that are artifacts of prewhitening a *linear* process with an
        estimated coefficient. Chaos survives: its structure is nonlinear
        and reappears in every AR surrogate's residual.
        """
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        n = x.size
        rng = np.random.default_rng(seed)

        # fit AR(p) by AIC on the (detrended) series
        xc = x - x.mean()
        best = None
        for p_ord in range(1, max_order + 1):
            X = np.column_stack([xc[p_ord - 1 - i: n - 1 - i] for i in range(p_ord)])
            y = xc[p_ord:]
            coef, res_ss, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            s2 = float(np.mean(resid ** 2))
            aic = n * np.log(s2 + 1e-300) + 2 * p_ord
            if best is None or aic < best[0]:
                best = (aic, p_ord, coef, resid)
        _, p_ord, coef, innov = best
        innov = innov - innov.mean()

        xr = _residual(x, self.config) if self.residualize else x
        T_obs = self._scan_T(xr).max()

        from scipy.signal import lfilter

        # stabilize: shrink AR roots inside the unit circle (periodic signals
        # fit near-unit-root AR whose simulation would diverge)
        coef = np.asarray(coef, float)
        roots = np.roots(np.concatenate([[1.0], -coef])) if p_ord > 0 else np.array([])
        max_root = np.max(np.abs(roots)) if roots.size else 0.0
        if max_root >= 0.995:
            shrink = 0.99 / max_root
            coef = coef * shrink ** np.arange(1, p_ord + 1)
        a_poly = np.concatenate([[1.0], -coef])

        T_boot = []
        attempts = 0
        while len(T_boot) < b_boot and attempts < 2 * b_boot:
            attempts += 1
            e = rng.choice(innov, size=n + 200, replace=True)
            xs = lfilter([1.0], a_poly, e)[200:]
            if not np.all(np.isfinite(xs)) or np.std(xs) < 1e-12:
                continue
            try:
                xsr = _residual(xs, self.config) if self.residualize else xs
                if xsr.size < self.n // 2:
                    continue
                T_boot.append(self._scan_T(xsr).max())
            except (ValueError, FloatingPointError):
                continue
        T_boot = np.asarray(T_boot)
        if T_boot.size < b_boot // 2:
            return np.nan, int(p_ord)

        return self._mc_p(T_obs, T_boot), int(p_ord)

    def analyze(self, x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        xr = _residual(x, self.config) if self.residualize else x

        z_cells = {}
        for j, (kn, K) in enumerate(self.cells):
            Dv = _ks_D(_phases(xr, K, KERNELS[kn]), self.pools[(kn, K)])
            z_cells[(kn, K)] = (Dv - self.D_mean[j]) / self.D_std[j]

        T = max(z_cells.values())
        best_cell = max(z_cells, key=z_cells.get)
        p_ens = self._mc_p(T, self.T_null)
        profile = {kn: max(v for c, v in z_cells.items() if c[0] == kn)
                   for kn in KERNELS}
        p_by_kernel = {kn: self._mc_p(profile[kn], self.T_null_by_kernel[kn])
                       for kn in KERNELS}

        # pair-count scaling -> dimension estimate (2D delay embedding).
        # Computed on the RAW series: prewhitening mixes delay coordinates and
        # distorts the attractor geometry, while the structure test needs the
        # residual to avoid linear confounds.
        counts = []
        for K in K_DIM_GRID:
            Nc, dN = _pairs(x, K)
            counts.append(np.unique(Nc.astype(np.int64) * 10_000_019 + dN).size)
        counts = np.asarray(counts, float)
        valid = counts < len(x) / 3.0
        if valid.sum() >= 3:
            lk = np.log(np.asarray(K_DIM_GRID, float)[valid])
            lc = np.log(counts[valid])
            d2 = float(np.polyfit(lk, lc, 1)[0])
        else:
            d2 = np.nan
        # Atomicity: how sparse the (N, dN) pair set is compared with the same
        # values in random order. The shuffle control is essential -- it has
        # the identical value multiset, so instrumental quantization (an RR
        # series at 128 Hz carries ~50 distinct values in 10k points and would
        # otherwise look strongly "deterministic") cancels exactly, and only
        # the ordering-induced sparsity is left.
        j200 = K_DIM_GRID.index(200)
        rng_sh = np.random.default_rng(4242)
        shuf_counts = []
        for _ in range(5):
            Nc, dN = _pairs(rng_sh.permutation(x), K_DIM_GRID[j200])
            shuf_counts.append(
                np.unique(Nc.astype(np.int64) * 10_000_019 + dN).size)
        shuffle_ref = float(np.mean(shuf_counts))
        atomicity = float(counts[j200] / max(shuffle_ref, 1.0))

        reg = regularity_gate(xr, self.config)

        # Instrumental quantization confound: the core signal of this method is
        # sparsity of (N, dN) pairs, and a coarsely digitized recording
        # manufactures exactly that sparsity out of nothing. RR-interval series
        # sampled at 128 Hz, for instance, carry ~50 distinct values in 10k
        # points and look strongly "deterministic" for purely instrumental
        # reasons. Flag it rather than silently reporting determinism.
        # With shuffle-referenced atomicity above, the confound is already
        # controlled; this stays as an informational input-resolution flag and
        # fires only on severe digitization.
        uniq_ratio = float(np.unique(x).size) / max(len(x), 1)
        quantized = uniq_ratio < 0.10

        if p_ens >= 0.05:
            verdict = "NOISE"
        elif reg["is_regular"]:
            verdict = "REGULAR"
        else:
            verdict = "CHAOS"

        return {
            "verdict": verdict,
            "p_structure": p_ens,
            "p_by_kernel": p_by_kernel,
            "profile_z": profile,
            "best_cell": best_cell,
            "T": float(T),
            "dim2_pairs": d2,
            "pair_counts": counts.tolist(),
            "atomicity_ratio_K200": atomicity,
            "regularity": reg,
            "n_resid": int(len(xr)),
            "unique_ratio": uniq_ratio,
            "quantized": bool(quantized),
        }


def save_calibration(cal, path):
    np.savez_compressed(
        path,
        n=cal.n,
        D_mean=cal.D_mean, D_std=cal.D_std, D_null=cal.D_null,
        pairs_iid_mean=cal.pairs_iid_mean,
        pools=np.array([cal.pools[c] for c in cal.cells], dtype=object),
        cells=np.array([f"{kn}|{K}" for kn, K in cal.cells]),
    )


if __name__ == "__main__":
    cal = Calibration(10000, verbose=True)
    x = [0.1]
    for _ in range(9999):
        x.append(4.0 * x[-1] * (1 - x[-1]))
    r = cal.analyze(x)
    print(r["verdict"], r["p_structure"], r["profile_z"])
