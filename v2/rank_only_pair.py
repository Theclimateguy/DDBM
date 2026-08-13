"""Test the preprocessing-free rank-based transition-occupancy test.

This combines the two things the earlier experiments singled out:

  * the best-performing simple statistic  -- occupancy of the (N, dN) pair;
  * the only pivotal order of operations  -- ranks only, no detrending, no
    AR(1) prewhitening, no volatility standardization.

That combination has not been run before: E4 used the original (non-pivotal)
preprocessing, and the pivotality study used the cyclotomic phase statistic.

Three statistics are compared under this preprocessing-free protocol:
pair occupancy, missing ordinal patterns, cyclotomic phases. Sizes are also
re-checked across marginals so the calibration claim is verified for the
statistic actually used.
"""

import sys as _sys, pathlib as _pathlib

# Use the ddbm package from this repository's src/ tree.
_SRC = _pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import csv
import math

import numpy as np
from scipy import stats

from passport import KERNELS, K_GRID, _ks_D, _phases
from ddbm.ddbm_core import rank_normalize_01, quantize_timeseries

N = 10000
B_CAL = 400
B_SURR = 99
K_OCC = (100, 200, 400, 800)
D_ORD = (4, 5, 6)


# ------------------------------------------------------- preprocessing-free
def prep(x):
    """Ranks only. No detrending, no prewhitening, no volatility scaling."""
    u = rank_normalize_01(np.asarray(x, float))
    return u[np.isfinite(u)]


def pairs_from_ranks(u, K):
    Nq = quantize_timeseries(u, K)
    return Nq[:-1], np.diff(Nq)


def stat_occupancy(u):
    out = []
    for K in K_OCC:
        Nc, dN = pairs_from_ranks(u, K)
        key = Nc.astype(np.int64) * 10_000_019 + dN
        out.append(np.unique(key).size / float(key.size))
    return np.asarray(out)


def stat_ordinal(u):
    out = []
    for d in D_ORD:
        w = np.lib.stride_tricks.sliding_window_view(u, d)
        order = np.argsort(w, axis=1, kind="stable")
        code = np.zeros(order.shape[0], dtype=np.int64)
        for j in range(d):
            code = code * d + order[:, j]
        out.append(1.0 - np.unique(code).size / float(math.factorial(d)))
    return np.asarray(out)


class PhaseStat:
    """Cyclotomic phases, kept for comparison. Needs its own null pools."""

    def __init__(self, seed=11):
        rng = np.random.default_rng(seed)
        self.pools = {}
        for kn, kern in KERNELS.items():
            for K in K_GRID:
                parts = [_phases(prep(rng.normal(size=N)), K, kern)
                         for _ in range(20)]
                self.pools[(kn, K)] = np.sort(np.concatenate(parts))
        self.cells = [(kn, K) for kn in KERNELS for K in K_GRID]

    def __call__(self, u):
        return np.array([_ks_D(_phases(u, K, KERNELS[kn]), self.pools[(kn, K)])
                         for kn, K in self.cells])


# ------------------------------------------------------------------ harness
class Test:
    def __init__(self, stat, seed=2026):
        self.stat = stat
        rng = np.random.default_rng(seed)
        null = np.array([stat(prep(rng.normal(size=N))) for _ in range(B_CAL)])
        self.mean = null.mean(axis=0)
        sd = null.std(axis=0, ddof=1)
        self.sd = np.maximum(sd, 0.25 * np.median(sd))
        self.T_null = np.abs((null - self.mean) / self.sd).max(axis=1)

    def T(self, x):
        return float(np.max(np.abs((self.stat(prep(x)) - self.mean) / self.sd)))

    def _p(self, obs, null):
        return float((1 + np.sum(null >= obs)) / (1 + null.size))

    def screen(self, x):
        return self._p(self.T(x), self.T_null)

    def surrogate(self, x, b=B_SURR, seed=7, n_iter=100):
        x = np.asarray(x, float)
        rng = np.random.default_rng(seed)
        T_obs = self.T(x)
        amp = np.abs(np.fft.rfft(x))
        xs = np.sort(x)
        boot = []
        for _ in range(b):
            y = rng.permutation(x)
            for _ in range(n_iter):
                Y = np.fft.rfft(y)
                y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=x.size)
                y = xs[np.argsort(np.argsort(y))]
            try:
                boot.append(self.T(y))
            except ValueError:
                continue
        if len(boot) < b // 2:
            return np.nan
        return self._p(T_obs, np.asarray(boot))


MARGINALS = {
    "gaussian": lambda n, r: r.normal(size=n),
    "uniform": lambda n, r: r.uniform(size=n),
    "t3": lambda n, r: r.standard_t(3, size=n),
    "lognormal": lambda n, r: r.lognormal(sigma=1.0, size=n),
    "exponential": lambda n, r: r.exponential(size=n),
    "cauchy": lambda n, r: r.standard_cauchy(size=n),
}


def main():
    print("building tests (preprocessing-free, ranks only) ...", flush=True)
    tests = {"occupancy": Test(stat_occupancy),
             "ordinal": Test(stat_ordinal),
             "phases": Test(PhaseStat())}

    print("\n=== empirical size of the screening stage, by marginal ===")
    print(f"{'statistic':12s} {'marginal':12s} {'size@.05':>9s} {'KS':>7s} {'p_KS':>7s}")
    for tname, t in tests.items():
        for mname, gen in MARGINALS.items():
            rng = np.random.default_rng(abs(hash(mname)) % (2 ** 31))
            ps = np.array([t.screen(gen(N, rng)) for _ in range(120)])
            ks = stats.kstest(ps, "uniform")
            print(f"{tname:12s} {mname:12s} {np.mean(ps < 0.05):9.3f} "
                  f"{ks.statistic:7.3f} {ks.pvalue:7.3f}", flush=True)
        print()

    # --------------------------------------------------------- benchmark
    man = [m for m in csv.DictReader(open("data_bench/manifest.csv"))
           if m["label"] not in ("unknown", "Mixed")]
    from run_all import is_narrowband
    from passport import Calibration
    aux = Calibration(N, b_pool=20, b_cal=150)   # only for the auxiliary criteria

    print(f"=== benchmark, strict scoring, {len(man)} labeled series ===")
    rows = []
    for m in man:
        x = np.loadtxt(f"data_bench/series/{m['name']}.csv", skiprows=1)[:N]
        r = aux.analyze(x)
        rec = {"name": m["name"], "label": m["label"]}
        for tname, t in tests.items():
            p = t.screen(x)
            if p >= 0.05:
                v = "Noise"
            else:
                ps = t.surrogate(x)
                if np.isnan(ps):
                    v = "Regular" if is_narrowband(r["regularity"]) else "Noise"
                elif ps < 0.05:
                    if is_narrowband(r["regularity"]):
                        v = "Regular"
                    else:
                        v = ("Chaos" if r["atomicity_ratio_K200"] <= 0.85
                             else "Noise")
                else:
                    reg = r["regularity"]
                    fft, acf = reg["fft_conc_topk"], reg["acf_max_abs"]
                    v = ("Regular"
                         if ((np.isfinite(fft) and fft > 0.60)
                             or (np.isfinite(acf) and acf > 0.85)) else "Noise")
            rec[tname] = v
        rows.append(rec)
        print(f"  {m['name']:22s} {m['label']:8s} "
              + "  ".join(f"{k}={rec[k]:8s}" for k in tests), flush=True)

    print()
    for k in tests:
        ok = sum(r[k] == r["label"] for r in rows)
        print(f"  {k:12s} {ok}/{len(rows)} = {100*ok/len(rows):.1f}%")
        miss = [r["name"] for r in rows if r[k] != r["label"]]
        print(f"      misses: {', '.join(miss) if miss else 'none'}")

    with open("rank_only_pair_results.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved rank_only_pair_results.csv")


if __name__ == "__main__":
    main()
