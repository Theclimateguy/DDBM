"""Experiments demanded by both referee reports, run before rewriting the paper.

E1  Pivotality.        Is the pipeline really distribution-free? Amplitude
                       operations (detrend, AR(1), rolling-vol) act on raw
                       values BEFORE rank normalization, so the claim "exact
                       Monte-Carlo calibration" needs empirical support across
                       marginals. Calibrate under Gaussian, then test iid draws
                       from other continuous marginals: p-values must stay U(0,1).

E2  Calibration size.  1000 null replicates; KS test of screening p-values
                       against U(0,1) on fresh series; per-marginal empirical size.

E3  IAAFT stage size.  Rejection rate of the confirmation stage on a battery of
                       LINEAR and HETEROSCEDASTIC processes. The null of the
                       IAAFT test is "monotone transform of a linear Gaussian
                       process"; heteroscedastic series violate it without being
                       deterministic, which is exactly the financial-data worry.

E4  Decisive test.     Xi is a deterministic function of the pair (N, dN), so any
                       test on Xi is majorized by a direct test on the joint
                       occupancy of that pair. If a plain occupancy statistic
                       matches the cyclotomic phases, the arithmetic adds nothing.

E5  Scoring.           Accuracy with the "Mixed counts either way" rule removed,
                       plus class-wise sensitivity/specificity.

E6  Lyapunov CI.       lambda for the logistic map at r = 3.57 with a confidence
                       interval, to see whether the relabeling is defensible.

Each block prints as it finishes; partial runs are still usable.
"""

import sys as _sys, pathlib as _pathlib

# Use the ddbm package from this repository's src/ tree.
_SRC = _pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import csv
import math
import sys

import numpy as np
from scipy import stats

from passport import (Calibration, _residual, _pairs, _phases, KERNELS,
                      POWER_KERNELS, K_GRID, _ks_D)
from ddbm.config import DEFAULT_CONFIG

N = 10000
CFG = dict(DEFAULT_CONFIG)


def hdr(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72, flush=True)


# ----------------------------------------------------------------- marginals
def draw(marginal, n, rng):
    if marginal == "gaussian":
        return rng.normal(size=n)
    if marginal == "uniform":
        return rng.uniform(size=n)
    if marginal == "t3":
        return rng.standard_t(3, size=n)
    if marginal == "lognormal":
        return rng.lognormal(sigma=1.0, size=n)
    if marginal == "exponential":
        return rng.exponential(size=n)
    if marginal == "cauchy":
        return rng.standard_cauchy(size=n)
    raise ValueError(marginal)


MARGINALS = ["gaussian", "uniform", "t3", "lognormal", "exponential", "cauchy"]


# ------------------------------------------------------------------ E1 + E2
def e1_e2(b_cal=1000, n_test=200):
    hdr("E1/E2  pivotality and calibration size "
        f"(calibrated under Gaussian, B={b_cal})")
    cal = Calibration(N, b_pool=40, b_cal=b_cal, seed=20260806)
    print(f"null scan statistic T: mean={cal.T_null.mean():.3f} "
          f"sd={cal.T_null.std():.3f} q95={np.quantile(cal.T_null, 0.95):.3f}")

    print(f"\n{'marginal':12s} {'n':>4s} {'size@.05':>9s} {'size@.10':>9s} "
          f"{'KS vs U(0,1)':>13s} {'p_KS':>7s}")
    out = {}
    for m in MARGINALS:
        rng = np.random.default_rng(hash(m) % (2**31))
        ps = []
        for i in range(n_test):
            x = draw(m, N, rng)
            try:
                ps.append(cal.analyze(x)["p_structure"])
            except Exception:
                continue
        ps = np.asarray(ps)
        ks = stats.kstest(ps, "uniform")
        out[m] = ps
        print(f"{m:12s} {len(ps):4d} {np.mean(ps < 0.05):9.3f} "
              f"{np.mean(ps < 0.10):9.3f} {ks.statistic:13.3f} {ks.pvalue:7.3f}",
              flush=True)
    print("\nnominal size is 0.05 / 0.10; KS p < 0.05 means the p-values are NOT "
          "uniform,\ni.e. the calibration does not transfer to that marginal.")
    return cal, out


# ---------------------------------------------------------------------- E3
def linear_battery(rng, n=N):
    from scipy.signal import lfilter
    t = np.arange(n)
    bat = {}
    bat["white_gauss"] = rng.normal(size=n)
    bat["ar1_0.5"] = lfilter([1.0], [1.0, -0.5], rng.normal(size=n + 500))[500:]
    bat["ar1_0.9"] = lfilter([1.0], [1.0, -0.9], rng.normal(size=n + 500))[500:]
    bat["ar2_osc"] = lfilter([1.0], [1.0, -1.4, 0.7], rng.normal(size=n + 500))[500:]
    bat["arma22"] = lfilter([1.0, 0.4, -0.3], [1.0, -0.6, 0.2],
                            rng.normal(size=n + 500))[500:]
    # fractional Gaussian-ish long memory via cumulative filtering
    e = rng.normal(size=n + 2000)
    w = (np.arange(1, 401) ** (-0.8))
    bat["long_memory"] = np.convolve(e, w, mode="valid")[:n]
    bat["sine_plus_noise"] = np.sin(2 * np.pi * 0.05 * t) + 0.5 * rng.normal(size=n)
    bat["t3_iid"] = rng.standard_t(3, size=n)
    # static monotone transform of a linear process: still inside the IAAFT null
    bat["cubed_ar1"] = np.sign(bat["ar1_0.5"]) * np.abs(bat["ar1_0.5"]) ** 3
    # heteroscedastic: OUTSIDE the IAAFT null, though not deterministic
    s2 = 0.01 / (1 - 0.05 - 0.94)
    g = np.zeros(n)
    for i in range(n):
        g[i] = np.sqrt(s2) * rng.normal()
        s2 = 0.01 + 0.05 * g[i] ** 2 + 0.94 * s2
    bat["garch_returns"] = g
    bat["stochvol"] = np.exp(0.5 * lfilter([1.0], [1.0, -0.98],
                                           0.1 * rng.normal(size=n + 500))[500:]) \
        * rng.normal(size=n)
    return bat


def e3(cal, n_rep=8):
    hdr("E3  size of the IAAFT confirmation stage on linear / heteroscedastic "
        "processes")
    print("first six rows are INSIDE the IAAFT null (rejection = false positive).")
    print("garch/stochvol are heteroscedastic: outside the null but NOT "
          "deterministic.\n")
    print(f"{'process':18s} {'screen<.05':>11s} {'IAAFT<.05':>10s} {'n_rep':>6s}")
    for name in linear_battery(np.random.default_rng(0)).keys():
        rej_s, rej_i, used = 0, 0, 0
        for r in range(n_rep):
            rng = np.random.default_rng(1000 + r)
            x = linear_battery(rng)[name]
            try:
                res = cal.analyze(x)
            except Exception:
                continue
            used += 1
            if res["p_structure"] < 0.05:
                rej_s += 1
                try:
                    p = cal.iaaft_p(x, b_boot=99, seed=r)
                except Exception:
                    p = np.nan
                if np.isfinite(p) and p < 0.05:
                    rej_i += 1
        if used:
            print(f"{name:18s} {rej_s/used:11.2f} {rej_i/used:10.2f} {used:6d}",
                  flush=True)


# ---------------------------------------------------------------------- E4
def pair_occupancy(x, K):
    """Plain occupancy statistic on the joint (N, dN) pair -- no arithmetic."""
    Nc, dN = _pairs(x, K)
    key = Nc.astype(np.int64) * 10_000_019 + dN
    return float(np.unique(key).size) / float(len(key))


def missing_ordinal(x, d):
    x = np.asarray(x, float)
    if x.size - d + 1 <= 0:
        return 0.0
    win = np.lib.stride_tricks.sliding_window_view(x, d)
    order = np.argsort(win, axis=1, kind="stable")
    code = np.zeros(order.shape[0], dtype=np.int64)
    for j in range(d):
        code = code * d + order[:, j]
    return 1.0 - np.unique(code).size / float(math.factorial(d))


class GenericTest:
    """Same protocol, arbitrary statistic vector."""

    def __init__(self, stat_fn, n, b_cal=300, seed=5150):
        self.f = stat_fn
        self.n = n
        rng = np.random.default_rng(seed)
        null = np.array([self.f(_residual(rng.normal(size=n), CFG))
                         for _ in range(b_cal)])
        if null.ndim == 1:
            null = null[:, None]
        self.mean = null.mean(axis=0)
        self.std = np.maximum(null.std(axis=0, ddof=1), 1e-12)
        self.T_null = ((null - self.mean) / self.std).max(axis=1)

    def _T(self, x):
        v = np.atleast_1d(self.f(x))
        return float(np.max(np.abs((v - self.mean) / self.std)))

    def _p(self, obs, null):
        return float((1 + np.sum(null >= obs)) / (1 + null.size))

    def screen(self, x):
        return self._p(self._T(_residual(np.asarray(x, float), CFG)), self.T_null)

    def iaaft_p(self, x, b=99, seed=7, n_iter=100):
        x = np.asarray(x, float)
        rng = np.random.default_rng(seed)
        T_obs = self._T(_residual(x, CFG))
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
                boot.append(self._T(_residual(y, CFG)))
            except ValueError:
                continue
        if len(boot) < b // 2:
            return np.nan
        return self._p(T_obs, np.asarray(boot))


def e4_e5(cal):
    hdr("E4/E5  decisive comparison: cyclotomic phases vs plain pair occupancy "
        "vs ordinal patterns")
    print("Xi is a deterministic function of (N, dN); a direct occupancy test on "
          "that pair\nmajorizes any test on Xi. If occupancy matches the phases, "
          "the arithmetic is idle.\n")

    from run_all import classify, is_narrowband

    occ = GenericTest(lambda x: np.array([pair_occupancy(x, K) for K in K_GRID]), N)
    ordn = GenericTest(lambda x: np.array([missing_ordinal(x, d) for d in (4, 5, 6)]), N)

    man = [m for m in csv.DictReader(open("data_bench/manifest.csv"))
           if m["label"] != "unknown"]

    def verdict_generic(t, x, r):
        p = t.screen(x)
        if p >= 0.05:
            return "Noise"
        ps = t.iaaft_p(x)
        if np.isnan(ps):
            return "Regular" if is_narrowband(r["regularity"]) else "Noise"
        if ps < 0.05:
            if is_narrowband(r["regularity"]):
                return "Regular"
            return "Chaos" if r["atomicity_ratio_K200"] <= 0.85 else "NonlinStoch"
        reg = r["regularity"]
        fft, acf = reg["fft_conc_topk"], reg["acf_max_abs"]
        per = ((np.isfinite(fft) and fft > 0.60) or (np.isfinite(acf) and acf > 0.85))
        return "Regular" if per else "Noise"

    rows = []
    for m in man:
        x = np.loadtxt(f"data_bench/series/{m['name']}.csv", skiprows=1)[:N]
        r = cal.analyze(x)
        rows.append(dict(name=m["name"], label=m["label"],
                         cyc=classify(cal, x, r)[0],
                         occ=verdict_generic(occ, x, r),
                         ordn=verdict_generic(ordn, x, r)))
        print(f"  {m['name']:22s} {m['label']:8s} cyc={rows[-1]['cyc']:12s} "
              f"occ={rows[-1]['occ']:12s} ord={rows[-1]['ordn']:12s}", flush=True)

    def strict(exp, pred):
        if pred == "NonlinStoch":
            pred = "Noise"
        return exp == pred

    hard = [r for r in rows if r["label"] != "Mixed"]
    print(f"\n--- STRICT scoring, Mixed cases excluded ({len(hard)} series) ---")
    for k, nm in [("cyc", "cyclotomic phases"), ("occ", "pair occupancy"),
                  ("ordn", "missing ordinal patterns")]:
        ok = sum(strict(r["label"], r[k]) for r in hard)
        print(f"  {nm:26s} {ok}/{len(hard)} = {100*ok/len(hard):.1f}%")

    print("\n--- class-wise (strict) ---")
    for cls in ["Chaos", "Regular", "Noise"]:
        sub = [r for r in hard if r["label"] == cls]
        line = f"  {cls:8s} (n={len(sub):2d}) "
        for k in ["cyc", "occ", "ordn"]:
            line += f"{k}={sum(strict(r['label'], r[k]) for r in sub)}/{len(sub)}  "
        print(line)

    with open("reviewer_e4_results.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved reviewer_e4_results.csv")


# ---------------------------------------------------------------------- E6
def e6():
    hdr("E6  Lyapunov exponent of the logistic map near r = 3.57, with CI")
    fk = 3.569945672  # Feigenbaum accumulation point
    print(f"Feigenbaum accumulation point r_inf = {fk:.9f}\n")
    print(f"{'r':>12s} {'lambda':>10s} {'95% CI':>22s}")
    for r in [3.5699, fk, 3.5700, 3.5705, 3.57, 3.575, 3.58]:
        vals = []
        for seed in range(12):
            x = np.random.default_rng(seed).uniform(0.2, 0.8)
            for _ in range(20000):
                x = r * x * (1 - x)
            acc = []
            for _ in range(200000):
                acc.append(np.log(abs(r * (1 - 2 * x)) + 1e-300))
                x = r * x * (1 - x)
            vals.append(np.mean(acc))
        v = np.asarray(vals)
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"{r:12.7f} {v.mean():10.5f}   [{lo:8.5f}, {hi:8.5f}]", flush=True)
    print("\nIf the CI at r = 3.57 straddles zero, the v2 relabeling to 'Chaos' "
          "is not defensible\nand the case must be reported as indeterminate.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cal = None
    if which in ("all", "e1", "e3", "e4"):
        cal, _ = e1_e2(b_cal=1000 if which in ("all", "e1") else 300)
    if which in ("all", "e3"):
        e3(cal)
    if which in ("all", "e4"):
        e4_e5(cal)
    if which in ("all", "e6"):
        e6()
