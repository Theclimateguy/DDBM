"""Experiments demanded as essential by the third referee report.

R2  IAAFT audit at 50 replicates instead of 8, so the leakage rates in the
    surrogate-stage table carry usable uncertainty.

R3  Sensitivity of the ordinal baseline to the pattern-length grid. The paper
    uses d in {6,7,8}; the referee asks whether that choice is principled and
    whether the ranking survives reasonable variation.

R5  The corrected, preprocessing-free procedure applied explicitly to S&P 500
    returns and realized volatility, with BOTH an IAAFT null and a model-based
    GARCH(1,1) null. The IAAFT null is known to be wrong for heteroscedastic
    data (that is why the v1 conclusion was withdrawn); the model-based null is
    the appropriate one.

R1/R4 are editorial plus a reproducibility manifest; handled in the paper and by
    hash_manifest.py.
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
from scipy import stats, optimize

from rank_only_pair import prep, stat_occupancy, Test, N
from ddbm.ddbm_core import rank_normalize_01


def hdr(s):
    print("\n" + "=" * 74 + f"\n{s}\n" + "=" * 74, flush=True)


def missing_ordinal(u, d):
    if u.size - d + 1 <= 0:
        return 0.0
    w = np.lib.stride_tricks.sliding_window_view(u, d)
    order = np.argsort(w, axis=1, kind="stable")
    code = np.zeros(order.shape[0], dtype=np.int64)
    for j in range(d):
        code = code * d + order[:, j]
    return 1.0 - np.unique(code).size / float(math.factorial(d))


def make_ord_stat(ds):
    def f(u):
        return np.array([missing_ordinal(u, d) for d in ds])
    return f


# --------------------------------------------------------------------- R2
def linear_battery(rng, n=N):
    from scipy.signal import lfilter
    t = np.arange(n)
    b = {}
    b["white_gauss"] = rng.normal(size=n)
    b["ar1_0.9"] = lfilter([1.0], [1.0, -0.9], rng.normal(size=n + 500))[500:]
    b["ar2_osc"] = lfilter([1.0], [1.0, -1.4, 0.7], rng.normal(size=n + 500))[500:]
    b["arma22"] = lfilter([1.0, 0.4, -0.3], [1.0, -0.6, 0.2],
                          rng.normal(size=n + 500))[500:]
    e = rng.normal(size=n + 2000)
    w = np.arange(1, 401) ** (-0.8)
    b["long_memory"] = np.convolve(e, w, mode="valid")[:n]
    b["sine_plus_noise"] = np.sin(2 * np.pi * 0.05 * t) + 0.5 * rng.normal(size=n)
    a1 = b["ar1_0.9"]
    b["cubed_ar1"] = np.sign(a1) * np.abs(a1) ** 3
    s2 = 0.01 / (1 - 0.05 - 0.94)
    g = np.zeros(n)
    for i in range(n):
        g[i] = np.sqrt(s2) * rng.normal()
        s2 = 0.01 + 0.05 * g[i] ** 2 + 0.94 * s2
    b["garch_returns"] = g
    return b


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def r2_iaaft_audit(n_rep=50, b_boot=49):
    hdr(f"R2  IAAFT confirmation-stage size, {n_rep} replicates, B={b_boot}")
    test = Test(stat_occupancy, seed=77)
    names = list(linear_battery(np.random.default_rng(0)).keys())
    print(f"{'process':18s} {'screen':>7s} {'IAAFT':>7s} {'95% CI (IAAFT)':>18s} {'n':>4s}")
    rows = []
    for name in names:
        rej_s = rej_i = used = 0
        for r in range(n_rep):
            x = linear_battery(np.random.default_rng(5000 + r))[name]
            try:
                p = test.screen(x)
            except Exception:
                continue
            used += 1
            if p < 0.05:
                rej_s += 1
                try:
                    pi = test.surrogate(x, b=b_boot, seed=r)
                except Exception:
                    pi = np.nan
                if np.isfinite(pi) and pi < 0.05:
                    rej_i += 1
        lo, hi = wilson(rej_i, used)
        print(f"{name:18s} {rej_s/used:7.2f} {rej_i/used:7.2f} "
              f"     [{lo:.2f}, {hi:.2f}] {used:4d}", flush=True)
        rows.append((name, rej_s / used, rej_i / used, lo, hi, used))
    return rows


# --------------------------------------------------------------------- R3
def r3_pattern_length():
    hdr("R3  sensitivity of the ordinal baseline to the pattern-length grid")
    print("expected missing fraction under an i.i.d. null at n = %d:" % N)
    for d in range(5, 10):
        f = math.factorial(d)
        print(f"   d={d}: d!={f:7d}   E[missing]={(1 - 1.0/f)**(N-d+1):.4f}")
    print("\na usable grid keeps E[missing] away from both 0 and 1.\n")

    grids = {"{7}": (7,), "{8}": (8,), "{7,8}": (7, 8),
             "{6,7,8}": (6, 7, 8), "{5,6,7,8}": (5, 6, 7, 8), "{7,8,9}": (7, 8, 9)}
    man = [m for m in csv.DictReader(open("data_bench/manifest.csv"))
           if m["label"] not in ("unknown", "Mixed")]
    series = {m["name"]: np.loadtxt(f"data_bench/series/{m['name']}.csv",
                                    skiprows=1)[:N] for m in man}
    from run_all import is_narrowband
    from passport import Calibration
    aux = Calibration(N, b_pool=20, b_cal=150)
    auxres = {m["name"]: aux.analyze(series[m["name"]]) for m in man}

    print(f"{'grid':12s} {'accuracy':>10s} {'size(gauss)':>12s} {'size(cauchy)':>13s}")
    for gname, ds in grids.items():
        t = Test(make_ord_stat(ds), seed=abs(hash(gname)) % 10000)
        ok = 0
        for m in man:
            x = series[m["name"]]
            r = auxres[m["name"]]
            p = t.screen(x)
            if p >= 0.05:
                v = "Noise"
            else:
                ps = t.surrogate(x, b=49)
                if np.isnan(ps):
                    v = "Regular" if is_narrowband(r["regularity"]) else "Noise"
                elif ps < 0.05:
                    v = ("Regular" if is_narrowband(r["regularity"])
                         else ("Chaos" if r["atomicity_ratio_K200"] <= 0.85
                               else "Noise"))
                else:
                    reg = r["regularity"]
                    fft, acf = reg["fft_conc_topk"], reg["acf_max_abs"]
                    v = ("Regular" if ((np.isfinite(fft) and fft > 0.60)
                                       or (np.isfinite(acf) and acf > 0.85))
                         else "Noise")
            ok += (v == m["label"])
        sg = np.mean([t.screen(np.random.default_rng(900 + i).normal(size=N))
                      < 0.05 for i in range(60)])
        sc = np.mean([t.screen(np.random.default_rng(950 + i).standard_cauchy(size=N))
                      < 0.05 for i in range(60)])
        print(f"{gname:12s} {ok:4d}/{len(man):<5d} {sg:12.3f} {sc:13.3f}", flush=True)


# --------------------------------------------------------------------- R5
def garch_fit(x):
    """Gaussian QMLE for GARCH(1,1) on mean-centred data."""
    x = np.asarray(x, float)
    x = x - x.mean()
    v = np.var(x)

    def nll(theta):
        om, al, be = np.exp(theta)
        if al + be >= 0.999:
            return 1e10
        s2 = np.empty(x.size)
        s2[0] = v
        for i in range(1, x.size):
            s2[i] = om + al * x[i - 1] ** 2 + be * s2[i - 1]
        if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e10
        return 0.5 * np.sum(np.log(s2) + x ** 2 / s2)

    th0 = np.log([v * 0.05, 0.05, 0.90])
    res = optimize.minimize(nll, th0, method="Nelder-Mead",
                            options=dict(maxiter=2000, fatol=1e-3))
    om, al, be = np.exp(res.x)
    return float(om), float(min(al, 0.5)), float(min(be, 0.98))


def garch_surrogate(x, params, rng):
    om, al, be = params
    n = x.size
    s2 = om / max(1 - al - be, 1e-6)
    out = np.empty(n)
    for i in range(n):
        out[i] = math.sqrt(s2) * rng.normal()
        s2 = om + al * out[i] ** 2 + be * s2
    return out


def r5_sp500(b=99):
    hdr("R5  corrected procedure on S&P 500, IAAFT null vs model-based GARCH null")
    test = Test(stat_occupancy, seed=88)
    targets = ["sp500_log_returns_full", "sp500_log_returns_2010_2024",
               "sp500_realized_vol"]
    print(f"{'series':30s} {'n':>6s} {'p_screen':>9s} {'p_IAAFT':>8s} "
          f"{'p_GARCH':>8s}  reading")
    for nm in targets:
        try:
            x = np.loadtxt(f"data_bench/series/{nm}.csv", skiprows=1)[:N]
        except OSError:
            print(f"{nm:30s}  (series file absent; run build_dataset.py)")
            continue
        p_s = test.screen(x)
        if p_s >= 0.05:
            print(f"{nm:30s} {x.size:6d} {p_s:9.3f} {'-':>8s} {'-':>8s}  "
                  f"no structure beyond i.i.d.")
            continue
        p_i = test.surrogate(x, b=b)

        params = garch_fit(x)
        rng = np.random.default_rng(4)
        T_obs = test.T(x)
        boot = []
        for _ in range(b):
            try:
                boot.append(test.T(garch_surrogate(x, params, rng)))
            except ValueError:
                continue
        p_g = float((1 + np.sum(np.asarray(boot) >= T_obs)) / (1 + len(boot)))

        if p_g >= 0.05:
            reading = "explained by GARCH(1,1)"
        elif p_i < 0.05:
            reading = "beyond both nulls"
        else:
            reading = "ambiguous"
        print(f"{nm:30s} {x.size:6d} {p_s:9.3f} {p_i:8.3f} {p_g:8.3f}  {reading}",
              flush=True)
        print(f"{'':30s}   fitted GARCH: omega={params[0]:.2e} "
              f"alpha={params[1]:.3f} beta={params[2]:.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "r5"):
        r5_sp500()
    if which in ("all", "r3"):
        r3_pattern_length()
    if which in ("all", "r2"):
        r2_iaaft_audit()
