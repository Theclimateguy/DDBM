"""Must the surrogate stage live in the same domain as the statistic?

R2 exposed a contradiction. A static monotone transform preserves ranks
exactly, so a rank-based statistic cannot distinguish AR(1) from a cubed AR(1).
Yet the confirmation stage rejects AR(1) in 8% of replicates and the cubed
series in 100%. The difference cannot come from the statistic; it comes from
the surrogates, which are built from raw amplitudes. For a strongly transformed
marginal the IAAFT iteration converges poorly, the surrogates fail to reproduce
the periodogram, and the test rejects the mismatch rather than any nonlinearity.

The fix is the same lesson as Section 3, applied once more: generate the
surrogates in the domain the statistic actually uses. This compares

  raw-domain  IAAFT on x, statistic on rank(x)     (as implemented)
  rank-domain IAAFT on rank(x), statistic on it    (domain-consistent)

on processes inside the IAAFT null, on heteroscedastic processes, and on
deterministic systems, so that any loss of power is visible too.
"""

import sys as _sys, pathlib as _pathlib

# Use the ddbm package from this repository's src/ tree.
_SRC = _pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import math

import numpy as np

from rank_only_pair import prep, stat_occupancy, Test, N
from ddbm.ddbm_core import rank_normalize_01


def iaaft(x, rng, n_iter=100):
    amp = np.abs(np.fft.rfft(x))
    xs = np.sort(x)
    y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=x.size)
        y = xs[np.argsort(np.argsort(y))]
    return y


def spectral_mismatch(x, y):
    """Relative L2 error between the periodograms of data and surrogate."""
    a = np.abs(np.fft.rfft(x)) ** 2
    b = np.abs(np.fft.rfft(y)) ** 2
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-300))


def surrogate_p(test, x, domain, b=49, seed=7):
    rng = np.random.default_rng(seed)
    base = np.asarray(x, float) if domain == "raw" else prep(x)
    T_obs = test.T(base)
    boot, mism = [], []
    for _ in range(b):
        y = iaaft(base, rng)
        mism.append(spectral_mismatch(base, y))
        try:
            boot.append(test.T(y))
        except ValueError:
            continue
    if len(boot) < b // 2:
        return np.nan, np.nan
    p = float((1 + np.sum(np.asarray(boot) >= T_obs)) / (1 + len(boot)))
    return p, float(np.mean(mism))


def battery(rng, n=N):
    from scipy.signal import lfilter
    t = np.arange(n)
    b = {}
    a1 = lfilter([1.0], [1.0, -0.9], rng.normal(size=n + 500))[500:]
    b["ar1_0.9"] = a1
    b["cubed_ar1"] = np.sign(a1) * np.abs(a1) ** 3
    b["exp_ar1"] = np.exp(a1 / a1.std())
    b["white_gauss"] = rng.normal(size=n)
    b["sine_plus_noise"] = np.sin(2 * np.pi * 0.05 * t) + 0.5 * rng.normal(size=n)
    s2 = 0.01 / (1 - 0.05 - 0.94)
    g = np.zeros(n)
    for i in range(n):
        g[i] = np.sqrt(s2) * rng.normal()
        s2 = 0.01 + 0.05 * g[i] ** 2 + 0.94 * s2
    b["garch_returns"] = g
    return b


DETERMINISTIC = ["logistic_r4.00", "henon_x_a1.4", "lorenz_x_rho28",
                 "rossler_x_c5.7", "chua_x"]


def main():
    test = Test(stat_occupancy, seed=99)
    n_rep = 30

    print("=== inside the IAAFT null, or stochastic: rejection rate should be "
          "at most nominal 0.05 ===")
    print(f"{'process':16s} {'raw p<.05':>10s} {'rank p<.05':>11s} "
          f"{'spec.mismatch raw':>18s} {'rank':>8s}")
    for name in battery(np.random.default_rng(0)):
        r_raw = r_rank = 0
        m_raw = m_rank = []
        m_raw, m_rank = [], []
        for r in range(n_rep):
            x = battery(np.random.default_rng(7000 + r))[name]
            p1, s1 = surrogate_p(test, x, "raw", seed=r)
            p2, s2 = surrogate_p(test, x, "rank", seed=r)
            r_raw += (np.isfinite(p1) and p1 < 0.05)
            r_rank += (np.isfinite(p2) and p2 < 0.05)
            m_raw.append(s1)
            m_rank.append(s2)
        print(f"{name:16s} {r_raw/n_rep:10.2f} {r_rank/n_rep:11.2f} "
              f"{np.nanmean(m_raw):18.3f} {np.nanmean(m_rank):8.3f}", flush=True)

    print("\n=== deterministic systems: rejection is the desired outcome ===")
    print(f"{'system':18s} {'raw p':>8s} {'rank p':>8s}")
    for nm in DETERMINISTIC:
        try:
            x = np.loadtxt(f"data_bench/series/{nm}.csv", skiprows=1)[:N]
        except OSError:
            continue
        p1, _ = surrogate_p(test, x, "raw", b=99)
        p2, _ = surrogate_p(test, x, "rank", b=99)
        print(f"{nm:18s} {p1:8.3f} {p2:8.3f}", flush=True)


if __name__ == "__main__":
    main()
