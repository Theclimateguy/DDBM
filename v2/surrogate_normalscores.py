"""A third surrogate domain: normal scores.

The IAAFT null is "a static monotone transform of a linear Gaussian process".
Matching the periodogram of the *transformed* series is therefore the wrong
target: a static nonlinearity changes the autocorrelation function, so IAAFT
run on raw amplitudes is only approximately faithful, and for strong transforms
it is not faithful at all (cubed and exponentiated AR(1) are rejected in 100%
of replicates). Running IAAFT on the ranks instead over-corrects and breaks
strongly autocorrelated cases.

The textbook remedy is to work in normal scores: map to Phi^{-1}(rank/(n+1)),
which is the Gaussian series the null is actually about, generate surrogates
there, and evaluate the rank-based statistic on them. This tests whether that
third domain gives valid size where the other two do not, and whether power on
deterministic systems survives.
"""
import numpy as np
from scipy import stats as sstats

from rank_only_pair import prep, stat_occupancy, Test, N
from surrogate_domain import iaaft, spectral_mismatch, battery, DETERMINISTIC


def normal_scores(x):
    x = np.asarray(x, float)
    r = sstats.rankdata(x, method="average")
    return sstats.norm.ppf(r / (len(x) + 1.0))


def surrogate_p_ns(test, x, b=49, seed=7):
    rng = np.random.default_rng(seed)
    base = normal_scores(x)
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
    return (float((1 + np.sum(np.asarray(boot) >= T_obs)) / (1 + len(boot))),
            float(np.mean(mism)))


def main():
    test = Test(stat_occupancy, seed=99)
    n_rep = 30

    print("=== size, normal-scores surrogates (nominal 0.05) ===")
    print(f"{'process':16s} {'p<.05':>7s} {'spec.mismatch':>14s}")
    for name in battery(np.random.default_rng(0)):
        rej, mism = 0, []
        for r in range(n_rep):
            x = battery(np.random.default_rng(7000 + r))[name]
            p, m = surrogate_p_ns(test, x, seed=r)
            rej += (np.isfinite(p) and p < 0.05)
            mism.append(m)
        print(f"{name:16s} {rej/n_rep:7.2f} {np.nanmean(mism):14.3f}", flush=True)

    print("\n=== power on deterministic systems (rejection desired) ===")
    print(f"{'system':18s} {'p':>7s}")
    for nm in DETERMINISTIC:
        try:
            x = np.loadtxt(f"data_bench/series/{nm}.csv", skiprows=1)[:N]
        except OSError:
            continue
        p, _ = surrogate_p_ns(test, x, b=99)
        print(f"{nm:18s} {p:7.3f}", flush=True)


if __name__ == "__main__":
    main()
