"""Does reordering the pipeline restore distribution-freeness?

E1 showed the implemented pipeline is NOT pivotal: detrending, AR(1) fitting and
volatility standardization act on raw amplitudes, so the ranks of the residual
still depend on the marginal. Empirical size at nominal 0.05 reaches 0.19
(lognormal) and 0.88 (Cauchy).

Candidate remedies, in increasing order of severity:

  A  "raw"        current order: amplitude ops -> rank      (the broken baseline)
  B  "rank-first" rank -> amplitude ops on the ranked series
  C  "rank-only"  rank, no amplitude ops at all
  D  "rank-diff"  rank, then difference on ranks (removes drift, stays ordinal)

A pivotal variant gives the same null distribution for every continuous
marginal, hence uniform p-values everywhere.
"""

import sys as _sys, pathlib as _pathlib

# Use the ddbm package from this repository's src/ tree.
_SRC = _pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import numpy as np
from scipy import stats

from passport import (KERNELS, K_GRID, _ks_D, _pairs, _phases)
from ddbm.config import DEFAULT_CONFIG
from ddbm.ddbm_core import rank_normalize_01
from ddbm.preprocessing import make_residual

N = 10000
B_POOL = 25
B_CAL = 400
N_TEST = 150
CFG = dict(DEFAULT_CONFIG)


def prep_raw(x):
    """Current order: amplitude operations first, ranking happens later."""
    y, *_ = make_residual(np.asarray(x, float), CFG)
    return y


def prep_rank_first(x):
    """Rank first, then run the same amplitude operations on the ranked series."""
    u = rank_normalize_01(np.asarray(x, float))
    u = u[np.isfinite(u)]
    y, *_ = make_residual(u, CFG)
    return y


def prep_rank_only(x):
    u = rank_normalize_01(np.asarray(x, float))
    return u[np.isfinite(u)]


def prep_rank_diff(x):
    u = rank_normalize_01(np.asarray(x, float))
    u = u[np.isfinite(u)]
    return np.diff(u)


PREPS = {"A raw (current)": prep_raw,
         "B rank-first": prep_rank_first,
         "C rank-only": prep_rank_only,
         "D rank-diff": prep_rank_diff}

MARGINALS = {
    "gaussian": lambda n, r: r.normal(size=n),
    "uniform": lambda n, r: r.uniform(size=n),
    "t3": lambda n, r: r.standard_t(3, size=n),
    "lognormal": lambda n, r: r.lognormal(sigma=1.0, size=n),
    "exponential": lambda n, r: r.exponential(size=n),
    "cauchy": lambda n, r: r.standard_cauchy(size=n),
}


class Scan:
    """Minimal calibrated scan for one preprocessing variant."""

    def __init__(self, prep, seed=4242):
        self.prep = prep
        rng = np.random.default_rng(seed)
        self.pools = {}
        for kn, kern in KERNELS.items():
            for K in K_GRID:
                parts = [_phases(prep(rng.normal(size=N)), K, kern)
                         for _ in range(B_POOL)]
                self.pools[(kn, K)] = np.sort(np.concatenate(parts))
        self.cells = [(kn, K) for kn in KERNELS for K in K_GRID]
        D = np.empty((B_CAL, len(self.cells)))
        for b in range(B_CAL):
            z = prep(rng.normal(size=N))
            for j, (kn, K) in enumerate(self.cells):
                D[b, j] = _ks_D(_phases(z, K, KERNELS[kn]), self.pools[(kn, K)])
        self.mean = D.mean(axis=0)
        std = D.std(axis=0, ddof=1)
        self.std = np.maximum(std, 0.25 * np.median(std))
        self.T_null = ((D - self.mean) / self.std).max(axis=1)

    def p(self, x):
        y = self.prep(x)
        z = np.empty(len(self.cells))
        for j, (kn, K) in enumerate(self.cells):
            z[j] = (_ks_D(_phases(y, K, KERNELS[kn]), self.pools[(kn, K)])
                    - self.mean[j]) / self.std[j]
        T = z.max()
        return float((1 + np.sum(self.T_null >= T)) / (1 + self.T_null.size))


def main():
    print(f"{'variant':18s} {'marginal':12s} {'size@.05':>9s} {'size@.10':>9s} "
          f"{'KS':>7s} {'p_KS':>7s}")
    summary = {}
    for pname, prep in PREPS.items():
        scan = Scan(prep)
        worst = 0.0
        for mname, gen in MARGINALS.items():
            rng = np.random.default_rng(abs(hash(mname)) % (2 ** 31))
            ps = []
            for _ in range(N_TEST):
                try:
                    ps.append(scan.p(gen(N, rng)))
                except Exception:
                    continue
            ps = np.asarray(ps)
            ks = stats.kstest(ps, "uniform")
            worst = max(worst, abs(np.mean(ps < 0.05) - 0.05))
            print(f"{pname:18s} {mname:12s} {np.mean(ps < 0.05):9.3f} "
                  f"{np.mean(ps < 0.10):9.3f} {ks.statistic:7.3f} "
                  f"{ks.pvalue:7.3f}", flush=True)
        summary[pname] = worst
        print()
    print("worst |empirical size - 0.05| across marginals:")
    for k, v in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"  {k:18s} {v:.3f}")


if __name__ == "__main__":
    main()
