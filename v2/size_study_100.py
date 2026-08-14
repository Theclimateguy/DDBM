"""Size study at 100 replicates with Wilson intervals (referee item 1).

The 30-replicate table cannot separate 0.03 from 0.07, and a rate of 0.47 there
carries a 95% interval of roughly [0.29, 0.66]. This re-runs the decisive rows
at 100 replicates:

  * the three monotone transforms (AR(1), its cube, its exponential) in the raw
    and normal-scores domains -- the contrast that identifies the defect and
    the repair;
  * the remaining processes in the normal-scores domain, which is the one the
    procedure actually recommends.

The rank domain is left at 30 replicates; it is rejected on the qualitative
grounds that it fails the invariance check in the wrong direction, and no
conclusion here rests on its exact rate.
"""

import sys as _sys, pathlib as _pathlib

# Use the ddbm package from this repository's src/ tree.
_SRC = _pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

import math

import numpy as np

from rank_only_pair import stat_occupancy, Test, N
from surrogate_domain import battery, surrogate_p
from surrogate_normalscores import surrogate_p_ns

N_REP = 100
B = 49
TRANSFORMS = ["ar1_0.9", "cubed_ar1", "exp_ar1"]
OTHERS = ["white_gauss", "sine_plus_noise", "garch_returns"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(test, name, domain, n_rep=N_REP):
    rej = used = 0
    for r in range(n_rep):
        x = battery(np.random.default_rng(9000 + r))[name]
        try:
            p = (surrogate_p(test, x, domain, b=B, seed=r)[0] if domain != "ns"
                 else surrogate_p_ns(test, x, b=B, seed=r)[0])
        except Exception:
            continue
        if not np.isfinite(p):
            continue
        used += 1
        rej += (p < 0.05)
    lo, hi = wilson(rej, used)
    return rej / used, lo, hi, used


def main():
    test = Test(stat_occupancy, seed=99)
    print(f"nominal size 0.05, {N_REP} replicates, B={B}, Wilson 95% intervals\n")
    print(f"{'process':16s} {'domain':14s} {'size':>6s} {'95% CI':>16s} {'n':>5s}")

    for name in TRANSFORMS:
        for dom, label in [("raw", "raw amplitudes"), ("ns", "normal scores")]:
            s, lo, hi, n = rate(test, name, dom)
            print(f"{name:16s} {label:14s} {s:6.2f}  [{lo:.2f}, {hi:.2f}]   {n:5d}",
                  flush=True)
        print()

    for name in OTHERS:
        s, lo, hi, n = rate(test, name, "ns")
        print(f"{name:16s} {'normal scores':14s} {s:6.2f}  [{lo:.2f}, {hi:.2f}]   "
              f"{n:5d}", flush=True)


if __name__ == "__main__":
    main()
