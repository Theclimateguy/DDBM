"""Re-run the ordinal baseline with the pattern length scaled to the sample.

The earlier comparison used d in {4,5,6}. At n = 1e4 those are saturated: for
d = 6 there are 720 patterns and ~1e4 windows, so the expected number of
unobserved patterns for i.i.d. data is ~7e-4 -- the statistic is identically
zero under the null and its variance vanishes. Any accuracy figure obtained
that way is an artifact of whatever preprocessing happened to perturb it.

Amigo's methodology requires d! to be comparable to the number of windows.
For n = 1e4 that means d = 7 (5040 patterns, ~700 expected missing) or d = 8
(40320 patterns, most missing). This script redoes the comparison with
d in {6,7,8} and reports both the empirical size and the benchmark accuracy,
preprocessing-free, alongside pair occupancy and cyclotomic phases.
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

from ddbm.ddbm_core import rank_normalize_01, quantize_timeseries
from rank_only_pair import (prep, stat_occupancy, PhaseStat, Test, MARGINALS, N)

D_ORD = (6, 7, 8)


def stat_ordinal_scaled(u):
    out = []
    for d in D_ORD:
        if u.size - d + 1 <= 0:
            out.append(0.0)
            continue
        w = np.lib.stride_tricks.sliding_window_view(u, d)
        order = np.argsort(w, axis=1, kind="stable")
        code = np.zeros(order.shape[0], dtype=np.int64)
        for j in range(d):
            code = code * d + order[:, j]
        out.append(1.0 - np.unique(code).size / float(math.factorial(d)))
    return np.asarray(out)


def expected_missing_report():
    print("saturation check for i.i.d. data at n = %d:" % N)
    print(f"{'d':>3s} {'d!':>8s} {'windows':>8s} {'E[missing frac]':>16s}")
    for d in range(4, 10):
        f = math.factorial(d)
        w = N - d + 1
        frac = (1 - 1.0 / f) ** w
        print(f"{d:3d} {f:8d} {w:8d} {frac:16.4f}")
    print("a usable d has E[missing frac] well away from both 0 and 1\n")


def main():
    expected_missing_report()

    tests = {
        "occupancy": Test(stat_occupancy, seed=31),
        "ordinal_d678": Test(stat_ordinal_scaled, seed=32),
        "phases": Test(PhaseStat(seed=33), seed=34),
    }

    print("=== empirical size, preprocessing-free, by marginal ===")
    print(f"{'statistic':14s} {'marginal':12s} {'size@.05':>9s} {'KS':>7s} {'p_KS':>7s}")
    for tname, t in tests.items():
        for mname, gen in MARGINALS.items():
            rng = np.random.default_rng(abs(hash(mname)) % (2 ** 31))
            ps = np.array([t.screen(gen(N, rng)) for _ in range(120)])
            ks = stats.kstest(ps, "uniform")
            print(f"{tname:14s} {mname:12s} {np.mean(ps < 0.05):9.3f} "
                  f"{ks.statistic:7.3f} {ks.pvalue:7.3f}", flush=True)
        print()

    man = [m for m in csv.DictReader(open("data_bench/manifest.csv"))
           if m["label"] not in ("unknown", "Mixed")]
    from run_all import is_narrowband
    from passport import Calibration
    aux = Calibration(N, b_pool=20, b_cal=150)

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
        print(f"  {k:14s} {ok}/{len(rows)} = {100*ok/len(rows):.1f}%")
        miss = [r["name"] for r in rows if r[k] != r["label"]]
        print(f"      misses: {', '.join(miss) if miss else 'none'}")

    print("\n--- class-wise ---")
    for cls in ["Chaos", "Regular", "Noise"]:
        sub = [r for r in rows if r["label"] == cls]
        line = f"  {cls:8s} (n={len(sub):2d}) "
        for k in tests:
            line += f"{k}={sum(r[k] == r['label'] for r in sub)}/{len(sub)}  "
        print(line)

    with open("ordinal_scaled_results.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved ordinal_scaled_results.csv")


if __name__ == "__main__":
    main()
