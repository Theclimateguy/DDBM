"""Headline benchmark table in the final recommended configuration.

Both stages corrected:
  screening     -- rank-only symbolization, exact Monte-Carlo calibration;
  confirmation  -- IAAFT surrogates generated in NORMAL SCORES, the domain in
                   which the AAFT/IAAFT null is actually defined.

Earlier tables used amplitude-domain surrogates, which are invalid for static
monotone transforms of autocorrelated processes (rejection 1.00) and cost power
on Chua. This regenerates the comparison so the reported numbers match the
procedure the paper recommends.
"""
import csv
import math

import numpy as np

from rank_only_pair import prep, stat_occupancy, PhaseStat, Test, N
from surrogate_normalscores import surrogate_p_ns


def missing_ordinal(u, d):
    if u.size - d + 1 <= 0:
        return 0.0
    w = np.lib.stride_tricks.sliding_window_view(u, d)
    order = np.argsort(w, axis=1, kind="stable")
    code = np.zeros(order.shape[0], dtype=np.int64)
    for j in range(d):
        code = code * d + order[:, j]
    return 1.0 - np.unique(code).size / float(math.factorial(d))


def stat_ordinal(u):
    return np.array([missing_ordinal(u, d) for d in (6, 7, 8)])


def main():
    tests = {"occupancy": Test(stat_occupancy, seed=131),
             "ordinal": Test(stat_ordinal, seed=132),
             "phases": Test(PhaseStat(seed=133), seed=134)}

    man = [m for m in csv.DictReader(open("data_bench/manifest.csv"))
           if m["label"] not in ("unknown", "Mixed")]
    from run_all import is_narrowband
    from passport import Calibration
    aux = Calibration(N, b_pool=20, b_cal=150)

    print(f"=== final configuration, {len(man)} labeled series, strict scoring ===")
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
                ps, _ = surrogate_p_ns(t, x, b=99)
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
                    v = ("Regular" if ((np.isfinite(fft) and fft > 0.60)
                                       or (np.isfinite(acf) and acf > 0.85))
                         else "Noise")
            rec[tname] = v
        rows.append(rec)
        print(f"  {m['name']:22s} {m['label']:8s} "
              + "  ".join(f"{k}={rec[k]:8s}" for k in tests), flush=True)

    print()
    for k in tests:
        ok = sum(r[k] == r["label"] for r in rows)
        print(f"  {k:11s} {ok}/{len(rows)} = {100*ok/len(rows):.1f}%")
        miss = [r["name"] for r in rows if r[k] != r["label"]]
        print(f"      misses: {', '.join(miss) if miss else 'none'}")

    print("\n--- class-wise ---")
    for cls in ["Chaos", "Regular", "Noise"]:
        sub = [r for r in rows if r["label"] == cls]
        line = f"  {cls:8s} (n={len(sub):2d}) "
        for k in tests:
            line += f"{k}={sum(r[k] == r['label'] for r in sub)}/{len(sub)}  "
        print(line)

    with open("final_benchmark_results.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved final_benchmark_results.csv")


if __name__ == "__main__":
    main()
