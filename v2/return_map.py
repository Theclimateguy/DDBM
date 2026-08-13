"""Return-map (Poincare-style) reduction for continuous flows.

A densely sampled smooth flow puts most of its structure into the power
spectrum, which IAAFT surrogates reproduce exactly -- so the surrogate test
has little power on the raw flow. The classical remedy in nonlinear time
series analysis is to reduce the flow to its return map: the sequence of
successive local maxima. For the Lorenz z-component this is the textbook
tent-like map. The reduction discards the smooth interpolation between
extrema and keeps exactly the deterministic map that generates them.

Usage: take successive local maxima of the observable, then run the passport
on that (much shorter, map-like) series.
"""
import numpy as np


def local_maxima(x, min_sep=1):
    """Values of successive local maxima."""
    x = np.asarray(x, float)
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if min_sep > 1 and idx.size:
        keep = [idx[0]]
        for i in idx[1:]:
            if i - keep[-1] >= min_sep:
                keep.append(i)
        idx = np.asarray(keep)
    return x[idx]


if __name__ == "__main__":
    import flows
    from passport import Calibration

    N_LONG = 400_000  # long run: return maps are far shorter than the flow
    targets = ["lorenz_z_rho28", "lorenz_x_rho28", "chua_x", "rossler_x_c5.7",
               "rossler_x_c2.5", "lorenz_x_rho10"]

    series = {}
    for nm in targets:
        x, _, expected = flows.generate(nm, n=N_LONG)
        rm = local_maxima(x)
        series[nm] = (rm, expected)
        print(f"{nm:18s} flow n={N_LONG} -> return map n={len(rm)}  ({expected})")

    # A fixed-point attractor yields no local maxima at all; that is itself a
    # decisive answer ("Regular"), so it is reported separately rather than
    # dragging the common length to zero.
    degenerate = {k: v for k, v in series.items() if len(v[0]) < 1000}
    series = {k: v for k, v in series.items() if len(v[0]) >= 1000}
    for k, (rm, expected) in degenerate.items():
        print(f"{k:18s} {expected:8s} return map has {len(rm)} extrema "
              f"-> Regular by construction")

    n_use = min(len(v[0]) for v in series.values())
    n_use = min(n_use, 10000)
    print(f"\nusing n={n_use} for all return maps")
    cal = Calibration(n_use, b_pool=40, b_cal=300)

    print(f"\n{'system':18s} {'truth':8s} {'p_scan':>7s} {'p_IAAFT':>8s} {'dim2':>5s} "
          f"{'atom':>5s}  verdict")
    for nm, (rm, expected) in series.items():
        x = rm[:n_use]
        r = cal.analyze(x)
        if r["p_structure"] >= 0.05:
            verdict, pi = "Noise", float("nan")
        elif r["regularity"]["is_regular"]:
            verdict, pi = "Regular", float("nan")
        else:
            pi = cal.iaaft_p(x, b_boot=99)
            verdict = "Noise" if (np.isnan(pi) or pi >= 0.05) else "Chaos"
        ok = "OK" if verdict == expected else "MISS"
        print(f"{nm:18s} {expected:8s} {r['p_structure']:7.3f} {pi:8.3f} "
              f"{r['dim2_pairs']:5.2f} {r['atomicity_ratio_K200']:5.2f}  {verdict:8s} {ok}",
              flush=True)
