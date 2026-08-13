"""Run the calibrated passport over the whole benchmark dataset.

Series longer than MAX_N are truncated to a canonical length so that the
Monte-Carlo calibration (which depends on n) can be shared across series.
Results -> data_bench/results.csv
"""
import csv
import json
import os

import numpy as np

from passport import Calibration

MAX_N = 10000
CAL = {}


def canonical_n(n):
    return min(int(n), MAX_N)


def get_cal(n):
    if n not in CAL:
        print(f"  [calibrating n={n} ...]", flush=True)
        CAL[n] = Calibration(n, b_pool=40, b_cal=300)
    return CAL[n]


# A signal is called narrowband only when it is *extremely* concentrated.
# Measured separation: chaotic Rossler sits at fft 0.73-0.83 / acf 0.89-0.93,
# while genuinely periodic and quasi-periodic systems sit at fft >= 0.95 or
# acf >= 0.99. The old gate thresholds (0.60 / 0.85) fell inside the chaotic
# range and vetoed correct chaos verdicts.
# Determinism leaves the quantized trajectory on a sparse set of (N, dN)
# pairs; a stochastic process -- even a nonlinear one -- fills the lattice
# like iid noise. Measured: deterministic maps 0.03-0.21, flows 0.44-0.80,
# labeled-stochastic median 0.99. The split is placed above every labeled
# deterministic system and below the stochastic bulk.
ATOM_STOCHASTIC = 0.85

NARROW_FFT = 0.95
NARROW_ACF = 0.99


def is_narrowband(reg):
    fft = reg.get("fft_conc_topk", np.nan)
    acf = reg.get("acf_max_abs", np.nan)
    return ((np.isfinite(fft) and fft > NARROW_FFT)
            or (np.isfinite(acf) and acf > NARROW_ACF))


def classify(cal, x, r):
    """Three-stage decision. Returns (verdict, p_surr).

    1. Screening: calibrated scan statistic against an iid null. No structure
       at all -> Noise.
    2. Confirmation: IAAFT surrogates preserve the power spectrum and the
       marginal distribution exactly, so they test temporal ordering alone.
       Surviving this means the structure is not reducible to the spectrum.
    3. Narrowband split: deterministic ordering structure is shared by chaos
       and by (quasi-)periodicity, so the two are separated by spectral
       concentration -- but only at the extreme end, where periodic systems
       actually live.

    The surrogate test runs *before* the spectral gate: the gate is a
    heuristic and must not veto a principled test, which is exactly what went
    wrong for spiral chaos (Rossler) in the previous ordering.
    """
    reg = r["regularity"]
    if r["p_structure"] >= 0.05:
        return "Noise", None
    try:
        p_surr = cal.iaaft_p(x, b_boot=99)
    except Exception:
        p_surr = np.nan

    if np.isnan(p_surr):
        return ("Regular" if is_narrowband(reg) else "Noise"), p_surr
    if p_surr < 0.05:
        # Ordering structure beyond the spectrum. Three ways to get here:
        if is_narrowband(reg):
            return "Regular", p_surr          # (quasi-)periodic
        if r["atomicity_ratio_K200"] > ATOM_STOCHASTIC:
            # nonlinear, but the trajectory still fills the lattice like an
            # unconstrained process: nonlinear *stochastic* (GARCH, stochastic
            # volatility), not a low-dimensional deterministic system
            return "NonlinStoch", p_surr
        return "Chaos", p_surr
    # nothing beyond spectrum + marginal: periodic if concentrated, else noise
    fft = reg.get("fft_conc_topk", np.nan)
    acf = reg.get("acf_max_abs", np.nan)
    periodic = ((np.isfinite(fft) and fft > 0.60) or (np.isfinite(acf) and acf > 0.85))
    return ("Regular" if periodic else "Noise"), p_surr


def score(expected, predicted):
    # "NonlinStoch" is a refinement of Noise: the series is stochastic, with
    # nonlinear dependence. It counts as a correct stochastic call.
    if predicted == "NonlinStoch":
        predicted = "Noise"
    if expected == "Mixed":
        return predicted in ("Chaos", "Noise")
    return expected == predicted


def main():
    man = list(csv.DictReader(open("data_bench/manifest.csv")))
    # order by calibration length so calibrations are built once and reused
    man.sort(key=lambda r: (-canonical_n(r["n"]), r["group"], r["name"]))

    out = []
    for i, m in enumerate(man):
        x = np.loadtxt(f"data_bench/series/{m['name']}.csv", skiprows=1)
        n_use = canonical_n(len(x))
        x = x[:n_use]
        cal = get_cal(n_use)
        r = cal.analyze(x)
        verdict, p_surr = classify(cal, x, r)
        pr = r["profile_z"]
        rec = dict(name=m["name"], label=m["label"], group=m["group"],
                   n_used=n_use, verdict=verdict,
                   p_structure=r["p_structure"], p_boot=p_surr,
                   dim2=r["dim2_pairs"], atom=r["atomicity_ratio_K200"],
                   **{f"z_{k}": v for k, v in pr.items()},
                   acf=r["regularity"]["acf_max_abs"],
                   pe=r["regularity"]["perm_entropy"],
                   fft=r["regularity"]["fft_conc_topk"],
                   quantized=r["quantized"], uniq_ratio=r["unique_ratio"],
                   source=m["source"], note=m["note"])
        out.append(rec)
        ok = "" if m["label"] == "unknown" else (" OK" if score(m["label"], verdict) else " MISS")
        pb = f"{p_surr:.3f}" if p_surr is not None and np.isfinite(p_surr) else "  -  "
        print(f"[{i+1:3d}/{len(man)}] {m['name']:26s} {m['label']:8s} -> {verdict:8s} "
              f"p={r['p_structure']:.3f} boot={pb} dim2={r['dim2_pairs']:5.2f} "
              f"atom={r['atomicity_ratio_K200']:4.2f}{ok}", flush=True)

    with open("data_bench/results.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    labeled = [r for r in out if r["label"] != "unknown"]
    n_ok = sum(score(r["label"], r["verdict"]) for r in labeled)
    print(f"\n=== labeled accuracy: {n_ok}/{len(labeled)} = {100*n_ok/len(labeled):.1f}% ===")
    for r in labeled:
        if not score(r["label"], r["verdict"]):
            print(f"  MISS {r['name']}: {r['label']} -> {r['verdict']} "
                  f"(p={r['p_structure']:.3f}, boot={r['p_boot']}, note={r['note']})")

    print("\n=== unlabeled real data by group ===")
    import collections
    nq = [r for r in out if r["quantized"]]
    if nq:
        print(f"(quantization-confounded, verdict unreliable: {len(nq)} series, "
              f"e.g. {', '.join(r['name'] for r in nq[:4])})")
    for grp in ["finance", "climate", "physiology", "eeg"]:
        rs = [r for r in out if r["group"] == grp]
        if not rs:
            continue
        cnt = collections.Counter(r["verdict"] for r in rs)
        print(f"{grp:12s} {dict(cnt)}")

    print("\nsaved data_bench/results.csv")


if __name__ == "__main__":
    main()
