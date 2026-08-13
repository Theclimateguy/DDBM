# v2 — code and benchmark

Everything here supports the manuscript in [`../paper/`](../paper), which is the
authoritative description. This file is a map, not a summary.

## Recommended procedure

1. **Symbolize** — rank-normalize to `[0,1]`, average ranks for ties. No
   detrending, no prewhitening, no volatility scaling: any amplitude operation
   before ranking destroys distribution-freeness (paper §3).
2. **Screen** — statistic on a grid, each cell standardized by a Monte-Carlo
   null, maximum taken; `p = (1 + #{T_null ≥ T_obs}) / (1 + B)`. Scan
   multiplicity is inside the null, so no Bonferroni.
3. **Confirm** — IAAFT surrogates, which probe temporal ordering.
4. **Classify** — extremely narrowband ⇒ *(quasi-)periodic*; otherwise
   *confirmed nonlinear*, split by transition occupancy into *compatible with
   low-dimensional determinism* and *nonlinear stochastic*. Not "chaos": no
   surrogate test establishes determinism.

**Scale the pattern length to the sample.** At `n = 1e4` the expected fraction
of unobserved ordinal patterns is 0.000 for `d ≤ 6`, 0.138 for `d = 7`, 0.781
for `d = 8`. At `d ≤ 6` the statistic is identically zero under the null.

## Files

| file | reproduces |
|---|---|
| `passport.py` | calibration, kernels, surrogate stages (imported by the rest) |
| `ordinal_scaled.py` | Tables 4–5 and the saturation report |
| `rank_only_pair.py` | the preprocessing-free comparison |
| `pivotality_fix.py` | Table 3 — four orders of operations |
| `reviewer_experiments.py` | Tables 2 and 6, Lyapunov intervals |
| `cyclotomic_kernels.py` | §2 symbolic derivations |
| `flows.py` | Benettin-verified flow generators |
| `return_map.py` | flow reduction (§6.3) |
| `build_dataset.py` | all 165 series and the manifest |
| `make_figures.py` | Figure 1 |
| `run_all.py` | the superseded pipeline, kept for the auxiliary spectral criteria |
| `data_bench/` | manifest with provenance and labels, plus result tables |

## Reproducing

```bash
pip install -r ../requirements.txt
pip install wfdb nolds sympy matplotlib scipy
cd v2
python build_dataset.py
python pivotality_fix.py        # Table 3
python reviewer_experiments.py  # Tables 2, 6
python ordinal_scaled.py        # Tables 4, 5
```

Scripts pick up the `ddbm` package from `../src/`; no install needed. Seeds are
fixed. Series files are not redistributed — `build_dataset.py` rebuilds them
from Yahoo Finance, the UK Met Office, NOAA, PhysioNet and the Bonn EEG archive,
whose own terms apply.

## Limits (paper §6)

Surrogate tests detect nonlinearity, not determinism — GARCH volatility is
misread. Instrumental quantization manufactures transition sparsity; controlled
by referencing against a shuffle of the same values, and flagged. Densely
sampled flows need the return-map reduction, validated in-sample only. Dropping
preprocessing costs power: AR(1) at `φ = 0.9` becomes a false positive for the
pair statistic. Needs `n ≳ 1000` and SNR `≳ 10 dB`.
