# v2 — code and benchmark

Everything here supports the manuscript in [`../paper/`](../paper), which is the
authoritative description. This file is a map, not a summary.

## Recommended procedure

One principle governs both stages: **perform every operation in the domain where
its null is defined.** Violating it in either place destroys validity silently.

1. **Symbolize** — rank-normalize to `[0,1]`, average ranks for ties. No
   detrending, no prewhitening, no volatility scaling: any amplitude operation
   before ranking destroys distribution-freeness (size reaches 0.88 at nominal
   0.05 on a Cauchy marginal).
2. **Screen** — statistic on a grid, each cell standardized by a Monte-Carlo
   null, maximum taken; `p = (1 + #{T_null ≥ T_obs}) / (1 + B)`. Scan
   multiplicity sits inside the null, so no Bonferroni.
3. **Confirm** — IAAFT surrogates generated in **normal scores**,
   `Φ⁻¹(rank/(n+1))`, which is the domain the AAFT null is stated in. Built from
   raw amplitudes they reject a cubed AR(1) — inside their own null — in 100% of
   replicates. For heteroscedastic data prefer a model-based null.
4. **Classify** — extremely narrowband ⇒ *(quasi-)periodic*; otherwise
   *confirmed nonlinear*, split by transition occupancy. Not "chaos": no
   surrogate test establishes determinism.

**Diagnostic worth running always.** A static monotone transform preserves ranks,
so a rank statistic must give AR(1), its cube and its exponential the same
rejection rate. Amplitude domain: 0.03 / 1.00 / 1.00. Rank domain: 0.47 / 0.47 /
0.47. Normal scores: 0.07 / 0.07 / 0.07. Agreement is the signature of a
correctly specified test; disagreement is proof of misspecification.

**Pattern length has a floor, not a tuning knob.** At `n = 1e4` the expected
fraction of unobserved ordinal patterns is 0.000 for `d ≤ 6`, 0.138 for `d = 7`,
0.781 for `d = 8`. A fully saturated grid gives a statistic identically zero
under the null. Above that floor the choice is immaterial: 37/40 for every grid
tried.

## Matched comparison (40 labeled series, strict scoring, final configuration)

| statistic | accuracy | Chaos | Regular | Noise |
|---|---|---|---|---|
| missing ordinal patterns, `d ∈ {6,7,8}` | **37/40** | 13/14 | 9/10 | 15/16 |
| cyclotomic phase scan | 36/40 | 11/14 | 9/10 | **16/16** |
| pair occupancy `(N_t, ΔN_t)` | 34/40 | 13/14 | 9/10 | 12/16 |

The last two **exchange places** when the surrogate domain is repaired (with
amplitude-domain surrogates it was occupancy 35, phases 34). A ranking that flips
when an unrelated stage is fixed is not a ranking — accuracy at a single α on
forty series cannot separate close competitors. These are calibration results on
the set used to fix thresholds, not independent validation.

## Files

| file | reproduces |
|---|---|
| `passport.py` | calibration, kernels, surrogate stages (imported by the rest) |
| `final_benchmark.py` | the matched comparison above (paper Tables 4–5) |
| `surrogate_normalscores.py` | the normal-scores repair (paper Table 3) |
| `surrogate_domain.py` | the amplitude/rank domain failures (paper Table 3) |
| `pivotality_fix.py` | four orders of operations (paper Table 2) |
| `essential_revisions.py` | surrogate-size audit, pattern-length study, S&P case |
| `ordinal_scaled.py` | saturation report |
| `rank_only_pair.py` | preprocessing-free harness used by the above |
| `hash_manifest.py` | SHA-256 for every series; `check` verifies a rebuild |
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
python hash_manifest.py check     # verify the regenerated series
python pivotality_fix.py          # screening-stage domain study
python surrogate_normalscores.py  # confirmation-stage repair
python essential_revisions.py     # surrogate audit, pattern length, S&P
python final_benchmark.py         # the matched comparison
```

Scripts pick up the `ddbm` package from `../src/`; no install needed. Seeds are
fixed. Series files are not redistributed — `build_dataset.py` rebuilds them
from Yahoo Finance, the UK Met Office, NOAA, PhysioNet and the Bonn EEG archive,
whose own terms apply.

## Limits (paper §6)

Surrogate tests detect nonlinearity, not determinism — GARCH volatility is
misread. Instrumental quantization manufactures transition sparsity; controlled
by referencing against a shuffle of the same values, and flagged. Densely
sampled flows need the return-map reduction, validated in-sample only. IAAFT
assumes stationarity, so random walks and long-memory series are false positives.
Dropping preprocessing costs specificity on strongly autocorrelated processes.
Needs `n ≳ 1000` and SNR `≳ 10 dB`.
