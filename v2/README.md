# v2 — a calibrated, preprocessing-free rank test

Code and benchmark for version 2 of this work. The manuscript is in
[`../paper/`](../paper).

## What v2 concludes

**The v1 detector claim is withdrawn.** Three of its claims do not survive:

- It is **not embedding-free.** The phase is a function of the pair
  `(N_t, N_{t+1})` — a two-dimensional delay vector with `m = 2`, `τ = 1`,
  quantized at resolution `1/K`. Scanning `K` scans box scale.
- **The stated mechanism contradicts its own preprocessing.** v1 attributed the
  signal to a singular *marginal* measure occupying few bins, but rank
  normalization removes exactly that. What the test sees is the geometry of
  *transitions*.
- **The p-values were not valid** — and worse than a dependence artifact: with
  the published order of operations the procedure is not distribution-free at
  all (see below).

**The cyclotomic identity is real but useless.** `S_p(N) = (N+1)^p − N^p`
factors into homogeneous cyclotomic polynomials at consecutive integers, so
`(N+1)/N` is an exact p-th root of unity mod `S_p` and the phase lattice has
denominator `Φ_n(1)`. Since the phase is a deterministic function of
`(N, ΔN)`, the map is a lossy projection: a plain occupancy count on that pair
beats it.

## What v2 contributes

1. **Operation order determines validity.** Detrending, AR(1) prewhitening and
   volatility standardization applied *before* ranking destroy
   distribution-freeness: empirical size at nominal 0.05 reaches **0.19**
   (lognormal) and **0.85** (Cauchy). Ranks-only restores it, with a one-line
   proof, and gives nominal size across six marginals.
2. **Measured surrogate limits.** The IAAFT confirmation stage rejects a static
   monotone transform of a linear process — inside its own null — in **25%** of
   replicates, and heteroscedastic processes in **50–75%**. This invalidates the
   financial conclusion of v1.
3. **A baseline-specification trap.** At `n = 1e4` the expected fraction of
   unobserved ordinal patterns is 0.0000 for `d ≤ 6`, 0.138 for `d = 7`, 0.781
   for `d = 8`. At `d ≤ 6` the statistic is identically zero under the null and
   any standardized score is meaningless.
4. **An open benchmark** whose synthetic labels are computed from Lyapunov
   exponents rather than asserted.

## Matched comparison (40 labeled series, strict scoring, preprocessing-free)

| statistic | accuracy | empirical size range |
|---|---|---|
| missing ordinal patterns, d ∈ {6,7,8} | **37/40 = 92.5%** | 0.067–0.100 |
| pair occupancy `(N_t, ΔN_t)` | 35/40 = 87.5% | 0.033–0.092 |
| cyclotomic phase scan | 34/40 = 85.0% | 0.025–0.067 |

No new detection principle is claimed: the mechanism is the forbidden/missing
pattern paradigm and coarse-grained transition statistics.

## Files

```
passport.py            calibration, cyclotomic + power kernels, surrogate stages
rank_only_pair.py      preprocessing-free comparison (paper Tables 4, 5)
ordinal_scaled.py      same with the pattern length scaled to n; saturation report
pivotality_fix.py      four orders of operations (paper Table 3)
reviewer_experiments.py  size by marginal, IAAFT audit, Lyapunov intervals
                         (paper Tables 2, 6)
cyclotomic_kernels.py  symbolic derivation of the cyclotomic forms and lattices
flows.py               Benettin-verified chaotic flow generators
return_map.py          return-map reduction for continuous flows
build_dataset.py       regenerates / downloads all 165 series and the manifest
make_figures.py        manuscript figure
run_all.py             the superseded pipeline, retained for the auxiliary
                       spectral criteria and the real-data appendix numbers
data_bench/            manifest with provenance and labels, plus result tables
```

## Reproducing

```bash
pip install -r ../requirements.txt
pip install wfdb nolds sympy matplotlib scipy
cd v2
python build_dataset.py       # downloads real data, regenerates synthetic series
python pivotality_fix.py      # Table 3
python reviewer_experiments.py  # Tables 2 and 6, Lyapunov intervals
python ordinal_scaled.py      # Tables 4 and 5, saturation report
```

Scripts import the `ddbm` package from this repository's `src/` tree; no install
step is needed for that. Seeds are fixed throughout.

## Data

The series themselves are **not** redistributed. Synthetic ones are regenerated
deterministically; real ones are downloaded by `build_dataset.py` from Yahoo
Finance, the UK Met Office (HadCET), NOAA CPC and PSL, PhysioNet and the Bonn
EEG archive, whose own terms apply.

## Known limits

- Surrogate tests detect **nonlinearity, not determinism**: GARCH volatility is
  misread as deterministic.
- **Instrumental quantization** manufactures transition sparsity; controlled by
  referencing against a shuffle of the same values, and flagged.
- **Densely sampled flows** need the return-map reduction, which is validated
  in-sample only.
- Dropping preprocessing costs power: AR(1) with φ = 0.9 becomes a false
  positive for the pair statistic.
- Reliable operation needs n ≳ 1000 and SNR ≳ 10 dB.
