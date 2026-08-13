# DDBM

A screening test for deterministic structure in univariate time series, and an
open benchmark for evaluating such tests.

> **Status: version 2 supersedes version 1, and withdraws its detector claim.**
> The v1 method is retained in this repository for reference only; its reported
> accuracy came from an uncalibrated procedure and should not be relied upon.
> Current work is on the [`v2` branch](https://github.com/Theclimateguy/DDBM/tree/v2).

## What v2 says

v1 proposed a modular-arithmetic "Diophantine phase" statistic. Three of its
claims do not survive:

- **Not embedding-free.** The phase is a function of the delay pair
  `(N_t, N_{t+1})` with `m = 2`, `τ = 1`, quantized at `1/K`; scanning `K` scans
  box scale.
- **The stated mechanism is self-cancelling.** v1 attributed the signal to a
  singular *marginal* measure occupying few bins; rank normalization removes
  exactly that. The test sees transition geometry.
- **The p-values were invalid** — and not merely because overlapping pairs are
  dependent: with the published order of operations the procedure is not
  distribution-free, reaching empirical size **0.88** at nominal 0.05 on a
  Cauchy marginal.

What v2 contributes instead:

1. **Operation order determines validity.** Amplitude preprocessing before
   ranking destroys distribution-freeness (size 0.19 lognormal, 0.88 Cauchy);
   ranks-only restores it, with a one-line proof and nominal size across six
   marginals.
2. **Measured surrogate limits.** The IAAFT stage rejects a static monotone
   transform of a linear process — inside its own null — in 25% of replicates,
   and heteroscedastic processes in 50–75%. This invalidates the financial
   conclusion of v1.
3. **A baseline-specification trap.** At `n = 1e4` the missing-ordinal-pattern
   statistic is identically zero under the null for `d ≤ 6`.
4. **An open benchmark** of 165 series whose synthetic labels are computed from
   Lyapunov exponents rather than asserted.

Matched comparison — same calibration, same surrogate stage, same decision rule,
preprocessing-free, 40 labeled series, strict scoring:

| statistic | accuracy | empirical size |
|---|---|---|
| missing ordinal patterns, `d ∈ {6,7,8}` | **37/40** | 0.067–0.100 |
| pair occupancy `(N_t, ΔN_t)` | 35/40 | 0.033–0.092 |
| cyclotomic phase scan | 34/40 | 0.025–0.067 |

**No new detection principle is claimed.** The mechanism is the
forbidden/missing-pattern paradigm and coarse-grained transition statistics; the
contribution is the calibrated decision layer, the measured limits and the
benchmark.

## Layout

```
paper/     v2 manuscript (LaTeX source, bibliography, figure, PDF)
v2/        code and benchmark for v2 — start at v2/README.md
src/ddbm/  the v1 library, retained for reference
data/      the v1 benchmark manifest
```

## Quick start

```bash
pip install -r requirements.txt
pip install wfdb nolds sympy matplotlib scipy
cd v2
python build_dataset.py     # downloads real data, regenerates synthetic series
python ordinal_scaled.py    # the matched comparison above
```

Series files are not redistributed; `build_dataset.py` rebuilds them from
primary sources, whose own terms apply.

## Citation

See [`CITATION.cff`](CITATION.cff). The record is
[10.5281/zenodo.18753233](https://doi.org/10.5281/zenodo.18753233); cite version
2 unless you specifically mean the withdrawn v1 claim.

## License

MIT — see [LICENSE](LICENSE).
