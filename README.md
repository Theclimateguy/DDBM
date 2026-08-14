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

1. **Two domain errors, one principle.** Every operation must be performed in
   the domain where its null is defined. Amplitude preprocessing before ranking
   destroys distribution-freeness (size 0.19 lognormal, 0.88 Cauchy); ranks-only
   restores it with a one-line proof. IAAFT surrogates built from raw amplitudes
   reject a cubed AR(1) — inside their own null — in 100% of replicates;
   generated in normal scores the rate returns to nominal with no loss of power.
2. **A three-line diagnostic.** A monotone transform preserves ranks, so a rank
   statistic must treat AR(1), its cube and its exponential identically. Over
   100 replicates — amplitude domain: 0.02 / **1.00** / **1.00**; normal scores:
   0.05 / 0.05 / 0.05. Conditional heteroscedasticity survives the repair
   (0.15) and needs a model-based null.
3. **A baseline-specification trap.** At `n = 1e4` the missing-ordinal-pattern
   statistic is identically zero under the null for `d ≤ 6`. Above that floor
   the choice of `d` is immaterial.
4. **A benchmark whose scored part is frozen** — all 46 labeled series are
   byte-reproducible from fixed seeds and pinned by SHA-256; the 119 unlabeled
   series come from live sources and are illustrative only.
5. **S&P 500 resolved, not just withdrawn.** Log returns are rejected by an
   IAAFT null (p = 0.02) but not by a fitted GARCH(1,1) null (p = 0.31): the
   apparent structure is conditional heteroscedasticity.

Matched comparison — identical calibration, normal-scores confirmation, decision
rule and auxiliary criteria; 40 labeled series, strict scoring (the six
chaos+noise cases are excluded, not counted correct under either label):

| statistic | accuracy | Chaos | Regular | Noise |
|---|---|---|---|---|
| missing ordinal patterns, `d ∈ {6,7,8}` | **37/40** | 13/14 | 9/10 | 15/16 |
| cyclotomic phase scan | 36/40 | 11/14 | 9/10 | **16/16** |
| pair occupancy `(N_t, ΔN_t)` | 34/40 | 13/14 | 9/10 | 12/16 |

The last two exchange places when the surrogate domain is repaired (it was
occupancy 35, phases 34 with amplitude-domain surrogates), which is itself
evidence that accuracy at a single α on forty series cannot separate close
competitors. These are calibration results on the set used to fix thresholds,
not independent validation.

**No new detection principle is claimed.** The mechanism is the
forbidden/missing-pattern paradigm and coarse-grained transition statistics; the
contribution is the validated inference layer, the measured limits and the
benchmark.

## Layout

```
paper/     v2 manuscript (LaTeX source, bibliography, figure, PDF)
v2/        code and benchmark for v2 — start at v2/README.md
v2/results/ verbatim output of every run behind the manuscript
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

See [`CITATION.cff`](CITATION.cff).

- **Cite the work** (always resolves to the latest version):
  [10.5281/zenodo.18753232](https://doi.org/10.5281/zenodo.18753232)
- **Cite this version** (v2): [10.5281/zenodo.21924169](https://doi.org/10.5281/zenodo.21924169)
- Version 1, whose detector claim is withdrawn:
  [10.5281/zenodo.18753233](https://doi.org/10.5281/zenodo.18753233)

## License

MIT — see [LICENSE](LICENSE).
