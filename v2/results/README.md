# Raw output of every run behind the manuscript

Verbatim stdout, so the numbers in the paper can be checked without re-running
anything. Each file names the script that produced it and the table it backs.

| log | produced by | backs |
|---|---|---|
| `01_pivotality_and_audit.log` | `reviewer_experiments.py` | Table 1 (size by marginal, $B{=}1000$), the 8-replicate IAAFT audit, the first three-way comparison, Lyapunov intervals near $r=3.57$ |
| `02_operation_orders.log` | `pivotality_fix.py` | Table 2 (four orders of operations) |
| `03_surrogate_domain_raw_vs_rank.log` | `surrogate_domain.py` | Table 3, amplitude and rank columns, with spectral-mismatch diagnostics |
| `04_surrogate_normalscores.log` | `surrogate_normalscores.py` | Table 3, normal-scores column (30 replicates) |
| `05_size_study_100reps.log` | `size_study_100.py` | Table 3 as printed: 100 replicates with Wilson intervals |
| `06_sp500_patternlength_audit.log` | `essential_revisions.py` | the S&P 500 case under two nulls, the pattern-length sensitivity, the 50-replicate audit |
| `07_ordinal_saturation_sensitivity.log` | `ordinal_scaled.py` | the saturation table and the corrected ordinal baseline |
| `08_rank_only_comparison.log` | `rank_only_pair.py` | the preprocessing-free comparison with amplitude-domain surrogates |
| `09_final_benchmark.log` | `final_benchmark.py` | Tables 4–5 (final configuration) |
| `00_superseded_saturated_baseline.log` | `compare_ordinal.py` | **superseded.** The draft comparison that used $d\in\{4,5,6\}$ at $n=10^4$, where the statistic is identically zero under the null. Kept because the paper discusses this trap; the numbers in it are not valid. |

## Two things visible here that the paper reports against itself

**The 30-replicate table was wrong about GARCH.** `04` reads `garch_returns
0.00` in normal scores; `05`, at 100 replicates, reads `0.15 [0.09, 0.23]`. The
larger run overturns the smaller one, and the paper reports the corrected value.

**The first ordinal baseline was degenerate.** `00` reports an accuracy for a
statistic whose null variance is zero (see the saturation table in `07`). It is
retained as evidence for the trap described in the manuscript, not as a result.

## Derived tables

Machine-readable results are in `../data_bench/`: `final_benchmark_results.csv`
(per-series verdicts in the final configuration), `ordinal_scaled_results.csv`,
`rank_only_pair_results.csv`, `reviewer_e4_results.csv`, `results.csv`,
plus `checksums.csv` pinning every series.
