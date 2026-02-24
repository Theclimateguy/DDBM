# DDBM: Diophantine Dynamical Boundary Method

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18753233-blue)](https://doi.org/10.5281/zenodo.18753233)

DDBM is a statistical method for distinguishing deterministic chaotic dynamics from periodic/quasi-periodic oscillations and noise in **univariate** time series, without phase-space reconstruction or training data.

Repository: https://github.com/Theclimateguy/DDBM

## Motivation

In climate science, finance, and other complex systems, a practical question precedes modeling: does the observed signal contain deterministic structure (potentially exploitable for prediction/control), or is it indistinguishable from a stochastic process after removing trends and linear dependence?

Many chaos diagnostics require embedding choices, long sample sizes, or can be sensitive to colored noise. DDBM targets a conservative, hypothesis-testing-based screening workflow on scalar observables.

## Core idea (Diophantine lattice resonance)

Chaotic attractors often carry singular (fractal) invariant measures. When a scalar observable is rank-normalized to $[0,1]$ and quantized onto an integer lattice, the trajectory can occupy a sparse subset of bins at certain resolutions.

DDBM amplifies this sparsity via modular arithmetic, producing a “Diophantine phase” whose empirical distribution deviates from $\mathrm{Uniform}(0,1)$ for resonant quantization scales in deterministic chaos, while i.i.d. noise remains asymptotically equidistributed.

## Method (high level)

Given a time series $x_1,\dots,x_n$:

1) **Two-level preprocessing**
- Level 1 (raw): rank/quantile normalization to $[0,1]$.
- Level 2 (residual): linear detrending, then AR(1) prewhitening; optionally standardize conditional volatility if ARCH effects are detected (Ljung–Box on squared residuals).
Classification requires structure detection at the residual level to reduce confounding by trends/linear dynamics.

2) **Lattice quantization**
Use a two-stage scan over $K \in [10,1000]$: a coarse grid (step 50), then a fine local search (step 5) around the best coarse scale. Map $u_t\in[0,1]$ to integers $N_t=\lfloor Ku_t+0.5\rfloor.$

3) **Diophantine phase construction**
Using the cubic kernel $S_3(N)=3N^2+3N+1$ and increments $\Delta N_t=N_{t+1}-N_t$, compute a normalized modular phase $\Xi_t\in[0,1)$.

4) **Uniformity testing across $K$**
Test $H_0:\ \Xi_t \sim \mathrm{Uniform}(0,1)$ using a KS-based goodness-of-fit procedure with multiple-testing correction over scanned $K$ values; rejection implies “structured”.

5) **Regularity filter**
After “structured” is detected, separate chaos vs regular dynamics using spectral concentration and autocorrelation persistence (plus a high permutation-entropy gate), yielding labels like CHAOS vs REGULAR; otherwise classify as NOISE/NOT-CHAOS-CANDIDATE.

## Validation (benchmark summary)

On 40 benchmark systems spanning canonical chaotic attractors (Lorenz, Hénon, Rössler, Chua, logistic map), periodic/quasi-periodic dynamics, and stochastic processes (i.i.d. noise, AR, GARCH, random walk, plus financial data), the current report is **92.5% overall accuracy (37/40)**.

In that benchmark set, i.i.d. Gaussian white noise produced zero false positives, while observed misclassifications concentrate at boundary cases: quasi-periodic irrational rotations (circle map), very low SNR chaos+noise mixtures (~5 dB), and an IAAFT surrogate artifact.

## Reproducible batch harness (v7.2)

This repository includes a standalone reproducible runner:

- `ddbm_batch_runner_v7_2.py`
- `data/test/benchmark_manifest_v7_1.csv`

What it adds:

- manifest-defined 40-case benchmark scoring (including explicit `Mixed` scoring rule),
- resume support (reuses existing per-file JSON outputs),
- persistent null-pool cache under `results_test/null_cache/`,
- explicit benchmark report outputs.

Run:

```bash
python3 ddbm_batch_runner_v7_2.py
```

Outputs:

- `results_test/batch_summary_twolevel.csv`
- `results_test/benchmark_report_v7_1.json`
- `results_test/benchmark_category_summary.csv`

## Quick start (API-style example)

```python
from ddbm import analyze_timeseries

# Logistic map (chaotic regime)
x = [0.1]
for _ in range(5000):
    x.append(4.0 * x[-1] * (1 - x[-1]))

result = analyze_timeseries(x)
print(result["final_status"])
```


## Installation

Clone and install locally:

```bash
git clone https://github.com/Theclimateguy/DDBM.git
cd DDBM
python setup.py install
```

(If you publish a PyPI package later, add the `pip install ddbm` line here.) [file:2]

## Manuscript / preprint

Manuscript prepared for arXiv submission (identifier will be added after upload): 

**Diophantine Lattice Resonance for Chaos Detection in Time Series**
Sotiriadi Nazar (2026). arXiv: TBA. 

## Citation

```bibtex
@misc{ddbm2026,
  title={Diophantine Lattice Resonance for Chaos Detection in Time Series},
  author={Sotiriadi, Nazar},
  year={2026},
  archivePrefix={arXiv},
  primaryClass={nlin.CD},
  eprint={TBA}
}
```


## License

MIT License. See LICENSE for details. [file:2]

## Contact

- Issues: GitHub Issues
- Email: n.sotiriadi@gmail.com [file:2]

---

Version: 7.2
Last updated: February 2026 [file:2]
