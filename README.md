# DDBM: Diophantine Dynamical Boundary Method

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) 

arXiv link will be provided later with an article release - for now consult DDBM_Methodology.md

**A statistical framework for distinguishing chaotic dynamics from periodic oscillations and noise in time series data.**

## Overview

DDBM detects deterministic chaos by analyzing non-uniform phase distributions in Diophantine residual space after quantization onto integer lattices. The method achieves **100% classification accuracy** on benchmark chaotic systems (logistic map, Hénon, Lorenz) and correctly identifies stylized facts in financial data.

**Key Features:**
- Two-level detection: raw signal + residualized dynamics
- Automatic separation of chaos from periodic/limit-cycle behavior
- Robust to trends, autocorrelation, and heteroscedasticity
- No parameter tuning required (adaptive K-scan with hard gates)
- Fast batch processing for multiple time series

## Quick Start

```python
import numpy as np
from ddbm import analyze_timeseries

# Generate chaotic logistic map
x = [0.1]
for _ in range(5000):
    x.append(4.0 * x[-1] * (1 - x[-1]))

# Analyze
result = analyze_timeseries(x)
print(result['final_status'])  # → "CHAOS_CANDIDATE"
print(result['chaos_candidate'])  # → True
```

## Installation

```bash
pip install numpy scipy pandas
git clone https://github.com/yourusername/DDBM.git
cd DDBM
python setup.py install
```

Or directly from source:
```bash
pip install git+https://github.com/yourusername/DDBM.git
```

## Documentation

- **[API Reference](API.md)** – Function documentation
- **[Usage Examples](examples/)** – Practical applications
- **[Validation Report](VALIDATION.md)** – Benchmark results

## Scientific Background

The mathematical foundation and theoretical proofs are detailed in our arXiv preprint:

> **Diophantine Phase Detection for Time Series Classification**  
> [Nazar Sotiriadi] (2026). arXiv:XXXX.XXXXX  
> https://arxiv.org/abs/XXXX.XXXXX

**Core principle:** Chaotic attractors with non-integer fractal dimension generate quantization artifacts that violate uniformity in modular arithmetic space.

## Citation

If you use DDBM in research, please cite:

```bibtex
@misc{ddbm2026,
  title={Diophantine Dynamical Boundary Method for Chaos Detection},
  author={Nazar Sotiriadi},
  year={2026},
  preprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={nlin.CD}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/DDBM/issues)
- Email: n.sotiriadi@gmail.com
- arXiv: https://arxiv.org/abs/XXXX.XXXXX

---

**Version:** 7.1  
**Last updated:** February 2026
