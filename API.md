# API Reference

**DDBM v7.1 Function Documentation**

---

## Core Functions

### `analyze_timeseries(x, alpha=0.05, config=None)`

**Main entry point for chaos detection.**

**Parameters:**
- `x` : array-like  
  Univariate time series (length ≥ 500 recommended, ≥ 1000 optimal)
- `alpha` : float, default=0.05  
  Significance level for hypothesis testing (after Bonferroni correction)
- `config` : dict, optional  
  Override default parameters (see Configuration section)

**Returns:**
- `result` : dict  
  Dictionary with keys:
  - `final_status` : str – "CHAOS_CANDIDATE" | "REGULAR_NONCHAOTIC" | "NOT_CHAOS_CANDIDATE" | "DEGENERATE"
  - `chaos_candidate` : bool – True if deterministic chaos detected
  - `n_points_raw` : int – Valid data points in raw signal
  - `n_points_resid` : int – Valid points after residualization
  - `cheap` : dict – Preprocessing diagnostics (Ljung-Box, trend, AR1)
  - `regularity` : dict – Spectral/complexity metrics (FFT, ACF, PE)
  - `ddbm_raw` : dict – Level 1 analysis results
  - `ddbm_resid` : dict – Level 2 analysis results

**Example:**
```python
import numpy as np
from ddbm import analyze_timeseries

# Generate Lorenz attractor (chaotic)
from scipy.integrate import odeint

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

t = np.linspace(0, 100, 10000)
xyz = odeint(lorenz, [1, 1, 1], t)
x = xyz[:, 0]  # Extract x-coordinate

result = analyze_timeseries(x)
print(result['final_status'])  # → "CHAOS_CANDIDATE"
print(result['ddbm_resid']['optimal']['K_opt'])  # → optimal K
```

---

### `batch_analyze(data_dir, output_dir, pattern="*.csv", value_col=-1)`

**Process multiple CSV files in batch mode.**

**Parameters:**
- `data_dir` : str or Path  
  Directory containing CSV files
- `output_dir` : str or Path  
  Directory for JSON results and summary CSV
- `pattern` : str, default="*.csv"  
  File pattern for glob matching
- `value_col` : int, default=-1  
  Column index for time series data (negative = from end)

**Returns:**
- `summary` : pandas.DataFrame  
  Aggregated results with columns:
  - `file`, `final_status`, `chaos_candidate`, `n_raw`, `n_resid`
  - `raw_K_opt`, `raw_D`, `raw_p_adj` (Level 1 optimal results)
  - `resid_K_opt`, `resid_D`, `resid_p_adj` (Level 2 optimal results)
  - `reg_is_regular`, `reg_fft_conc`, `reg_acf_max`, `reg_perm_entropy`

**Side effects:**
- Creates JSON file per input CSV in `output_dir`
- Writes `batch_summary.csv` with flattened results

**Example:**
```python
from ddbm import batch_analyze

summary = batch_analyze(
    data_dir="./timeseries_data",
    output_dir="./results",
    pattern="*.csv"
)

# Filter chaos candidates
chaos = summary[summary['chaos_candidate'] == True]
print(chaos[['file', 'resid_K_opt', 'resid_D', 'resid_p_adj']])
```

---

## Utility Functions

### `rank_normalize(x)`

**Rank-transform to [0,1] via empirical CDF.**

**Parameters:**
- `x` : array-like  
  Raw time series (NaN/Inf handled)

**Returns:**
- `u` : ndarray  
  Rank-normalized values in [0, 1]

---

### `compute_phases(x, K)`

**Compute Diophantine phases for given quantization scale.**

**Parameters:**
- `x` : array-like  
  Raw time series
- `K` : int  
  Quantization parameter (10 ≤ K ≤ 1000 typical)

**Returns:**
- `Xi` : ndarray  
  Diophantine phases in [0, 1)
- `N` : ndarray  
  Quantized lattice coordinates

---

### `ks_test_phases(Xi, n_null=200, seed=42)`

**Two-sample KS test against simulated uniform null.**

**Parameters:**
- `Xi` : array-like  
  Empirical phases
- `n_null` : int, default=200  
  Number of null simulations
- `seed` : int, default=42  
  Random seed for reproducibility

**Returns:**
- `D` : float  
  KS statistic
- `p` : float  
  p-value (raw, before Bonferroni correction)

---

## Configuration

**Default parameters (override via `config` dict):**

```python
DEFAULT_CONFIG = {
    # K scan
    'K_MIN': 10,
    'K_MAX': 1000,
    'COARSE_STEP': 50,
    'FINE_STEP': 5,

    # Decision thresholds
    'ALPHA': 0.05,
    'NEWNULL_N': 200,

    # Hard gates (quality control)
    'Q_MIN_UNIQUE_N': 20,
    'Q_MIN_PHASES': 500,
    'Q_MIN_STD': 1e-6,

    # Residualization
    'LB_LAGS': 20,
    'TREND_MIN_N': 50,
    'AR1_CLIP': 0.99,
    'ROLLVOL_WIN': 200,
    'ROLLVOL_POLICY': 'auto',  # 'off' / 'auto' / 'on'

    # Regularity gate
    'REG_FFT_TOPK': 3,
    'REG_FFT_CONC_THR': 0.60,
    'REG_ACF_MAX_LAG': 200,
    'REG_ACF_MAX_THR': 0.85,
    'REG_PE_M': 5,
    'REG_PE_OVERRIDE_HI': 0.70,
}
```

**Custom configuration example:**
```python
from ddbm import analyze_timeseries

config = {
    'ALPHA': 0.01,  # Stricter threshold
    'K_MAX': 500,   # Reduce K range for speed
    'ROLLVOL_POLICY': 'off'  # Disable volatility standardization
}

result = analyze_timeseries(x, config=config)
```

---

## Output Schema

### Result Dictionary Structure

```python
{
  'final_status': str,  # "CHAOS_CANDIDATE" | "REGULAR_NONCHAOTIC" | etc.
  'chaos_candidate': bool,
  'n_points_raw': int,
  'n_points_resid': int,

  'cheap': {
    'lb_p_x': float,      # Ljung-Box p-value on raw series
    'lb_p_x2': float,     # Ljung-Box p-value on squared series (ARCH test)
    'trend_beta': float,  # OLS trend coefficient
    'trend_p': float,     # Trend significance
    'ar1_phi': float,     # AR(1) coefficient
    'roll_applied': bool, # Volatility standardization applied?
    'roll_reason': str    # Reason for rollvol decision
  },

  'regularity': {  # None if resid_status != "STRUCTURED"
    'fft_conc_topk': float,  # Frequency concentration (0-1)
    'acf_max_abs': float,    # Max autocorrelation lag 2-200
    'perm_entropy': float,   # Normalized permutation entropy
    'is_regular': bool,      # Regularity gate decision
    'reason': str            # Triggering condition
  },

  'ddbm_raw': {
    'status': str,  # "STRUCTURED" | "NOISE"
    'optimal': {
      'K_opt': int,
      'D': float,          # KS statistic
      'p': float,          # Raw p-value
      'p_adj_bonf': float, # Bonferroni-corrected p-value
      'm_tested': int      # Number of K values tested
    },
    'diagnostics': {...}   # Gate failure counts, etc.
  },

  'ddbm_resid': {
    'status': str,  # "STRUCTURED" | "NOISE" | "STRUCTURED_DEGENERATE"
    'optimal': {...},  # Same structure as ddbm_raw
    'diagnostics': {...}
  }
}
```

---

## Error Handling

**Raised exceptions:**
- `ValueError` : Input validation failures
  - Series too short (< 500 points)
  - All values NaN/Inf
  - Invalid config parameters
- `RuntimeError` : Computational failures
  - All K fail hard gates
  - Null simulation produces no phases

**Example:**
```python
from ddbm import analyze_timeseries

try:
    result = analyze_timeseries(x)
except ValueError as e:
    print(f"Input error: {e}")
except RuntimeError as e:
    print(f"Analysis failed: {e}")
```

---

## Dependencies

**Required:**
- numpy >= 1.20
- scipy >= 1.7
- pandas >= 1.3

**Optional:**
- matplotlib (for visualization utilities)

---

**Version:** 7.1  
**Last updated:** February 2026
