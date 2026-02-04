"""
Core DDBM functions: quantization, Diophantine residuals, KS testing
"""

import numpy as np
from scipy import stats


def remove_transients(x, method="none", n_transient=None):
    """Remove transient points from beginning of series"""
    x = np.asarray(x, dtype=float)
    if method == "none":
        return x, 0
    if method == "fixed":
        if n_transient is None:
            raise ValueError("n_transient must be set when method='fixed'")
        n_transient = int(max(0, min(int(n_transient), len(x))))
        return x[n_transient:], n_transient
    raise ValueError("method must be 'none' or 'fixed'")


def rank_normalize_01(x):
    """Rank-normalize time series to [0,1] via empirical CDF"""
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if int(m.sum()) < 2:
        raise ValueError("Too few finite points for rank normalization")
    
    xr = x[m]
    order = np.argsort(xr, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(xr) + 1, dtype=float)
    
    # Handle ties
    xs = xr[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j+1]] = avg
        i = j + 1
    
    u = (ranks - 0.5) / len(xr)
    x01 = np.full_like(x, np.nan, dtype=float)
    x01[m] = u
    return x01


def quantize_timeseries(values, K, dtype=np.int64):
    """Quantize [0,1] values onto integer lattice {0, 1, ..., K}"""
    x = np.asarray(values, dtype=np.float64)
    if x.min() < -0.01 or x.max() > 1.01:
        raise ValueError(f"x must be in [0,1] (tol). got [{x.min():.6f},{x.max():.6f}]")
    return np.floor(x * float(K) + 0.5).astype(dtype, copy=False)


def compute_diophantine_residuals(N):
    """Compute cubic Diophantine residuals from quantized series"""
    N = np.asarray(N)
    if N.ndim != 1 or N.size < 2:
        raise ValueError("N must be 1D length>=2")
    
    N_curr = N[:-1].astype(object)
    dN = np.diff(N.astype(object))
    
    S3 = 3 * (N_curr**2) + 3 * N_curr + 1
    E = (N_curr**3) + (dN**3)
    R = E % S3
    
    Xi = np.asarray(R / S3, dtype=np.float64)
    Xi = Xi[np.isfinite(Xi)]
    return np.mod(Xi, 1.0)


def phases_and_N_from_xraw(x_raw, K):
    """Full pipeline: raw data -> phases"""
    x01 = rank_normalize_01(x_raw)
    N = quantize_timeseries(x01, K)
    Xi = compute_diophantine_residuals(N)
    Xi = np.asarray(Xi, dtype=float)
    Xi = Xi[np.isfinite(Xi)]
    return Xi, N


def gate_diagnostics(x_raw, K, config):
    """Check if K passes hard quality gates"""
    Xi, N = phases_and_N_from_xraw(x_raw, K)
    n_unique = int(np.unique(N).size)
    n_phases = int(Xi.size)
    sd = float(np.std(Xi)) if n_phases else np.nan
    
    ok = (
        (n_unique >= config["Q_MIN_UNIQUE_N"]) and
        (n_phases >= config["Q_MIN_PHASES"]) and
        np.isfinite(sd) and
        (sd >= config["Q_MIN_STD"])
    )
    return dict(ok=bool(ok), n_unique_N=n_unique, n_phases=n_phases, std=sd)


def ks2_newnull(x_raw, K, config):
    """Two-sample KS test: data phases vs. simulated uniform null"""
    rng = np.random.default_rng(config["RNG_SEED"])
    
    Xi_data, _ = phases_and_N_from_xraw(x_raw, K)
    n_data = int(Xi_data.size)
    if n_data == 0:
        raise ValueError("No phases in data")
    
    n = int(len(x_raw))
    Xi_null_list = []
    for _ in range(int(config["NEWNULL_N"])):
        if config["NEWNULL_GEN"] == "u01":
            x_sim = rng.uniform(0.0, 1.0, n)
        elif config["NEWNULL_GEN"] == "normal":
            x_sim = rng.normal(0.0, 1.0, n)
        else:
            raise ValueError("generator must be 'u01' or 'normal'")
        
        Xi_b, _ = phases_and_N_from_xraw(x_sim, K)
        if Xi_b.size:
            Xi_null_list.append(Xi_b)
    
    if not Xi_null_list:
        raise ValueError("Null simulation produced no phases")
    
    if config["NEWNULL_POOL"] == "concat":
        Xi_null = np.concatenate(Xi_null_list)
        res = stats.ks_2samp(Xi_data, Xi_null, alternative="two-sided", method=config["KS2_METHOD"])
        return float(res.statistic), float(res.pvalue), n_data, int(Xi_null.size)
    
    if config["NEWNULL_POOL"] == "match_n":
        Ds = []
        for Xi_b in Xi_null_list:
            res_b = stats.ks_2samp(Xi_data, Xi_b, alternative="two-sided", method=config["KS2_METHOD"])
            Ds.append(float(res_b.statistic))
        D_med = float(np.median(Ds))
        p_rep = float(np.mean(np.asarray(Ds) >= D_med))
        return D_med, p_rep, n_data, None
    
    raise ValueError("pool must be 'concat' or 'match_n'")


def bonferroni(p, m):
    """Bonferroni correction for multiple testing"""
    return min(1.0, float(p) * max(int(m), 1))
