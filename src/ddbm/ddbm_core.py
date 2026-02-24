"""
Core DDBM functions: quantization, Diophantine residuals, KS testing.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats


_NULL_CACHE = OrderedDict()


def remove_transients(x, method="none", n_transient=None):
    """Remove transient points from beginning of series."""
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
    """Rank-normalize time series to [0,1] via empirical CDF."""
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if int(m.sum()) < 2:
        raise ValueError("Too few finite points for rank normalization")

    xr = x[m]
    order = np.argsort(xr, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(xr) + 1, dtype=float)

    # Tie handling via averaged ranks.
    xs = xr[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1

    u = (ranks - 0.5) / len(xr)
    x01 = np.full_like(x, np.nan, dtype=float)
    x01[m] = u
    return x01


def quantize_timeseries(values, K, dtype=np.int64):
    """Quantize [0,1] values onto integer lattice {0, 1, ..., K}."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return np.asarray([], dtype=dtype)
    if x.min() < -0.01 or x.max() > 1.01:
        raise ValueError(f"x must be in [0,1] (tol). got [{x.min():.6f},{x.max():.6f}]")
    return np.floor(x * float(K) + 0.5).astype(dtype, copy=False)


def compute_diophantine_residuals(N):
    """Compute cubic Diophantine residuals from quantized series."""
    N = np.asarray(N, dtype=np.int64)
    if N.ndim != 1 or N.size < 2:
        raise ValueError("N must be 1D length>=2")

    N_curr = N[:-1]
    dN = np.diff(N)
    S3 = 3 * (N_curr**2) + 3 * N_curr + 1
    E = (N_curr**3) + (dN**3)
    R = E % S3

    Xi = np.asarray(R / S3, dtype=np.float64)
    Xi = Xi[np.isfinite(Xi)]
    return np.mod(Xi, 1.0)


def phases_and_N_from_xraw(x_raw, K):
    """Full pipeline: raw data -> phases."""
    x01 = rank_normalize_01(x_raw)
    N = quantize_timeseries(x01, K)
    Xi = compute_diophantine_residuals(N)
    Xi = np.asarray(Xi, dtype=float)
    Xi = Xi[np.isfinite(Xi)]
    return Xi, N


def phases_and_gate_from_xraw(x_raw, K, config):
    """Compute phases and hard-gate diagnostics in a single pass."""
    Xi, N = phases_and_N_from_xraw(x_raw, K)
    n_unique = int(np.unique(N).size)
    n_phases = int(Xi.size)
    sd = float(np.std(Xi)) if n_phases else np.nan
    ok = (
        (n_unique >= config["Q_MIN_UNIQUE_N"])
        and (n_phases >= config["Q_MIN_PHASES"])
        and np.isfinite(sd)
        and (sd >= config["Q_MIN_STD"])
    )
    gate = dict(ok=bool(ok), n_unique_N=n_unique, n_phases=n_phases, std=sd)
    return Xi, gate


def gate_diagnostics(x_raw, K, config):
    """Check if K passes hard quality gates."""
    _, gate = phases_and_gate_from_xraw(x_raw, K, config)
    return gate


def _cache_put(config, key, value):
    _NULL_CACHE[key] = value
    _NULL_CACHE.move_to_end(key)
    max_keys = int(config.get("NULL_CACHE_MAX_KEYS", 6))
    while len(_NULL_CACHE) > max_keys:
        _NULL_CACHE.popitem(last=False)


def _null_cache_path(n, K, config):
    cache_dir = config.get("NULL_CACHE_DIR")
    if not cache_dir:
        return None
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fname = (
        f"null_concat_n{int(n)}_K{int(K)}_B{int(config['NEWNULL_N'])}_"
        f"{config['NEWNULL_GEN']}.npy"
    )
    return cache_path / fname


def _build_null_pool(n, K, config):
    rng = np.random.default_rng(config["RNG_SEED"])
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
        return np.concatenate(Xi_null_list)
    if config["NEWNULL_POOL"] == "match_n":
        return Xi_null_list
    raise ValueError("pool must be 'concat' or 'match_n'")


def get_null_pool(n, K, config):
    """Get null pool with in-memory LRU cache and optional disk cache."""
    key = (
        int(n),
        int(K),
        int(config["NEWNULL_N"]),
        str(config["NEWNULL_GEN"]),
        str(config["NEWNULL_POOL"]),
    )
    if key in _NULL_CACHE:
        _NULL_CACHE.move_to_end(key)
        return _NULL_CACHE[key]

    if config["NEWNULL_POOL"] == "concat":
        cache_file = _null_cache_path(n=n, K=K, config=config)
        if cache_file is not None and cache_file.exists():
            val = np.load(cache_file, allow_pickle=False)
            _cache_put(config, key, val)
            return val

    val = _build_null_pool(n=n, K=K, config=config)

    if config["NEWNULL_POOL"] == "concat":
        cache_file = _null_cache_path(n=n, K=K, config=config)
        if cache_file is not None:
            np.save(cache_file, val)

    _cache_put(config, key, val)
    return val


def ks2_newnull_from_phases(Xi_data, x_len, K, config):
    """Two-sample KS test from precomputed data phases."""
    Xi_data = np.asarray(Xi_data, dtype=float)
    n_data = int(Xi_data.size)
    if n_data == 0:
        raise ValueError("No phases in data")

    null_pool = get_null_pool(n=int(x_len), K=int(K), config=config)

    if config["NEWNULL_POOL"] == "concat":
        Xi_null = np.asarray(null_pool, dtype=float)
        res = stats.ks_2samp(
            Xi_data,
            Xi_null,
            alternative="two-sided",
            method=config["KS2_METHOD"],
        )
        return float(res.statistic), float(res.pvalue), n_data, int(Xi_null.size)

    Ds = []
    Xi_null_list = null_pool
    for Xi_b in Xi_null_list:
        res_b = stats.ks_2samp(
            Xi_data,
            Xi_b,
            alternative="two-sided",
            method=config["KS2_METHOD"],
        )
        Ds.append(float(res_b.statistic))
    D_med = float(np.median(Ds))
    p_rep = float(np.mean(np.asarray(Ds) >= D_med))
    return D_med, p_rep, n_data, None


def ks2_newnull(x_raw, K, config):
    """Two-sample KS test: data phases vs simulated null."""
    Xi_data, _ = phases_and_N_from_xraw(x_raw, K)
    return ks2_newnull_from_phases(Xi_data=Xi_data, x_len=len(x_raw), K=K, config=config)


def bonferroni(p, m):
    """Bonferroni correction for multiple testing."""
    return min(1.0, float(p) * max(int(m), 1))
