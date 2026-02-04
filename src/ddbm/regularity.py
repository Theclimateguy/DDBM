"""
Regularity gate: separate periodic/limit-cycle from chaos
"""

import numpy as np


def fft_concentration(x, config):
    """Frequency concentration (narrowband test)"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 256:
        return np.nan
    x = x - np.mean(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    P = (X.real**2 + X.imag**2)
    if P.size <= 2:
        return np.nan
    P = P[1:]  # drop DC
    tot = float(P.sum()) + 1e-12
    k = int(min(int(config["REG_FFT_TOPK"]), P.size))
    idx = np.argpartition(P, -k)[-k:]
    return float(P[idx].sum() / tot)


def acf_max_abs(x, config):
    """Maximum absolute autocorrelation"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    max_lag = config["REG_ACF_MAX_LAG"]
    if n < max_lag + 10:
        return np.nan
    x = x - np.mean(x)
    var = float(np.dot(x, x) / n)
    if not np.isfinite(var) or var < 1e-12:
        return 1.0
    acfs = []
    for lag in range(1, int(max_lag) + 1):
        ac = float(np.dot(x[:-lag], x[lag:]) / (n - lag))
        acfs.append(ac / var)
    acfs = np.asarray(acfs, dtype=float)
    if acfs.size >= 2:
        return float(np.max(np.abs(acfs[1:])))  # ignore lag=1
    return float(np.max(np.abs(acfs)))


def permutation_entropy(x, config):
    """Normalized permutation entropy"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    m = config["REG_PE_M"]
    tau = 1
    L = n - (m - 1) * tau
    if L <= 300:
        return np.nan
    counts = {}
    for i in range(int(L)):
        v = x[i : i + m * tau : tau]
        pat = tuple(np.argsort(v, kind="mergesort"))
        counts[pat] = counts.get(pat, 0) + 1
    p = np.asarray(list(counts.values()), dtype=float)
    p /= float(p.sum())
    H = -float(np.sum(p * np.log(p + 1e-18)))
    Hmax = float(np.sum(np.log(np.arange(1, m + 1))))
    return float(H / (Hmax + 1e-18))


def regularity_gate(x_resid, config):
    """Apply regularity gate to distinguish chaos from periodic"""
    cfft = fft_concentration(x_resid, config)
    acfmx = acf_max_abs(x_resid, config)
    pe = permutation_entropy(x_resid, config)
    
    is_reg = False
    reasons = []
    
    if np.isfinite(cfft) and (cfft > config["REG_FFT_CONC_THR"]):
        is_reg = True
        reasons.append(f"fft_conc={cfft:.3f}>{config['REG_FFT_CONC_THR']}")
    
    if np.isfinite(acfmx) and (acfmx > config["REG_ACF_MAX_THR"]):
        if (np.isfinite(cfft) and (cfft > config["REG_FFT_CONC_THR"])):
            is_reg = True
            reasons.append(f"acf_max={acfmx:.3f}>{config['REG_ACF_MAX_THR']} (supported_by_fft)")
        else:
            reasons.append(f"acf_max_high_but_not_supported={acfmx:.3f}")
    
    if is_reg and np.isfinite(pe) and (pe > config["REG_PE_OVERRIDE_HI"]):
        is_reg = False
        reasons.append(f"override_high_PE={pe:.3f}>{config['REG_PE_OVERRIDE_HI']}")
    
    return {
        "fft_conc_topk": float(cfft) if np.isfinite(cfft) else np.nan,
        "acf_max_abs": float(acfmx) if np.isfinite(acfmx) else np.nan,
        "perm_entropy": float(pe) if np.isfinite(pe) else np.nan,
        "is_regular": bool(is_reg),
        "reason": ";".join(reasons) if reasons else None,
    }
