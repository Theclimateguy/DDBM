"""
Preprocessing utilities: detrending, AR(1), volatility standardization
"""

import numpy as np
from scipy import stats
from scipy.stats import chi2


def ljung_box_pvalue(x, lags=20):
    """Ljung-Box test for autocorrelation"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= lags + 2:
        return np.nan
    
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    
    acf = np.empty(lags, dtype=float)
    for k in range(1, lags + 1):
        acf[k - 1] = float(np.dot(x[:-k], x[k:]) / denom)
    
    Q = n * (n + 2) * np.sum((acf**2) / (n - np.arange(1, lags + 1)))
    p = 1.0 - chi2.cdf(Q, df=lags)
    return float(p)


def detrend_ols(x, config):
    """OLS detrending"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < config["TREND_MIN_N"]:
        return x, dict(trend_beta=np.nan, trend_p=np.nan)
    
    t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    resid = x - X @ beta
    
    yhat = X @ beta
    s2 = float(np.sum((x - yhat) ** 2) / max(n - 2, 1))
    XtX_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(s2 * XtX_inv[1, 1])) if np.isfinite(s2) else np.nan
    z = float(beta[1] / se_slope) if (np.isfinite(se_slope) and se_slope > 0) else np.nan
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    
    return resid, dict(trend_beta=float(beta[1]), trend_p=p)


def prewhiten_ar1(x, config):
    """AR(1) prewhitening"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return x, dict(ar1_phi=np.nan)
    
    x0 = x[:-1]
    x1 = x[1:]
    denom = float(np.dot(x0, x0))
    phi = float(np.dot(x0, x1) / denom) if denom > 0 else 0.0
    phi = float(np.clip(phi, -config["AR1_CLIP"], config["AR1_CLIP"]))
    
    resid = x1 - phi * x0
    return resid, dict(ar1_phi=phi)


def rolling_vol_standardize(x, config):
    """Rolling window volatility standardization"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    win = config["ROLLVOL_WIN"]
    eps = config["ROLLVOL_EPS"]
    
    if n < win + 5:
        return x, dict(rollwin=int(win), roll_applied=False)

    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    sum1 = c1[win:] - c1[:-win]
    sum2 = c2[win:] - c2[:-win]
    mean = sum1 / win
    var = np.maximum(sum2 / win - mean * mean, 0.0)
    vol = np.full(n, np.nan, dtype=float)
    vol[win - 1 :] = np.sqrt(var)
    ok = np.isfinite(vol) & (vol > eps)
    y = x[ok] / vol[ok]
    
    return y, dict(rollwin=int(win), roll_applied=True)


def make_residual(x, config):
    """Full residualization pipeline"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    
    # 1) detrend
    x_dt, trend_info = detrend_ols(x, config)
    
    # 2) prewhiten AR(1)
    x_pw, ar1_info = prewhiten_ar1(x_dt, config)
    
    # 3) decide whether to apply rolling-vol standardization
    lb_p_x2 = ljung_box_pvalue(x**2, lags=config["LB_LAGS"])
    
    roll_info = dict(roll_applied=False, roll_reason=None, rollwin=int(config["ROLLVOL_WIN"]))
    x_resid = x_pw
    
    if config["ROLLVOL_POLICY"] == "on":
        x_resid, rinfo = rolling_vol_standardize(x_resid, config)
        roll_info.update(rinfo)
        roll_info["roll_reason"] = "policy_on"
    
    elif config["ROLLVOL_POLICY"] == "auto":
        if np.isfinite(lb_p_x2) and (lb_p_x2 < config["ARCH_ALPHA"]):
            x_resid, rinfo = rolling_vol_standardize(x_resid, config)
            roll_info.update(rinfo)
            roll_info["roll_reason"] = f"auto_lb_x2_p<{config['ARCH_ALPHA']}"
        else:
            roll_info["roll_reason"] = "auto_not_triggered"
    
    elif config["ROLLVOL_POLICY"] == "off":
        roll_info["roll_reason"] = "policy_off"
    
    else:
        raise ValueError("ROLLVOL_POLICY must be 'off', 'auto', or 'on'")
    
    return x_resid, trend_info, ar1_info, roll_info
