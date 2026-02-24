import csv
import json
import math
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import chi2


warnings.filterwarnings("default", category=RuntimeWarning)


RNG_SEED = 42
np.random.seed(RNG_SEED)


def default_data_dir() -> Path:
    test_dir = Path("data/test")
    if test_dir.exists():
        return test_dir
    return Path("data")


DATA_DIR = default_data_dir()
RESULTS_DIR = Path("results_test")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE = Path("data/test/benchmark_manifest_v7_1.csv")
NULL_CACHE_DIR = RESULTS_DIR / "null_cache"
NULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# I/O
MIN_POINTS = 1000


# Transients
TRANSIENT_METHOD = "none"  # "none" / "fixed"
N_TRANSIENT = 0


# K scan
K_MIN, K_MAX = 10, 1000
COARSE_STEP, FINE_STEP = 50, 5


# Decision
ALPHA = 0.05


# New-null simulation (KS2 against simulated null)
NEWNULL_N = 200
NEWNULL_POOL = "concat"  # "concat" / "match_n"
NEWNULL_GEN = "u01"  # "u01" / "normal"
KS2_METHOD = "auto"
NULL_CACHE_MAX_KEYS = 6


# Hard gates
Q_MIN_UNIQUE_N = 20
Q_MIN_PHASES = 500
Q_MIN_STD = 1e-6


# Cheap screen and residualization
LB_LAGS = 20
AR1_CLIP = 0.99
TREND_MIN_N = 50


# Optional volatility standardization
ROLLVOL_WIN = 200
ROLLVOL_EPS = 1e-8
ROLLVOL_POLICY = "auto"  # "off" / "on" / "auto"
ARCH_ALPHA = 0.05


# Regularity gate (v7.1 behavior)
REG_FFT_TOPK = 3
REG_FFT_CONC_THR = 0.60
REG_ACF_MAX_LAG = 200
REG_ACF_MAX_THR = 0.85
REG_PE_M = 5
REG_PE_THR = 0.55
REG_PE_OVERRIDE_HI = 0.70


FINAL_TO_LABEL = {
    "CHAOS_CANDIDATE": "Chaos",
    "REGULAR_NONCHAOTIC": "Regular",
    "NOT_CHAOS_CANDIDATE": "Noise",
    "DEGENERATE": "Degenerate",
    "ERROR": "Error",
}


_NULL_CACHE = OrderedDict()


def _print_config() -> None:
    print("✓ Ready")
    print(f"  seed={RNG_SEED}")
    print(f"  data_dir={DATA_DIR.resolve()}")
    print(f"  results_dir={RESULTS_DIR.resolve()}")
    print(f"  manifest={MANIFEST_FILE.resolve()}")
    print(f"  null_cache={NULL_CACHE_DIR.resolve()}")
    print("✓ Config set")
    print(f"  K=[{K_MIN},{K_MAX}], coarse_step={COARSE_STEP}, fine_step={FINE_STEP}")
    print(f"  alpha={ALPHA}, newnull_n={NEWNULL_N}, pool={NEWNULL_POOL}, gen={NEWNULL_GEN}")
    print(f"  hard_gate: unique>={Q_MIN_UNIQUE_N}, phases>={Q_MIN_PHASES}, std>={Q_MIN_STD}")
    print(
        f"  regularity_gate: fft_topk={REG_FFT_TOPK}, fft_thr={REG_FFT_CONC_THR}, "
        f"acf_thr={REG_ACF_MAX_THR}, pe_thr={REG_PE_THR}"
    )


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        y = float(x)
        if not np.isfinite(y):
            return default
        return y
    except Exception:
        return default


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        print(f"! manifest missing: {path}")
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            file_stem = (row.get("file") or "").strip()
            if not file_stem:
                continue
            include_str = (row.get("include_in_paper") or "").strip().lower()
            include = include_str in {"1", "true", "yes", "y"}
            out[file_stem] = {
                "expected_label": (row.get("expected_label") or "").strip() or None,
                "group": (row.get("group") or "").strip() or None,
                "include_in_paper": include,
                "notes": (row.get("notes") or "").strip() or None,
            }
    return out


def read_series_from_csv(csv_file: Path) -> np.ndarray:
    with open(csv_file, newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration:
            return np.asarray([], dtype=float)

        headers = [str(h).strip().lower() for h in header]
        idx = -1
        for name in ("value", "x", "series", "y"):
            if name in headers:
                idx = headers.index(name)
                break
        if idx < 0:
            idx = max(0, len(header) - 1)

        vals = []
        for row in reader:
            if not row:
                continue
            if idx >= len(row):
                continue
            v = safe_float(row[idx], default=np.nan)
            if np.isfinite(v):
                vals.append(v)
    return np.asarray(vals, dtype=float)


def ljung_box_pvalue(x, lags=LB_LAGS):
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


def detrend_ols(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < TREND_MIN_N:
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


def prewhiten_ar1(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return x, dict(ar1_phi=np.nan)

    x0 = x[:-1]
    x1 = x[1:]
    denom = float(np.dot(x0, x0))
    phi = float(np.dot(x0, x1) / denom) if denom > 0 else 0.0
    phi = float(np.clip(phi, -AR1_CLIP, AR1_CLIP))
    resid = x1 - phi * x0
    return resid, dict(ar1_phi=phi)


def rolling_vol_standardize(x, win=ROLLVOL_WIN, eps=ROLLVOL_EPS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
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


def fft_concentration(x, topk=REG_FFT_TOPK):
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
    k = int(min(int(topk), P.size))
    idx = np.argpartition(P, -k)[-k:]
    return float(P[idx].sum() / tot)


def acf_max_abs(x, max_lag=REG_ACF_MAX_LAG):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
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


def permutation_entropy(x, m=REG_PE_M, tau=1):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
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
    Hmax = math.log(math.factorial(m))
    return float(H / (Hmax + 1e-18))


def regularity_gate(x_resid):
    cfft = fft_concentration(x_resid, topk=REG_FFT_TOPK)
    acfmx = acf_max_abs(x_resid, max_lag=REG_ACF_MAX_LAG)
    pe = permutation_entropy(x_resid, m=REG_PE_M)

    is_reg = False
    reasons = []

    if np.isfinite(cfft) and (cfft > REG_FFT_CONC_THR):
        is_reg = True
        reasons.append(f"fft_conc={cfft:.3f}>{REG_FFT_CONC_THR}")

    if np.isfinite(acfmx) and (acfmx > REG_ACF_MAX_THR):
        if np.isfinite(cfft) and (cfft > REG_FFT_CONC_THR):
            is_reg = True
            reasons.append(f"acf_max={acfmx:.3f}>{REG_ACF_MAX_THR} (supported_by_fft)")
        else:
            reasons.append(f"acf_max_high_but_not_supported={acfmx:.3f}")

    if is_reg and np.isfinite(pe) and (pe > REG_PE_OVERRIDE_HI):
        is_reg = False
        reasons.append(f"override_high_PE={pe:.3f}>{REG_PE_OVERRIDE_HI}")

    return {
        "fft_conc_topk": float(cfft) if np.isfinite(cfft) else np.nan,
        "acf_max_abs": float(acfmx) if np.isfinite(acfmx) else np.nan,
        "perm_entropy": float(pe) if np.isfinite(pe) else np.nan,
        "is_regular": bool(is_reg),
        "reason": ";".join(reasons) if reasons else None,
    }


def remove_transients(x, method="none", n_transient=None):
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
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if int(m.sum()) < 2:
        raise ValueError("Too few finite points for rank normalization")

    xr = x[m]
    order = np.argsort(xr, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(xr) + 1, dtype=float)

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
    x = np.asarray(values, dtype=np.float64)
    if x.min() < -0.01 or x.max() > 1.01:
        raise ValueError(f"x must be in [0,1] (tol). got [{x.min():.6f},{x.max():.6f}]")
    return np.floor(x * float(K) + 0.5).astype(dtype, copy=False)


def compute_diophantine_residuals(N):
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
    x01 = rank_normalize_01(x_raw)
    N = quantize_timeseries(x01, K)
    Xi = compute_diophantine_residuals(N)
    Xi = np.asarray(Xi, dtype=float)
    Xi = Xi[np.isfinite(Xi)]
    return Xi, N


def phases_and_gate_from_xraw(x_raw, K):
    Xi, N = phases_and_N_from_xraw(x_raw, K)
    n_unique = int(np.unique(N).size)
    n_phases = int(Xi.size)
    sd = float(np.std(Xi)) if n_phases else np.nan
    ok = (
        (n_unique >= Q_MIN_UNIQUE_N)
        and (n_phases >= Q_MIN_PHASES)
        and np.isfinite(sd)
        and (sd >= Q_MIN_STD)
    )
    gate = dict(ok=bool(ok), n_unique_N=n_unique, n_phases=n_phases, std=sd)
    return Xi, gate


def _cache_put(key, value):
    _NULL_CACHE[key] = value
    _NULL_CACHE.move_to_end(key)
    while len(_NULL_CACHE) > NULL_CACHE_MAX_KEYS:
        _NULL_CACHE.popitem(last=False)


def _null_cache_path(n, K, n_null=NEWNULL_N, generator=NEWNULL_GEN):
    fname = f"null_concat_n{int(n)}_K{int(K)}_B{int(n_null)}_{generator}.npy"
    return NULL_CACHE_DIR / fname


def _build_null_pool(n, K, n_null=NEWNULL_N, generator=NEWNULL_GEN):
    rng = np.random.default_rng(RNG_SEED)
    Xi_null_list = []
    for _ in range(int(n_null)):
        if generator == "u01":
            x_sim = rng.uniform(0.0, 1.0, n)
        elif generator == "normal":
            x_sim = rng.normal(0.0, 1.0, n)
        else:
            raise ValueError("generator must be 'u01' or 'normal'")
        Xi_b, _ = phases_and_N_from_xraw(x_sim, K)
        if Xi_b.size:
            Xi_null_list.append(Xi_b)
    if not Xi_null_list:
        raise ValueError("Null simulation produced no phases")

    if NEWNULL_POOL == "concat":
        return np.concatenate(Xi_null_list)

    if NEWNULL_POOL == "match_n":
        return Xi_null_list

    raise ValueError("pool must be 'concat' or 'match_n'")


def get_null_pool(n, K, n_null=NEWNULL_N, generator=NEWNULL_GEN):
    key = (int(n), int(K), int(n_null), str(generator), str(NEWNULL_POOL))
    if key in _NULL_CACHE:
        _NULL_CACHE.move_to_end(key)
        return _NULL_CACHE[key]
    if NEWNULL_POOL == "concat":
        cache_file = _null_cache_path(n=n, K=K, n_null=n_null, generator=generator)
        if cache_file.exists():
            val = np.load(cache_file)
            _cache_put(key, val)
            return val
    val = _build_null_pool(n=n, K=K, n_null=n_null, generator=generator)
    if NEWNULL_POOL == "concat":
        cache_file = _null_cache_path(n=n, K=K, n_null=n_null, generator=generator)
        np.save(cache_file, val)
    _cache_put(key, val)
    return val


def ks2_newnull_from_phases(Xi_data, x_len, K):
    Xi_data = np.asarray(Xi_data, dtype=float)
    n_data = int(Xi_data.size)
    if n_data == 0:
        raise ValueError("No phases in data")

    null_pool = get_null_pool(n=int(x_len), K=int(K), n_null=NEWNULL_N, generator=NEWNULL_GEN)
    if NEWNULL_POOL == "concat":
        Xi_null = null_pool
        res = stats.ks_2samp(Xi_data, Xi_null, alternative="two-sided", method=KS2_METHOD)
        return float(res.statistic), float(res.pvalue), n_data, int(Xi_null.size)

    Ds = []
    Xi_null_list = null_pool
    for Xi_b in Xi_null_list:
        res_b = stats.ks_2samp(Xi_data, Xi_b, alternative="two-sided", method=KS2_METHOD)
        Ds.append(float(res_b.statistic))
    D_med = float(np.median(Ds))
    p_rep = float(np.mean(np.asarray(Ds) >= D_med))
    return D_med, p_rep, n_data, None


def bonferroni(p, m):
    return min(1.0, float(p) * max(int(m), 1))


def run_ddbm_on_array(x, tag="raw"):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_POINTS:
        return dict(
            tag=tag,
            status="STRUCTURED_DEGENERATE",
            reason=f"Too few valid points for DDBM: {len(x)} (need >={MIN_POINTS} recommended)",
            n_points_clean=int(len(x)),
            diagnostics=dict(
                coarse_total=0,
                coarse_passed=0,
                coarse_failed=0,
                gate_fail_counts={"unique": 0, "phases": 0, "std": 0, "other": 1},
            ),
            optimal=None,
        )

    x_clean, _ = remove_transients(x, method=TRANSIENT_METHOD, n_transient=N_TRANSIENT)
    if len(x_clean) < MIN_POINTS:
        return dict(
            tag=tag,
            status="STRUCTURED_DEGENERATE",
            reason=f"Too few points after transient removal: {len(x_clean)}",
            n_points_clean=int(len(x_clean)),
            diagnostics=dict(
                coarse_total=0,
                coarse_passed=0,
                coarse_failed=0,
                gate_fail_counts={"unique": 0, "phases": 0, "std": 0, "other": 1},
            ),
            optimal=None,
        )

    Ks_coarse = list(range(K_MIN, K_MAX + 1, COARSE_STEP))
    res_coarse = {}
    fail_counts = {"unique": 0, "phases": 0, "std": 0, "other": 0}

    n_clean = int(len(x_clean))
    for K in Ks_coarse:
        Xi_data, gd = phases_and_gate_from_xraw(x_clean, K)
        if not gd["ok"]:
            if gd["n_unique_N"] < Q_MIN_UNIQUE_N:
                fail_counts["unique"] += 1
            elif gd["n_phases"] < Q_MIN_PHASES:
                fail_counts["phases"] += 1
            elif (not np.isfinite(gd["std"])) or (gd["std"] < Q_MIN_STD):
                fail_counts["std"] += 1
            else:
                fail_counts["other"] += 1
            continue
        D, p, n_data, n_null = ks2_newnull_from_phases(Xi_data=Xi_data, x_len=n_clean, K=K)
        res_coarse[K] = dict(D=D, p=p, n_data=n_data, n_null=n_null, gate=gd)

    if not res_coarse:
        return dict(
            tag=tag,
            status="STRUCTURED_DEGENERATE",
            reason=f"All coarse K failed hard gate; counts={fail_counts}",
            n_points_clean=int(len(x_clean)),
            diagnostics=dict(
                coarse_total=int(len(Ks_coarse)),
                coarse_passed=0,
                coarse_failed=int(len(Ks_coarse)),
                gate_fail_counts=fail_counts,
            ),
            optimal=None,
        )

    anchor = min(res_coarse, key=lambda K: res_coarse[K]["p"])
    K_start = max(K_MIN, anchor - COARSE_STEP)
    K_end = min(K_MAX, anchor + COARSE_STEP)
    Ks_fine = list(range(K_start, K_end + 1, FINE_STEP))

    res_all = dict(res_coarse)
    fine_failed = 0
    for K in Ks_fine:
        Xi_data, gd = phases_and_gate_from_xraw(x_clean, K)
        if not gd["ok"]:
            fine_failed += 1
            continue
        D, p, n_data, n_null = ks2_newnull_from_phases(Xi_data=Xi_data, x_len=n_clean, K=K)
        res_all[K] = dict(D=D, p=p, n_data=n_data, n_null=n_null, gate=gd)

    best_K = min(res_all, key=lambda K: res_all[K]["p"])
    best = res_all[best_K]
    m_tests = int(len(res_all))
    p_adj = bonferroni(best["p"], m_tests)
    status = "STRUCTURED" if (p_adj < ALPHA) else "NOISE"

    return dict(
        tag=tag,
        status=status,
        reason=None,
        n_points_clean=int(len(x_clean)),
        diagnostics=dict(
            n_K_tested=m_tests,
            coarse_total=int(len(Ks_coarse)),
            coarse_passed=int(len(res_coarse)),
            coarse_failed=int(len(Ks_coarse) - len(res_coarse)),
            fine_total=int(len(Ks_fine)),
            fine_failed=int(fine_failed),
            gate_fail_counts=fail_counts,
        ),
        optimal=dict(
            K_opt=int(best_K),
            D=float(best["D"]),
            p=float(best["p"]),
            p_adj_bonf=float(p_adj),
            m_tested=int(m_tests),
            gate=best["gate"],
        ),
    )


def make_residual(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    # 1) detrend
    x_dt, trend_info = detrend_ols(x)

    # 2) prewhiten AR(1)
    x_pw, ar1_info = prewhiten_ar1(x_dt)

    # 3) decide whether to apply rolling-vol standardization
    lb_p_x2 = ljung_box_pvalue(x**2, lags=LB_LAGS)
    roll_info = dict(roll_applied=False, roll_reason=None, rollwin=int(ROLLVOL_WIN))
    x_resid = x_pw

    if ROLLVOL_POLICY == "on":
        x_resid, rinfo = rolling_vol_standardize(x_resid, win=ROLLVOL_WIN, eps=ROLLVOL_EPS)
        roll_info.update(rinfo)
        roll_info["roll_reason"] = "policy_on"
    elif ROLLVOL_POLICY == "auto":
        if np.isfinite(lb_p_x2) and (lb_p_x2 < ARCH_ALPHA):
            x_resid, rinfo = rolling_vol_standardize(x_resid, win=ROLLVOL_WIN, eps=ROLLVOL_EPS)
            roll_info.update(rinfo)
            roll_info["roll_reason"] = f"auto_lb_x2_p<{ARCH_ALPHA}"
        else:
            roll_info["roll_reason"] = "auto_not_triggered"
    elif ROLLVOL_POLICY == "off":
        roll_info["roll_reason"] = "policy_off"
    else:
        raise ValueError("ROLLVOL_POLICY must be 'off', 'auto', or 'on'")

    return x_resid, trend_info, ar1_info, roll_info


def run_one_file(csv_file: Path):
    x = read_series_from_csv(csv_file)
    x = x[np.isfinite(x)]
    if len(x) < MIN_POINTS:
        raise ValueError(f"Too few valid points: {len(x)} (need >={MIN_POINTS} recommended)")

    lb_p_x = ljung_box_pvalue(x, lags=LB_LAGS)
    lb_p_x2 = ljung_box_pvalue(x**2, lags=LB_LAGS)
    x_resid, trend_info, ar1_info, roll_info = make_residual(x)

    ddbm_raw = run_ddbm_on_array(x, tag="raw")
    ddbm_resid = run_ddbm_on_array(x_resid, tag="resid")
    chaos_candidate_old = ddbm_resid["status"] == "STRUCTURED"

    reg = None
    final_status = "NOT_CHAOS_CANDIDATE"
    chaos_candidate = False

    if ddbm_resid["status"] == "NOISE":
        final_status = "NOT_CHAOS_CANDIDATE"
        chaos_candidate = False
    elif ddbm_resid["status"] == "STRUCTURED_DEGENERATE":
        final_status = "DEGENERATE"
        chaos_candidate = False
    elif ddbm_resid["status"] == "STRUCTURED":
        reg = regularity_gate(x_resid)
        if reg["is_regular"]:
            final_status = "REGULAR_NONCHAOTIC"
            chaos_candidate = False
        else:
            final_status = "CHAOS_CANDIDATE"
            chaos_candidate = True
    else:
        final_status = "UNKNOWN"
        chaos_candidate = bool(chaos_candidate_old)

    return dict(
        final_status=final_status,
        final_label=FINAL_TO_LABEL.get(final_status, "Unknown"),
        chaos_candidate=bool(chaos_candidate),
        chaos_candidate_old=bool(chaos_candidate_old),
        n_points_raw=int(len(x)),
        n_points_resid=int(len(x_resid)),
        cheap=dict(
            lb_p_x=float(lb_p_x) if np.isfinite(lb_p_x) else np.nan,
            lb_p_x2=float(lb_p_x2) if np.isfinite(lb_p_x2) else np.nan,
            trend_beta=trend_info.get("trend_beta", np.nan),
            trend_p=trend_info.get("trend_p", np.nan),
            ar1_phi=ar1_info.get("ar1_phi", np.nan),
            roll_applied=bool(roll_info.get("roll_applied", False)),
            roll_reason=roll_info.get("roll_reason", None),
            rollwin=int(roll_info.get("rollwin", ROLLVOL_WIN)),
        ),
        regularity=reg,
        ddbm_raw=ddbm_raw,
        ddbm_resid=ddbm_resid,
    )


def score_prediction(expected, predicted):
    if (expected is None) or (predicted is None):
        return None
    if expected == "Mixed":
        return predicted in {"Chaos", "Noise"}
    return expected == predicted


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fp:
            fp.write("")
        return

    preferred = [
        "file",
        "final_status",
        "final_label",
        "chaos_candidate",
        "benchmark_expected",
        "benchmark_pred_label",
        "benchmark_group",
        "include_in_paper",
        "benchmark_correct",
    ]
    keys = []
    seen = set()
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def benchmark_report(rows: list[dict]) -> dict:
    bench = [r for r in rows if bool(r.get("include_in_paper", False))]
    if not bench:
        return {
            "n_manifest": 0,
            "n_scored": 0,
            "n_correct": 0,
            "accuracy": np.nan,
            "category_summary": [],
            "misclassified": [],
        }

    n_scored = len(bench)
    n_correct = sum(1 for r in bench if r.get("benchmark_correct") is True)
    acc = float(n_correct / n_scored) if n_scored else np.nan

    by_group = {}
    for r in bench:
        g = r.get("benchmark_group") or "NA"
        if g not in by_group:
            by_group[g] = {"group": g, "n": 0, "correct": 0}
        by_group[g]["n"] += 1
        if r.get("benchmark_correct") is True:
            by_group[g]["correct"] += 1

    cat_rows = []
    for g in sorted(by_group):
        n = by_group[g]["n"]
        c = by_group[g]["correct"]
        cat_rows.append({"group": g, "n": n, "correct": c, "accuracy": float(c / n) if n else np.nan})

    mis = [
        {
            "file": r.get("file"),
            "benchmark_expected": r.get("benchmark_expected"),
            "benchmark_pred_label": r.get("benchmark_pred_label"),
            "final_status": r.get("final_status"),
        }
        for r in bench
        if r.get("benchmark_correct") is False
    ]

    return {
        "n_manifest": n_scored,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": acc,
        "category_summary": cat_rows,
        "misclassified": mis,
    }


def main():
    _print_config()
    manifest = load_manifest(MANIFEST_FILE)

    csv_files = sorted(p for p in DATA_DIR.glob("*.csv") if p.name != MANIFEST_FILE.name)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {DATA_DIR}/")

    print(f"Batch: {len(csv_files)} files in {DATA_DIR}/")
    rows = []
    for f in csv_files:
        try:
            json_file = RESULTS_DIR / f"{f.stem}_twolevel_analysis.json"
            res = None
            reused = False
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as fp:
                        prev = json.load(fp)
                    maybe_res = prev.get("result")
                    if isinstance(maybe_res, dict) and ("final_status" in maybe_res):
                        res = maybe_res
                        reused = True
                except Exception:
                    res = None
                    reused = False
            if res is None:
                res = run_one_file(f)
            out = {
                "metadata": {
                    "file": f.stem,
                    "path": str(f),
                    "timestamp": datetime.now().isoformat(),
                    "version": "ddbm_v7_2_manifest_reproducible",
                },
                "config": {
                    "seed": int(RNG_SEED),
                    "transient_method": TRANSIENT_METHOD,
                    "n_transient": int(N_TRANSIENT),
                    "K_min": int(K_MIN),
                    "K_max": int(K_MAX),
                    "coarse_step": int(COARSE_STEP),
                    "fine_step": int(FINE_STEP),
                    "alpha": float(ALPHA),
                    "newnull_n": int(NEWNULL_N),
                    "newnull_pool": NEWNULL_POOL,
                    "newnull_gen": NEWNULL_GEN,
                    "ks2_method": KS2_METHOD,
                    "hard_gate": {
                        "min_unique_N": int(Q_MIN_UNIQUE_N),
                        "min_phases": int(Q_MIN_PHASES),
                        "min_std": float(Q_MIN_STD),
                    },
                    "cheap_screen": {
                        "lb_lags": int(LB_LAGS),
                        "trend_min_n": int(TREND_MIN_N),
                        "ar1_clip": float(AR1_CLIP),
                        "rollvol_policy": ROLLVOL_POLICY,
                        "arch_alpha": float(ARCH_ALPHA),
                        "rollvol_win": int(ROLLVOL_WIN),
                        "rollvol_eps": float(ROLLVOL_EPS),
                    },
                },
                "result": res,
            }

            with open(json_file, "w", encoding="utf-8") as fp:
                json.dump(out, fp, indent=2)

            raw = res["ddbm_raw"]
            resid = res["ddbm_resid"]
            cheap = res["cheap"]

            def opt_field(block, key, default=np.nan):
                if (block is None) or (block.get("optimal") is None):
                    return default
                return block["optimal"].get(key, default)

            m = manifest.get(f.stem, {})
            pred_label = res.get("final_label")
            expected = m.get("expected_label")
            correct = score_prediction(expected=expected, predicted=pred_label)

            rows.append(
                {
                    "file": f.stem,
                    "final_status": res["final_status"],
                    "final_label": pred_label,
                    "chaos_candidate": res["chaos_candidate"],
                    "n_raw": res["n_points_raw"],
                    "n_resid": res["n_points_resid"],
                    "benchmark_expected": expected,
                    "benchmark_pred_label": pred_label,
                    "benchmark_group": m.get("group"),
                    "include_in_paper": bool(m.get("include_in_paper", False)),
                    "benchmark_correct": correct,
                    "benchmark_notes": m.get("notes"),
                    "lb_p_x": cheap.get("lb_p_x", np.nan),
                    "lb_p_x2": cheap.get("lb_p_x2", np.nan),
                    "trend_p": cheap.get("trend_p", np.nan),
                    "trend_beta": cheap.get("trend_beta", np.nan),
                    "ar1_phi": cheap.get("ar1_phi", np.nan),
                    "roll_applied": cheap.get("roll_applied", False),
                    "roll_reason": cheap.get("roll_reason", None),
                    "raw_status": raw.get("status", None),
                    "raw_K_opt": opt_field(raw, "K_opt"),
                    "raw_D": opt_field(raw, "D"),
                    "raw_p_adj": opt_field(raw, "p_adj_bonf"),
                    "raw_m_tested": opt_field(raw, "m_tested"),
                    "resid_status": resid.get("status", None),
                    "resid_K_opt": opt_field(resid, "K_opt"),
                    "resid_D": opt_field(resid, "D"),
                    "resid_p_adj": opt_field(resid, "p_adj_bonf"),
                    "resid_m_tested": opt_field(resid, "m_tested"),
                    "raw_note": raw.get("reason", None),
                    "resid_note": resid.get("reason", None),
                    "chaos_candidate_old": res.get("chaos_candidate_old", np.nan),
                    "reg_is_regular": (res.get("regularity", {}) or {}).get("is_regular", np.nan),
                    "reg_fft_conc_topk": (res.get("regularity", {}) or {}).get("fft_conc_topk", np.nan),
                    "reg_acf_max_abs": (res.get("regularity", {}) or {}).get("acf_max_abs", np.nan),
                    "reg_perm_entropy": (res.get("regularity", {}) or {}).get("perm_entropy", np.nan),
                    "reg_reason": (res.get("regularity", {}) or {}).get("reason", None),
                }
            )
            mark = "↺" if reused else "✓"
            print(
                f"{mark} {f.name}: final={res['final_status']}, raw={raw['status']}, "
                f"resid={resid['status']}, reg={(res.get('regularity') or {}).get('is_regular', None)}"
            )

        except Exception as e:
            m = manifest.get(f.stem, {})
            rows.append(
                {
                    "file": f.stem,
                    "final_status": "ERROR",
                    "final_label": "Error",
                    "benchmark_expected": m.get("expected_label"),
                    "benchmark_pred_label": "Error",
                    "benchmark_group": m.get("group"),
                    "include_in_paper": bool(m.get("include_in_paper", False)),
                    "benchmark_correct": False,
                    "note": str(e),
                }
            )
            print(f"✗ {f.name}: ERROR: {e}")

    summary_file = RESULTS_DIR / "batch_summary_twolevel.csv"
    write_rows_csv(summary_file, rows)

    report = benchmark_report(rows)
    report_file = RESULTS_DIR / "benchmark_report_v7_1.json"
    with open(report_file, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    cat_file = RESULTS_DIR / "benchmark_category_summary.csv"
    write_rows_csv(cat_file, report.get("category_summary", []))

    print("\nSaved:")
    print(f"  {summary_file}")
    print(f"  {report_file}")
    print(f"  {cat_file}")
    if np.isfinite(report.get("accuracy", np.nan)):
        print(
            f"Benchmark score: {report['n_correct']}/{report['n_scored']} "
            f"({100.0 * report['accuracy']:.1f}%)"
        )
    else:
        print("Benchmark score: n/a (manifest missing or unmatched)")


if __name__ == "__main__":
    main()
