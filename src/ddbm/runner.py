"""
Main DDBM runner: two-level analysis (raw + residual)
"""

import numpy as np
from .ddbm_core import (
    remove_transients,
    phases_and_gate_from_xraw,
    ks2_newnull_from_phases,
    bonferroni
)
from .preprocessing import make_residual, ljung_box_pvalue
from .regularity import regularity_gate


def run_ddbm_on_array(x, tag, config):
    """Run DDBM on single array (used for raw and residual series)"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    min_points = int(config.get("MIN_POINTS", 1000))
    if len(x) < min_points:
        return dict(
            tag=tag,
            status="STRUCTURED_DEGENERATE",
            reason=f"Too few valid points for DDBM: {len(x)} (need >={min_points} recommended)",
            n_points_clean=int(len(x)),
            diagnostics=dict(
                coarse_total=0,
                coarse_passed=0,
                coarse_failed=0,
                gate_fail_counts={"unique": 0, "phases": 0, "std": 0, "other": 1},
            ),
            optimal=None,
        )
    
    x_clean, _ = remove_transients(x, method=config["TRANSIENT_METHOD"], n_transient=config["N_TRANSIENT"])
    if len(x_clean) < min_points:
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
    
    # ----- coarse scan with hard gates -----
    Ks_coarse = list(range(config["K_MIN"], config["K_MAX"] + 1, config["COARSE_STEP"]))
    res_coarse = {}
    fail_counts = {"unique": 0, "phases": 0, "std": 0, "other": 0}
    n_clean = int(len(x_clean))
    
    for K in Ks_coarse:
        Xi_data, gd = phases_and_gate_from_xraw(x_clean, K, config)
        if not gd["ok"]:
            if gd["n_unique_N"] < config["Q_MIN_UNIQUE_N"]:
                fail_counts["unique"] += 1
            elif gd["n_phases"] < config["Q_MIN_PHASES"]:
                fail_counts["phases"] += 1
            elif (not np.isfinite(gd["std"])) or (gd["std"] < config["Q_MIN_STD"]):
                fail_counts["std"] += 1
            else:
                fail_counts["other"] += 1
            continue
        
        D, p, n_data, n_null = ks2_newnull_from_phases(
            Xi_data=Xi_data,
            x_len=n_clean,
            K=K,
            config=config,
        )
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
    
    # ----- fine scan around anchor -----
    K_start = max(config["K_MIN"], anchor - config["COARSE_STEP"])
    K_end = min(config["K_MAX"], anchor + config["COARSE_STEP"])
    Ks_fine = list(range(K_start, K_end + 1, config["FINE_STEP"]))
    
    res_all = dict(res_coarse)
    fine_failed = 0
    
    for K in Ks_fine:
        Xi_data, gd = phases_and_gate_from_xraw(x_clean, K, config)
        if not gd["ok"]:
            fine_failed += 1
            continue
        
        D, p, n_data, n_null = ks2_newnull_from_phases(
            Xi_data=Xi_data,
            x_len=n_clean,
            K=K,
            config=config,
        )
        res_all[K] = dict(D=D, p=p, n_data=n_data, n_null=n_null, gate=gd)
    
    best_K = min(res_all, key=lambda K: res_all[K]["p"])
    best = res_all[best_K]
    m_tests = int(len(res_all))
    
    p_adj = bonferroni(best["p"], m_tests)
    status = "STRUCTURED" if (p_adj < config["ALPHA"]) else "NOISE"
    
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


def analyze_array(x, config):
    """Two-level analysis: raw + residual + regularity gate"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    
    # cheap diagnostics on raw
    lb_p_x = ljung_box_pvalue(x, lags=config["LB_LAGS"])
    lb_p_x2 = ljung_box_pvalue(x**2, lags=config["LB_LAGS"])
    
    # residualize
    x_resid, trend_info, ar1_info, roll_info = make_residual(x, config)
    
    # DDBM on raw and on residual
    ddbm_raw = run_ddbm_on_array(x, tag="raw", config=config)
    ddbm_resid = run_ddbm_on_array(x_resid, tag="resid", config=config)
    chaos_candidate_old = ddbm_resid["status"] == "STRUCTURED"
    
    # decision logic
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
        reg = regularity_gate(x_resid, config)
        
        if reg["is_regular"]:
            final_status = "REGULAR_NONCHAOTIC"
            chaos_candidate = False
        else:
            final_status = "CHAOS_CANDIDATE"
            chaos_candidate = True
    
    else:
        final_status = "UNKNOWN"
        chaos_candidate = bool(chaos_candidate_old)
    
    final_to_label = config.get("FINAL_TO_LABEL", {})
    return dict(
        final_status=final_status,
        final_label=final_to_label.get(final_status, "Unknown"),
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
            rollwin=int(roll_info.get("rollwin", config["ROLLVOL_WIN"])),
        ),
        regularity=reg,
        ddbm_raw=ddbm_raw,
        ddbm_resid=ddbm_resid,
    )
