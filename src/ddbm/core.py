"""
Public API for DDBM library
"""

import json
from pathlib import Path
import pandas as pd
from .config import DEFAULT_CONFIG
from .runner import analyze_array


def analyze_timeseries(x, alpha=0.05, config=None):
    """
    Main entry point for chaos detection
    
    Parameters
    ----------
    x : array-like
        Univariate time series
    alpha : float, default=0.05
        Significance level (after Bonferroni correction)
    config : dict, optional
        Override default configuration
    
    Returns
    -------
    result : dict
        Analysis results with keys:
        - final_status : str
        - chaos_candidate : bool
        - n_points_raw : int
        - n_points_resid : int
        - cheap : dict
        - regularity : dict
        - ddbm_raw : dict
        - ddbm_resid : dict
    """
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    cfg["ALPHA"] = alpha
    
    return analyze_array(x, cfg)


def batch_analyze(data_dir, output_dir, pattern="*.csv", value_col=-1, config=None):
    """
    Process multiple CSV files in batch mode
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing CSV files
    output_dir : str or Path
        Directory for JSON results and summary CSV
    pattern : str, default="*.csv"
        File pattern for glob matching
    value_col : int, default=-1
        Column index for time series data
    config : dict, optional
        Override default configuration
    
    Returns
    -------
    summary : pandas.DataFrame
        Aggregated results
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    
    csv_files = sorted(data_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}/")
    
    rows = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            x = pd.to_numeric(df.iloc[:, value_col], errors="coerce").to_numpy(dtype=float)
            
            result = analyze_array(x, cfg)
            
            # Save JSON
            out = {
                "metadata": {
                    "file": f.stem,
                    "path": str(f),
                    "timestamp": str(pd.Timestamp.now()),
                    "version": "ddbm_v7.1",
                },
                "config": cfg,
                "result": result,
            }
            
            json_file = output_dir / f"{f.stem}_analysis.json"
            with open(json_file, "w") as fp:
                json.dump(out, fp, indent=2)
            
            # Flatten to summary row
            raw = result["ddbm_raw"]
            resid = result["ddbm_resid"]
            cheap = result["cheap"]
            
            def opt_field(block, key, default=None):
                if (block is None) or (block.get("optimal") is None):
                    return default
                return block["optimal"].get(key, default)
            
            rows.append({
                "file": f.stem,
                "final_status": result["final_status"],
                "chaos_candidate": result["chaos_candidate"],
                "n_raw": result["n_points_raw"],
                "n_resid": result["n_points_resid"],
                "raw_status": raw.get("status"),
                "raw_K_opt": opt_field(raw, "K_opt"),
                "raw_D": opt_field(raw, "D"),
                "raw_p_adj": opt_field(raw, "p_adj_bonf"),
                "resid_status": resid.get("status"),
                "resid_K_opt": opt_field(resid, "K_opt"),
                "resid_D": opt_field(resid, "D"),
                "resid_p_adj": opt_field(resid, "p_adj_bonf"),
                "reg_is_regular": (result.get("regularity") or {}).get("is_regular"),
                "reg_fft_conc": (result.get("regularity") or {}).get("fft_conc_topk"),
                "reg_acf_max": (result.get("regularity") or {}).get("acf_max_abs"),
                "reg_pe": (result.get("regularity") or {}).get("perm_entropy"),
            })
            
            print(f"✓ {f.name}: {result['final_status']}")
        
        except Exception as e:
            rows.append({"file": f.stem, "final_status": "ERROR", "error": str(e)})
            print(f"✗ {f.name}: ERROR: {e}")
    
    summary = pd.DataFrame(rows)
    summary_file = output_dir / "batch_summary.csv"
    summary.to_csv(summary_file, index=False)
    
    print(f"Saved summary: {summary_file}")
    return summary
