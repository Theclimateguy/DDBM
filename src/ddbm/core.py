"""
Public API for DDBM library.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG
from .runner import analyze_array

try:
    import pandas as pd  # Optional dependency for DataFrame return type
except Exception:  # pragma: no cover - optional import fallback
    pd = None


def analyze_timeseries(x, alpha=0.05, config=None):
    """
    Main entry point for chaos detection.

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
        Analysis result dictionary.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    cfg["ALPHA"] = alpha

    return analyze_array(x, cfg)


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        y = float(x)
        if not np.isfinite(y):
            return default
        return y
    except Exception:
        return default


def _read_series_from_csv(csv_file, value_col=-1):
    with open(csv_file, newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        try:
            header = next(reader)
        except StopIteration:
            return np.asarray([], dtype=float)

        headers = [str(h).strip().lower() for h in header]
        idx = None
        for name in ("value", "x", "series", "y"):
            if name in headers:
                idx = headers.index(name)
                break

        if idx is None:
            idx = int(value_col)
            if idx < 0:
                idx = max(0, len(header) + idx)
            idx = min(idx, max(0, len(header) - 1))

        vals = []
        for row in reader:
            if not row:
                continue
            if idx >= len(row):
                continue
            v = _safe_float(row[idx], default=np.nan)
            if np.isfinite(v):
                vals.append(v)
    return np.asarray(vals, dtype=float)


def _write_rows_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fp:
            fp.write("")
        return

    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def batch_analyze(data_dir, output_dir, pattern="*.csv", value_col=-1, config=None):
    """
    Process multiple CSV files in batch mode.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing CSV files
    output_dir : str or Path
        Directory for JSON results and summary CSV
    pattern : str, default="*.csv"
        File pattern for glob matching
    value_col : int, default=-1
        Column index fallback if named columns are not found
    config : dict, optional
        Override default configuration

    Returns
    -------
    summary : pandas.DataFrame or list[dict]
        Aggregated results. If pandas is available, returns DataFrame.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    if not cfg.get("NULL_CACHE_DIR"):
        cfg["NULL_CACHE_DIR"] = str(output_dir / "null_cache")

    csv_files = sorted(data_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}/")

    rows = []

    for f in csv_files:
        try:
            x = _read_series_from_csv(f, value_col=value_col)
            result = analyze_array(x, cfg)

            out = {
                "metadata": {
                    "file": f.stem,
                    "path": str(f),
                    "timestamp": datetime.now().isoformat(),
                    "version": "ddbm_v7_2_library",
                },
                "config": cfg,
                "result": result,
            }

            json_file = output_dir / f"{f.stem}_analysis.json"
            with open(json_file, "w", encoding="utf-8") as fp:
                json.dump(out, fp, indent=2)

            raw = result["ddbm_raw"]
            resid = result["ddbm_resid"]

            def opt_field(block, key, default=None):
                if (block is None) or (block.get("optimal") is None):
                    return default
                return block["optimal"].get(key, default)

            rows.append(
                {
                    "file": f.stem,
                    "final_status": result["final_status"],
                    "final_label": result.get("final_label"),
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
                }
            )

            print(f"✓ {f.name}: {result['final_status']}")

        except Exception as e:
            rows.append({"file": f.stem, "final_status": "ERROR", "error": str(e)})
            print(f"✗ {f.name}: ERROR: {e}")

    summary_file = output_dir / "batch_summary.csv"
    _write_rows_csv(summary_file, rows)
    print(f"Saved summary: {summary_file}")

    if pd is not None:
        return pd.DataFrame(rows)
    return rows
