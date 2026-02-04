"""
Default configuration parameters for DDBM
"""

DEFAULT_CONFIG = {
    # Random seed
    "RNG_SEED": 42,
    
    # K scan
    "K_MIN": 10,
    "K_MAX": 1000,
    "COARSE_STEP": 50,
    "FINE_STEP": 5,
    
    # Decision thresholds
    "ALPHA": 0.05,
    "NEWNULL_N": 200,
    "NEWNULL_POOL": "concat",  # "concat" / "match_n"
    "NEWNULL_GEN": "u01",      # "u01" / "normal"
    "KS2_METHOD": "auto",
    
    # Hard gates (quality control)
    "Q_MIN_UNIQUE_N": 20,
    "Q_MIN_PHASES": 500,
    "Q_MIN_STD": 1e-6,
    
    # Transients
    "TRANSIENT_METHOD": "none",  # "none" / "fixed"
    "N_TRANSIENT": 0,
    
    # Residualization
    "LB_LAGS": 20,
    "TREND_MIN_N": 50,
    "AR1_CLIP": 0.99,
    "ROLLVOL_WIN": 200,
    "ROLLVOL_EPS": 1e-8,
    "ROLLVOL_POLICY": "auto",  # "off" / "auto" / "on"
    "ARCH_ALPHA": 0.05,
    
    # Regularity gate
    "REG_FFT_TOPK": 3,
    "REG_FFT_CONC_THR": 0.60,
    "REG_ACF_MAX_LAG": 200,
    "REG_ACF_MAX_THR": 0.85,
    "REG_PE_M": 5,
    "REG_PE_THR": 0.55,
    "REG_PE_OVERRIDE_HI": 0.70,
}
