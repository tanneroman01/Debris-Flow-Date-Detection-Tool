"""
Feature engineering for the HGB date-detection model.

Shared by training (tools/train_model.py) and inference (pipeline/time_detect.py)
so that the model always sees features built exactly the way it was trained.
Ported from algo_experiments/features_v2.py.

For one polygon's per-scene index time series this produces, per index:
  {c}_z          robust z-score of residuals from a 2-harmonic seasonal fit
                 (deseasonalized anomaly)
  {c}_step{k}    robust-z of the local before/after mean shift, k in (3, 5, 8)
  {c}_t6         t-like sustained level-shift statistic (k=6)
plus doy_sin / doy_cos day-of-year season encoding.
"""

import numpy as np
import pandas as pd

# Spectral index columns expected in the per-scene series, in training order.
INDEX_COLS = ["NDVI", "NDRE", "NBR", "NDSI", "RECI", "B04", "score"]

N_HARMONICS = 2
STEP_KS = (3, 5, 8)
TSTAT_K = 6

# 2-harmonic design has 2*N_HARMONICS + 2 parameters; below ~10 scenes the
# fit (and the step windows) are meaningless.
MIN_SCENES = 10


def harmonic_design(x, nh=N_HARMONICS, period=365.25):
    cols = [np.ones_like(x), x]
    for k in range(1, nh + 1):
        cols.append(np.sin(2 * np.pi * k * x / period))
        cols.append(np.cos(2 * np.pi * k * x / period))
    return np.column_stack(cols)


def harmonic_residuals(x, y, nh=N_HARMONICS):
    X = harmonic_design(x, nh)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def robust_z(r):
    med = np.median(r)
    mad = np.median(np.abs(r - med)) or 1e-9
    return (r - med) / (1.4826 * mad)


def step_stat(y, k):
    """Local level shift: mean(next k) - mean(previous k) at each position."""
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        a = y[max(0, i - k):i]
        b = y[i:i + k]
        if len(a) >= 2 and len(b) >= 2:
            out[i] = b.mean() - a.mean()
    return out


def tstat(y, k=TSTAT_K):
    """t-like sustained level-shift statistic."""
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        a = y[max(0, i - k):i]
        b = y[i:i + k]
        if len(a) >= 3 and len(b) >= 3:
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) or 1e-9
            out[i] = (b.mean() - a.mean()) / se
    return out


def build_features(g):
    """
    Build the model feature frame for one polygon's scene series.

    Args:
        g: DataFrame with a datetime 'date' column and all INDEX_COLS,
           one row per surviving scene. Must be sorted by date.

    Returns:
        DataFrame of features, row-aligned with g.
    """
    x = (g["date"] - g["date"].min()).dt.days.to_numpy(float)
    F = {}
    for c in INDEX_COLS:
        y = g[c].to_numpy(float)
        F[f"{c}_z"] = robust_z(harmonic_residuals(x, y))
        for k in STEP_KS:
            F[f"{c}_step{k}"] = robust_z(step_stat(y, k))
        F[f"{c}_t{TSTAT_K}"] = tstat(y)
    F = pd.DataFrame(F, index=g.index)
    doy = g["date"].dt.dayofyear
    F["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    F["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return F


FEATURE_COLS = (
    [f"{c}_{s}" for c in INDEX_COLS
     for s in ["z"] + [f"step{k}" for k in STEP_KS] + [f"t{TSTAT_K}"]]
    + ["doy_sin", "doy_cos"]
)
