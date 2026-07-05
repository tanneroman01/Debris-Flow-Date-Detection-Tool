"""
Train the HGB date-detection model from a known-dates scene time-series CSV.

Input CSV (produced by Spectral_Separability_Analysis/pull_timeseries.py):
one row per polygon-scene with columns
    ID, date, Before_Scn, After_Scn, NDVI, NDRE, NBR, NDSI, RECI, B04, score
where [Before_Scn, After_Scn] is the known event window for that polygon.

Labels: a scene is positive if it falls within +/-LABEL_WINDOW_DAYS of the
known event window. Before fitting the final model, out-of-fold hit rates
are printed using GroupKFold grouped by event (fire), so no event leaks
between train and test folds. These should reproduce the numbers from
algo_experiments/evaluate_final.py.

The saved artifact is a joblib dict consumed by pipeline/time_detect.py:
    {"model", "feature_names", "trained", "training_csv",
     "label_window_days", "hyperparams", "oof_metrics", ...}

Usage:
    python tools/train_model.py --csv <timeseries_all_polygons.csv> [--out <path>]
"""

import argparse
import os
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.features import FEATURE_COLS, MIN_SCENES, build_features

LABEL_WINDOW_DAYS = 14
HYPERPARAMS = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.1, "random_state": 0}
EVAL_WINDOWS = (7, 14, 21, 30)

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "Spectral_Separability_Analysis", "timeseries_all_polygons.csv",
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "model", "hgb_date_model.joblib",
)


def build_training_frame(csv_path):
    """Per-scene features + labels + grouping keys for the whole training set."""
    df = pd.read_csv(csv_path, parse_dates=["date", "Before_Scn", "After_Scn"])
    parts = []
    for pid, g in df.groupby("ID"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < MIN_SCENES:
            print(f"  skipping polygon {pid}: only {len(g)} scenes")
            continue
        F = build_features(g)
        bf, af = g["Before_Scn"].iloc[0], g["After_Scn"].iloc[0]
        lo, hi = min(bf, af), max(bf, af)
        F.insert(0, "ID", pid)
        F.insert(1, "date", g["date"].values)
        F["event_key"] = f"{bf.date()}_{af.date()}"
        F["win_lo"], F["win_hi"] = lo, hi
        parts.append(F)
    return pd.concat(parts, ignore_index=True)


def in_window(dates, lo, hi, W):
    t = pd.Timedelta(days=W)
    return (dates >= lo - t) & (dates <= hi + t)


def oof_metrics(F, n_splits=5):
    """Out-of-fold poly/event hit rates for the recipe, GroupKFold by event."""
    y = in_window(F["date"].values, F["win_lo"].values, F["win_hi"].values,
                  LABEL_WINDOW_DAYS).astype(int)
    groups = F["event_key"].values
    prob = np.full(len(F), np.nan)
    for tr, te in GroupKFold(n_splits).split(F, y, groups):
        m = HistGradientBoostingClassifier(**HYPERPARAMS)
        m.fit(F.iloc[tr][FEATURE_COLS], y[tr])
        prob[te] = m.predict_proba(F.iloc[te][FEATURE_COLS])[:, 1]
    F = F.assign(prob=prob)
    pick = F.loc[F.groupby("ID")["prob"].idxmax()]
    metrics = {}
    for W in EVAL_WINDOWS:
        hit = in_window(pick["date"].values, pick["win_lo"].values,
                        pick["win_hi"].values, W)
        metrics[f"poly_W{W}"] = float(hit.mean())
        metrics[f"event_W{W}"] = float(
            pick.assign(h=hit).groupby("event_key")["h"].mean().mean()
        )
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", default=DEFAULT_CSV, help="training time-series CSV")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output model path")
    ap.add_argument("--skip-cv", action="store_true",
                    help="skip the out-of-fold evaluation, just fit and save")
    args = ap.parse_args()

    print(f"Building features from {args.csv}")
    F = build_training_frame(args.csv)
    y = in_window(F["date"].values, F["win_lo"].values, F["win_hi"].values,
                  LABEL_WINDOW_DAYS).astype(int)
    print(f"  {len(F)} scenes, {F['ID'].nunique()} polygons, "
          f"{F['event_key'].nunique()} events, {y.mean():.3f} positive rate")

    metrics = {}
    if not args.skip_cv:
        print("Out-of-fold evaluation (GroupKFold by event):")
        metrics = oof_metrics(F)
        print("  poly:  " + " ".join(f"W{W}={metrics[f'poly_W{W}']:.3f}" for W in EVAL_WINDOWS))
        print("  event: " + " ".join(f"W{W}={metrics[f'event_W{W}']:.3f}" for W in EVAL_WINDOWS))

    print("Fitting final model on all data...")
    model = HistGradientBoostingClassifier(**HYPERPARAMS)
    model.fit(F[FEATURE_COLS], y)

    artifact = {
        "model": model,
        "feature_names": FEATURE_COLS,
        "trained": date.today().isoformat(),
        "training_csv": os.path.abspath(args.csv),
        "label_window_days": LABEL_WINDOW_DAYS,
        "hyperparams": HYPERPARAMS,
        "n_scenes": len(F),
        "n_polygons": int(F["ID"].nunique()),
        "n_events": int(F["event_key"].nunique()),
        "oof_metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(artifact, args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
