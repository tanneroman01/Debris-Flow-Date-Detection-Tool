"""
Validate the SCAR v2 production date-detection path against known event dates.

The model's out-of-fold training metrics were computed on event-centered
extraction windows (+/-180 days around the known event). Production windows
are ignition-anchored: [ignition + post_fire_buffer_days, ignition + 5 years].
This script measures whether the hit rates survive that shift by running the
Sentinel-2-era polygons from known_dates_polygons.shp through the *production*
extraction (pipeline/time_detect._pull_scene_series) and scoring them.

Leakage control: some validation polygons are in the training set. For each
fire a held-out model is trained that excludes every training event within
EXCLUDE_RADIUS_M of that fire's polygons; results are reported for both the
held-out models and the shipped full model.

Scene pulls are cached under <out>/scene_cache/ so reruns skip GEE.

Usage:
    python tools/validate_production_path.py [--project dbflow-480621]
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import geopandas as gpd
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)
from pipeline import time_detect
from pipeline.features import FEATURE_COLS, INDEX_COLS
from tools.train_model import (
    DEFAULT_CSV, HYPERPARAMS, LABEL_WINDOW_DAYS, build_training_frame, in_window,
)

REPO_DIR = os.path.dirname(os.path.dirname(APP_DIR))
KNOWN_SHP = os.path.join(REPO_DIR, "date_train_set", "known_dates_polygons.shp")
TRAIN_SHP = os.path.join(REPO_DIR, "Spectral_Separability_Analysis", "polygons_with_scenes.shp")
FIRE_DEFAULTS = os.path.join(APP_DIR, "data", "fire_defaults.json")
DEFAULT_OUT = os.path.join(os.path.dirname(APP_DIR), "validation_production")

EXCLUDE_RADIUS_M = 20000
METRIC_CRS = "EPSG:26913"  # UTM 13N; all S2-era validation fires are Colorado
EVAL_WINDOWS = (7, 14, 21, 30)


def s2_validation_set():
    """S2-era polygons from the known-dates shapefile, with ignition dates."""
    kd = gpd.read_file(KNOWN_SHP).to_crs("EPSG:4326")
    kd["EVENT_DATE"] = pd.to_datetime(kd["EVENT_DATE"])
    with open(FIRE_DEFAULTS) as f:
        db = json.load(f)
    ign_by_fire = {}
    for k, v in db.items():
        if k.startswith("_"):
            continue
        ign_by_fire[(v.get("FIRENAME"), str(v.get("FIRE_YEAR")))] = v.get("IGN_DATE")
    rows = []
    for (fire, year), g in kd.groupby(["FIRE", "FIRE_YEAR"]):
        ign = ign_by_fire.get((fire, str(year)))
        if not ign or ign == "TODO":
            print(f"  skipping {fire} {year}: no ignition date in fire_defaults.json")
            continue
        ign_dt = datetime.strptime(ign, "%m/%d/%Y")
        if ign_dt < time_detect.S2_START:
            print(f"  skipping {fire} {year}: pre-Sentinel-2 ({len(g)} polygons)")
            continue
        g = g.copy()
        g["IGN_DATE"] = ign_dt
        rows.append(g)
    return pd.concat(rows).reset_index(drop=True)


def train_heldout_models(val):
    """One model per fire, trained without any event near that fire."""
    print(f"Building training features from {DEFAULT_CSV}")
    F = build_training_frame(DEFAULT_CSV)
    y_all = in_window(F["date"].values, F["win_lo"].values, F["win_hi"].values,
                      LABEL_WINDOW_DAYS).astype(int)

    ps = gpd.read_file(TRAIN_SHP).to_crs(METRIC_CRS)
    val_m = val.to_crs(METRIC_CRS)
    models = {}
    for fire, vg in val_m.groupby("FIRE"):
        fire_area = vg.geometry.union_all()
        near_ids = set(ps.loc[ps.distance(fire_area) < EXCLUDE_RADIUS_M, "ID"])
        excl_events = set(F.loc[F["ID"].isin(near_ids), "event_key"])
        keep = ~F["event_key"].isin(excl_events)
        print(f"  {fire}: excluding {len(near_ids)} training polygons / "
              f"{len(excl_events)} events ({(~keep).sum()} of {len(F)} scenes)")
        m = HistGradientBoostingClassifier(**HYPERPARAMS)
        m.fit(F.loc[keep, FEATURE_COLS], y_all[keep.values])
        models[fire] = {"model": m, "feature_names": FEATURE_COLS}
    return models


def pull_all(val, cfg, cache_dir):
    """Production-path scene pulls for every validation polygon, cached."""
    os.makedirs(cache_dir, exist_ok=True)

    def pull_one(i):
        row = val.iloc[i]
        cache = os.path.join(cache_dir, f"poly_{i}.csv")
        if os.path.exists(cache):
            return i, pd.read_csv(cache, parse_dates=["date"])
        ign = row["IGN_DATE"]
        start = (ign + timedelta(days=cfg["post_fire_buffer_days"])).strftime("%Y-%m-%d")
        end = min(ign + timedelta(days=cfg["search_years"] * 365),
                  datetime.now()).strftime("%Y-%m-%d")
        scenes = time_detect._pull_scene_series(row.geometry, start, end, cfg)
        scenes.to_csv(cache, index=False, date_format="%Y-%m-%d")
        return i, scenes

    series = {}
    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
        futures = {pool.submit(pull_one, i): i for i in range(len(val))}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                _, scenes = fut.result()
            except Exception as e:
                print(f"  polygon {i}: pull FAILED -- {e}")
                continue
            series[i] = scenes
            row = val.iloc[i]
            print(f"  [{len(series)}/{len(val)}] {row['FIRE']} poly {i}: "
                  f"{len(scenes)} usable scenes")
    return series


def summarize(res, col, label):
    d = res[col].abs()
    n_det = d.notna().sum()
    print(f"\n{label}: detected {n_det}/{len(res)} | "
          f"median |err| = {d.median():.0f}d")
    for W in EVAL_WINDOWS:
        hit = (d <= W)
        by_fire = res.assign(h=hit).groupby("FIRE")["h"].mean()
        print(f"  W{W:2d}: poly={hit.mean():.3f}  "
              + "  ".join(f"{f}={v:.2f}" for f, v in by_fire.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="dbflow-480621", help="GEE project ID")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print("Loading validation polygons...")
    val = s2_validation_set()
    print(f"{len(val)} S2-era polygons across "
          f"{val.groupby(['FIRE', 'FIRE_YEAR']).ngroups} fires")

    heldout = train_heldout_models(val)
    full = time_detect.load_model()

    import ee
    ee.Initialize(project=args.project)
    cfg = dict(time_detect.DEFAULTS)
    print(f"\nPulling production-path scene series "
          f"(buffer {cfg['post_fire_buffer_days']}d, {cfg['search_years']}y window)...")
    series = pull_all(val, cfg, os.path.join(args.out, "scene_cache"))

    rows = []
    for i in range(len(val)):
        row = val.iloc[i]
        out = {
            "poly": i, "FIRE": row["FIRE"], "FIRE_YEAR": row["FIRE_YEAR"],
            "NHID": row["NHID"], "known_date": row["EVENT_DATE"],
            "n_scenes": len(series.get(i, [])),
        }
        scenes = series.get(i)
        for tag, artifact in (("held", heldout[row["FIRE"]]), ("full", full)):
            ev = time_detect.predict_event(scenes, artifact) if scenes is not None else None
            if ev is None:
                out[f"{tag}_date"] = None
                out[f"{tag}_err_days"] = np.nan
                out[f"{tag}_prob"] = np.nan
            else:
                out[f"{tag}_date"] = ev["event_date"].date()
                out[f"{tag}_err_days"] = (ev["event_date"] - row["EVENT_DATE"]).days
                out[f"{tag}_prob"] = round(ev["probability"], 4)
        rows.append(out)
    res = pd.DataFrame(rows)

    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "results.csv")
    res.to_csv(res_path, index=False)
    print(f"\nSaved {res_path}")
    print(res[["poly", "FIRE", "known_date", "held_date", "held_err_days",
               "held_prob", "full_err_days", "n_scenes"]].to_string(index=False))

    summarize(res, "held_err_days", "HELD-OUT models (no event within 20 km in training)")
    summarize(res, "full_err_days", "FULL shipped model (validation fires seen in training)")
    print("\nNote: hit = |predicted - known event date| <= W days; this is stricter "
          "than the training metric, which measured hits against the known scene "
          "window +/- W.")


if __name__ == "__main__":
    main()
