"""
Step 3: Debris flow date detection using a gradient-boosted classifier
on per-scene Sentinel-2 index time series (SCAR v2).

Replaces the v1 rule-based composite-change detector. For each polygon this
pulls every individual Sentinel-2 scene in the search window (3-5 day revisit
instead of 30-day median composites), computes spectral indices, builds
deseasonalized anomaly features, and scores each scene with a pre-trained
HistGradientBoostingClassifier. The scene with the highest event probability
is the detected date.

Scene quality gate (mirrors the training extraction in
Spectral_Separability_Analysis/pull_timeseries.py): a scene is dropped
outright if more than ``cloud_max_frac`` of the polygon is flagged
cloud/shadow by the SCL band, or if the polygon-mean NDSI exceeds
``ndsi_snow_thresh`` (snow). Surviving scenes contribute a plain polygon
mean per index, with no per-pixel masking.

Sentinel-2 only: fires ignited before mid-2015 are rejected -- use the v1
app with the Landsat backend for those.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import ee

from pipeline.features import INDEX_COLS, MIN_SCENES, build_features

# ---------- Default detection parameters ----------
DEFAULTS = {
    # Search window: [ignition + buffer, min(ignition + search_years, today)].
    # The buffer keeps the burn itself (a large spectral anomaly) out of the
    # candidate scenes; reduce it for fires with first-monsoon debris flows.
    "post_fire_buffer_days": 270,
    "search_years": 5,
    # Per-scene extraction (must mirror the training pull)
    "scale": 10,
    "cloud_max_frac": 0.2,
    "ndsi_snow_thresh": 0.6,
    "scl_cloud_classes": [3, 8, 9, 10],  # shadow, cloud med/high, cirrus
    "max_workers": 10,
    # Composite-score weights: the v1 change score is kept as a model input
    # feature, so these must match the weights used for the training data.
    "weight_nbr": 0.35,
    "weight_ndvi": 0.45,
    "weight_b04": 0.20,
}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "model", "hgb_date_model.joblib",
)

S2_START = datetime(2015, 6, 23)  # Sentinel-2A operational start

# Output field names (same contract as v1; merger.py picks these up)
FIELD_EVENT_DATE = "EVENT_DATE"
FIELD_START = "DATE_START"
FIELD_END = "DATE_END"
FIELD_CONFIDENCE = "CONFIDENCE"
FIELD_CHG_SCORE = "CHG_SCORE"


def _shapely_to_ee(geom):
    if geom.geom_type == "Polygon":
        return ee.Geometry.Polygon(
            [[list(c) for c in geom.exterior.coords]]
            + [[list(c) for c in r.coords] for r in geom.interiors]
        )
    if geom.geom_type == "MultiPolygon":
        parts = [
            [[list(c) for c in p.exterior.coords]]
            + [[list(c) for c in r.coords] for r in p.interiors]
            for p in geom.geoms
        ]
        return ee.Geometry.MultiPolygon(parts)
    raise TypeError(f"Unsupported geometry type: {geom.geom_type}")


def _pull_scene_series(geom, start, end, cfg):
    """One polygon's per-scene S2 index series; cloudy/snowy scenes dropped."""
    ee_poly = _shapely_to_ee(geom)
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(ee_poly)
    )

    def per_image(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndre = img.normalizedDifference(["B8A", "B5"]).rename("NDRE")
        nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
        ndsi = img.normalizedDifference(["B3", "B11"]).rename("NDSI")
        reci = img.expression(
            "((R + RE3) - (RE1 + N8A)) / ((R + RE3) + (RE1 + N8A))",
            {"R": img.select("B4"), "RE1": img.select("B5"),
             "RE3": img.select("B7"), "N8A": img.select("B8A")},
        ).rename("RECI")
        b04 = img.select("B4").rename("B04")
        # mean of this 0/1 band is the polygon's cloud-cover fraction
        cloud = img.select("SCL").remap(
            cfg["scl_cloud_classes"], [1] * len(cfg["scl_cloud_classes"]), 0
        ).rename("cloud_frac")
        stats = (
            ndvi.addBands([ndre, nbr, ndsi, reci, b04, cloud])
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_poly,
                          scale=cfg["scale"], maxPixels=1e7)
        )
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "NDVI": stats.get("NDVI"), "NDRE": stats.get("NDRE"),
            "NBR": stats.get("NBR"), "NDSI": stats.get("NDSI"),
            "RECI": stats.get("RECI"), "B04": stats.get("B04"),
            "cloud_frac": stats.get("cloud_frac"),
        })

    feats = coll.map(per_image).getInfo()["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        cloud_frac = p.get("cloud_frac")
        ndsi = p.get("NDSI")
        # cloud_frac is None only when the polygon fell in an image nodata gap
        if cloud_frac is None:
            continue
        if cloud_frac > cfg["cloud_max_frac"]:
            continue
        if ndsi is not None and ndsi > cfg["ndsi_snow_thresh"]:
            continue
        ndvi, nbr, b04 = p.get("NDVI"), p.get("NBR"), p.get("B04")
        if any(v is None for v in (ndvi, nbr, b04, ndsi, p.get("NDRE"), p.get("RECI"))):
            continue
        nbr_n = (nbr + 1) / 2                          # map [-1,1] -> [0,1]
        b04_n = 1 - max(0.0, min(1.0, b04 / 3000.0))   # darker = higher
        score = (cfg["weight_nbr"] * nbr_n + cfg["weight_ndvi"] * ndvi
                 + cfg["weight_b04"] * b04_n)
        rows.append({
            "date": pd.to_datetime(p["date"]),
            "NDVI": ndvi, "NDRE": p.get("NDRE"),
            "NBR": nbr, "NDSI": ndsi,
            "RECI": p.get("RECI"), "B04": b04,
            "score": score,
        })
    if not rows:
        return pd.DataFrame(columns=["date"] + INDEX_COLS)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # multiple granules can cover the polygon on the same day; keep one
    df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)
    return df


def load_model(model_path=None):
    """Load the trained model artifact: {'model', 'feature_names', ...meta}."""
    path = os.path.abspath(model_path or MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at {path}. "
            "Run tools/train_model.py to create it."
        )
    return joblib.load(path)


def predict_event(scene_df, artifact):
    """
    Score one polygon's scene series and pick the most probable event scene.

    Returns dict with event_date, date_start, date_end, probability,
    or None when the series is too short.
    """
    g = scene_df.dropna(subset=INDEX_COLS).sort_values("date").reset_index(drop=True)
    if len(g) < MIN_SCENES:
        return None
    F = build_features(g)
    X = F[artifact["feature_names"]]
    prob = artifact["model"].predict_proba(X)[:, 1]
    i = int(np.argmax(prob))
    event_date = g["date"].iloc[i]
    # the anomaly is first visible at scene i, so the event happened between
    # the previous surviving scene and this one
    date_start = g["date"].iloc[i - 1] if i > 0 else event_date
    return {
        "event_date": event_date,
        "date_start": date_start,
        "date_end": event_date,
        "probability": float(prob[i]),
    }


def _confidence(prob):
    if prob >= 0.5:
        return "High"
    if prob >= 0.25:
        return "Medium"
    return "Low"


def run(
    polygons_shp: str,
    output_shp: str,
    ign_date_str: str,
    gee_project: str,
    gee_credentials: str = None,
    params: dict = None,
    log=print,
    progress_callback=None,
) -> str:
    """
    Run debris flow date detection.

    Args:
        polygons_shp: Path to polygons.shp from Step 1
        output_shp: Path for output timepolygons.shp
        ign_date_str: Fire ignition date as MM/DD/YYYY
        gee_project: GEE cloud project ID
        gee_credentials: Optional GEE credentials JSON string
        params: Optional dict overriding detection parameters
        log: Logging function
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        Path to output shapefile
    """
    cfg = {**DEFAULTS}
    if params:
        cfg.update(params)

    ign_date = datetime.strptime(ign_date_str, "%m/%d/%Y")
    if ign_date < S2_START:
        raise ValueError(
            f"Ignition date {ign_date_str} predates Sentinel-2 "
            f"({S2_START:%Y-%m-%d}). SCAR v2 is Sentinel-2 only -- use the "
            "v1 app with the Landsat backend for earlier fires."
        )

    artifact = load_model(cfg.get("model_path"))
    log(f"Loaded model: trained {artifact.get('trained', '?')}, "
        f"{len(artifact['feature_names'])} features")

    # Initialize GEE
    if gee_credentials:
        import json
        import google.oauth2.credentials
        from ee import oauth as ee_oauth
        creds_data = json.loads(gee_credentials)
        credentials = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=creds_data["refresh_token"],
            token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=creds_data.get("client_id", ee_oauth.CLIENT_ID),
            client_secret=creds_data.get("client_secret", ee_oauth.CLIENT_SECRET),
        )
        ee.Initialize(credentials=credentials, project=gee_project)
    else:
        try:
            ee.Initialize(project=gee_project)
        except Exception:
            log("GEE not initialized -- attempting authentication...")
            ee.Authenticate()
            ee.Initialize(project=gee_project)

    # Search window
    search_start = (ign_date + timedelta(days=cfg["post_fire_buffer_days"])).strftime("%Y-%m-%d")
    max_end = ign_date + timedelta(days=cfg["search_years"] * 365)
    search_end = min(max_end, datetime.now()).strftime("%Y-%m-%d")

    gdf = gpd.read_file(polygons_shp)
    gdf = gdf.to_crs("EPSG:4326")

    for field in [FIELD_EVENT_DATE, FIELD_START, FIELD_END]:
        if field not in gdf.columns:
            gdf[field] = None
    if FIELD_CONFIDENCE not in gdf.columns:
        gdf[FIELD_CONFIDENCE] = ""
    if FIELD_CHG_SCORE not in gdf.columns:
        gdf[FIELD_CHG_SCORE] = np.nan

    log(f"Processing {len(gdf)} polygons")
    log(f"Date range: {search_start} to {search_end}")

    # threaded pulls: the run is bottlenecked on blocking GEE round-trips
    def pull_one(idx):
        return _pull_scene_series(gdf.geometry.iloc[idx], search_start, search_end, cfg)

    done = 0
    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
        futures = {pool.submit(pull_one, idx): idx for idx in range(len(gdf))}
        for fut in as_completed(futures):
            idx = futures[fut]
            done += 1
            if progress_callback:
                progress_callback(done, len(gdf))
            # one bad geometry or rate-limit blip should not abort the run
            try:
                scenes = fut.result()
            except Exception as e:
                log(f"  Polygon {idx}: FAILED to pull time series -- {e}")
                continue

            event = predict_event(scenes, artifact)
            if event is None:
                log(f"  Polygon {idx}: only {len(scenes)} usable scenes "
                    f"(< {MIN_SCENES}) -- no detection")
                continue

            prob = event["probability"]
            confidence = _confidence(prob)
            gap_days = (event["date_end"] - event["date_start"]).days
            log(f"  Polygon {idx}: Event {event['event_date']:%Y-%m-%d} "
                f"[{gap_days}d scene gap] (prob: {prob:.3f}, conf: {confidence}, "
                f"{len(scenes)} scenes)")

            gdf.at[idx, FIELD_EVENT_DATE] = event["event_date"].strftime("%Y-%m-%d")
            gdf.at[idx, FIELD_START] = event["date_start"].strftime("%Y-%m-%d")
            gdf.at[idx, FIELD_END] = event["date_end"].strftime("%Y-%m-%d")
            gdf.at[idx, FIELD_CONFIDENCE] = confidence
            gdf.at[idx, FIELD_CHG_SCORE] = round(prob, 4)

    if progress_callback:
        progress_callback(len(gdf), len(gdf))

    os.makedirs(os.path.dirname(output_shp), exist_ok=True)
    gdf.to_file(output_shp)

    detected = gdf[FIELD_EVENT_DATE].notna().sum()
    log(f"Events detected: {detected} / {len(gdf)} polygons")
    log(f"Saved to: {output_shp}")

    return output_shp
