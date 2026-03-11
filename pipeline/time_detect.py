"""
Step 3: Debris flow date detection using Google Earth Engine.

Pulls Sentinel-2 time series over each deposit polygon, computes composite
change scores (NBR + NDVI + B04), and detects the first significant change
event with CHIRPS precipitation filtering.
"""

import os
from datetime import datetime, timedelta
from dateutil import parser
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import ee

# ---------- Default detection parameters (Run 7 best config) ----------
DEFAULTS = {
    "aggregation_interval": 30,
    "cloud_cover_max": 30,
    "ndsi_snow_threshold": 0.4,
    "active_season_months": [4, 5, 6, 7, 8, 9, 10, 11],
    "baseline_multiplier": 0.75,
    "fallback_abs_threshold": 0.06,
    "weight_dnbr": 0.35,
    "weight_ndvi": 0.45,
    "weight_b04": 0.20,
    "precip_window_days": 30,
    "precip_min_threshold": 10.0,
    "post_fire_buffer_days": 270,
    "local_ref_inner_buffer_m": 50,
    "local_ref_outer_buffer_m": 500,
    "local_ref_scale": 30,
    "local_ref_min_area_m2": 50000,
}

# Output field names
FIELD_EVENT_DATE = "EVENT_DATE"
FIELD_START = "DATE_START"
FIELD_END = "DATE_END"
FIELD_CONFIDENCE = "CONFIDENCE"
FIELD_PRECIP_MM = "PRECIP_MM"
FIELD_CHG_SCORE = "CHG_SCORE"


def _get_gee_timeseries_chunk(ee_polygon, collection, chunk_start, chunk_end, scale, cfg):
    start_date = ee.Date(chunk_start)
    end_date = ee.Date(chunk_end)
    total_days = end_date.difference(start_date, "day")
    num_intervals = total_days.divide(cfg["aggregation_interval"]).ceil()

    def make_interval(i):
        i = ee.Number(i)
        s = start_date.advance(i.multiply(cfg["aggregation_interval"]), "day")
        e = s.advance(cfg["aggregation_interval"], "day")
        return ee.List([s, e])

    intervals = ee.List.sequence(0, num_intervals.subtract(1)).map(make_interval)

    def calculate_interval_stats(date_range):
        date_range = ee.List(date_range)
        interval_start = ee.Date(date_range.get(0))
        interval_end = ee.Date(date_range.get(1))
        interval_collection = collection.filterDate(interval_start, interval_end)
        count = interval_collection.size()
        composite = interval_collection.median()

        nbr = ee.Algorithms.If(
            count.gt(0),
            composite.normalizedDifference(["B8", "B12"]).rename("NBR"),
            ee.Image.constant(0).rename("NBR"),
        )
        ndvi = ee.Algorithms.If(
            count.gt(0),
            composite.normalizedDifference(["B8", "B4"]).rename("NDVI"),
            ee.Image.constant(0).rename("NDVI"),
        )
        ndsi = ee.Algorithms.If(
            count.gt(0),
            composite.normalizedDifference(["B3", "B11"]).rename("NDSI"),
            ee.Image.constant(-1).rename("NDSI"),
        )

        selected = ee.Algorithms.If(
            count.gt(0),
            composite.select(["B4", "B8", "B12"])
            .addBands(ee.Image(nbr))
            .addBands(ee.Image(ndvi))
            .addBands(ee.Image(ndsi)),
            ee.Image.constant([0, 0, 0, 0, 0, -1]).rename(
                ["B4", "B8", "B12", "NBR", "NDVI", "NDSI"]
            ),
        )
        selected = ee.Image(selected)

        stats = selected.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_polygon,
            scale=scale,
            maxPixels=1e7,
        )

        return ee.Feature(
            None,
            {
                "interval_start": interval_start.format("YYYY-MM-dd"),
                "interval_end": interval_end.format("YYYY-MM-dd"),
                "B04": stats.get("B4"),
                "B08": stats.get("B8"),
                "B12": stats.get("B12"),
                "NBR": stats.get("NBR"),
                "NDVI": stats.get("NDVI"),
                "NDSI": stats.get("NDSI"),
                "count": count,
            },
        )

    results = ee.FeatureCollection(intervals.map(calculate_interval_stats))
    return results.getInfo()["features"]


def get_gee_timeseries(geom, search_start, search_end, cfg, scale=10):
    try:
        geom_coords = list(geom.exterior.coords)
        ee_polygon = ee.Geometry.Polygon(geom_coords)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(search_start, search_end)
            .filterBounds(ee_polygon)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["cloud_cover_max"]))
        )

        def mask_clouds_scl(image):
            scl = image.select("SCL")
            clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            return image.updateMask(clear)

        collection = collection.map(mask_clouds_scl)

        start_dt = datetime.strptime(search_start, "%Y-%m-%d")
        end_dt = datetime.strptime(search_end, "%Y-%m-%d")

        all_features = []
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=365), end_dt)
            features = _get_gee_timeseries_chunk(
                ee_polygon,
                collection,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                scale=scale,
                cfg=cfg,
            )
            all_features.extend(features)
            chunk_start = chunk_end

        formatted = []
        for feature in all_features:
            props = feature["properties"]
            cnt = props.get("count")
            if cnt is None or cnt == 0:
                continue
            formatted.append(
                {
                    "interval_start": props["interval_start"],
                    "interval_end": props["interval_end"],
                    "B04": props.get("B04"),
                    "B08": props.get("B08"),
                    "B12": props.get("B12"),
                    "NBR": props.get("NBR"),
                    "NDVI": props.get("NDVI"),
                    "NDSI": props.get("NDSI"),
                    "count": cnt,
                }
            )

        return formatted if formatted else None

    except Exception as e:
        return None


def compute_change_scores(ts, cfg):
    results = []
    for entry in ts:
        try:
            end_dt = parser.parse(entry["interval_end"])
        except Exception:
            continue
        if end_dt.month not in cfg["active_season_months"]:
            continue
        ndsi_val = entry.get("NDSI")
        if ndsi_val is not None and ndsi_val > cfg["ndsi_snow_threshold"]:
            continue
        nbr = entry.get("NBR")
        ndvi = entry.get("NDVI")
        b04 = entry.get("B04")
        if nbr is None and ndvi is None:
            continue
        results.append(
            {
                "interval_start": entry["interval_start"],
                "interval_end": entry["interval_end"],
                "end_dt": end_dt,
                "NBR": nbr if nbr is not None else 0.0,
                "NDVI": ndvi if ndvi is not None else 0.0,
                "B04": b04 if b04 is not None else 0.0,
                "count": entry.get("count", 0),
                "score": None,
            }
        )

    if len(results) < 3:
        return None

    nbr_vals = np.array([r["NBR"] for r in results])
    ndvi_vals = np.array([r["NDVI"] for r in results])
    b04_vals = np.array([r["B04"] for r in results])

    nbr_norm = (nbr_vals + 1.0) / 2.0
    ndvi_norm = (ndvi_vals + 1.0) / 2.0
    b04_norm = 1.0 - np.clip(b04_vals / 3000.0, 0, 1)

    composite = (
        cfg["weight_dnbr"] * nbr_norm
        + cfg["weight_ndvi"] * ndvi_norm
        + cfg["weight_b04"] * b04_norm
    )

    for i, r in enumerate(results):
        r["score"] = composite[i]

    return results


def get_chirps_precip(geom, event_date, window_days):
    try:
        centroid = geom.centroid
        ee_point = ee.Geometry.Point([centroid.x, centroid.y])
        end_date = event_date.strftime("%Y-%m-%d")
        start_date = (event_date - timedelta(days=window_days)).strftime("%Y-%m-%d")

        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start_date, end_date)
            .filterBounds(ee_point)
        )

        img_count = chirps.size().getInfo()
        if img_count == 0:
            return None

        total = chirps.sum().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_point.buffer(1000),
            scale=5000,
            maxPixels=1e6,
        )

        result = total.getInfo()
        precip = result.get("precipitation")
        return precip if precip is not None else None

    except Exception:
        return None


def get_local_reference_baseline(geom, gdf, idx, ign_date, search_start, cfg):
    try:
        import warnings
        import pyproj
        from shapely.ops import transform
        from shapely.geometry import MultiPolygon
        from functools import partial

        centroid = geom.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            to_utm = partial(
                pyproj.Transformer.from_crs(
                    "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True
                ).transform
            )
            to_wgs = partial(
                pyproj.Transformer.from_crs(
                    f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
                ).transform
            )

        geom_utm = transform(to_utm, geom)
        outer = transform(to_wgs, geom_utm.buffer(cfg["local_ref_outer_buffer_m"]))
        inner = transform(to_wgs, geom_utm.buffer(cfg["local_ref_inner_buffer_m"]))
        donut = outer.difference(inner)

        all_deposits = gdf.geometry.union_all()
        ref_area = donut.difference(all_deposits)

        if ref_area.is_empty:
            return None

        if isinstance(ref_area, MultiPolygon):
            ref_area = max(ref_area.geoms, key=lambda g: g.area)

        ref_area_utm = transform(to_utm, ref_area)
        if ref_area_utm.area < cfg["local_ref_min_area_m2"]:
            return None

        ref_end = (ign_date + timedelta(days=730)).strftime("%Y-%m-%d")
        ref_ts = get_gee_timeseries(
            ref_area,
            search_start=search_start,
            search_end=ref_end,
            cfg=cfg,
            scale=cfg["local_ref_scale"],
        )
        if ref_ts is None or len(ref_ts) < 5:
            return None

        scores = compute_change_scores(ref_ts, cfg)
        if scores is None or len(scores) < 3:
            return None

        vals = np.array([s["score"] for s in scores])
        diffs = np.diff(vals)
        ref_std = float(np.std(np.abs(diffs)))

        if ref_std < 1e-6:
            return None

        return ref_std

    except Exception:
        return None


def compute_confidence(change_score, threshold, precip_mm, obs_count):
    strong_count = 0
    if change_score > threshold * 2:
        strong_count += 1
    if precip_mm is not None and precip_mm > 15.0:
        strong_count += 1
    if obs_count is not None and obs_count >= 3:
        strong_count += 1

    if strong_count >= 3:
        return "High"
    elif strong_count >= 2:
        return "Medium"
    else:
        return "Low"


def detect_change_event(ts, cfg, ref_std=None, geom=None):
    scores = compute_change_scores(ts, cfg)
    if scores is None or len(scores) < 3:
        return None

    raw_scores = np.array([s["score"] for s in scores])
    diffs = np.abs(np.diff(raw_scores))

    if ref_std is not None and ref_std > 1e-6:
        threshold = ref_std * cfg["baseline_multiplier"]
    else:
        threshold = cfg["fallback_abs_threshold"]

    all_events = []
    for i in range(len(diffs)):
        if diffs[i] < threshold:
            continue

        candidate_date = scores[i + 1]["end_dt"]
        candidate_start = scores[i]["end_dt"]
        change_score = float(diffs[i])
        obs_count = scores[i + 1].get("count", 0)

        precip_mm = None
        if geom is not None:
            precip_mm = get_chirps_precip(
                geom, candidate_date, cfg["precip_window_days"]
            )
            if precip_mm is not None and precip_mm < cfg["precip_min_threshold"]:
                continue

        confidence = compute_confidence(change_score, threshold, precip_mm, obs_count)

        all_events.append(
            (
                candidate_date,
                candidate_start,
                candidate_date,
                change_score,
                precip_mm if precip_mm is not None else -1.0,
                confidence,
            )
        )

    if not all_events:
        return None
    return all_events


def run(
    polygons_shp: str,
    output_shp: str,
    ign_date_str: str,
    gee_project: str,
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
        params: Optional dict overriding detection parameters
        log: Logging function
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        Path to output shapefile
    """
    cfg = {**DEFAULTS}
    if params:
        cfg.update(params)

    # Initialize GEE
    try:
        ee.Initialize(project=gee_project)
    except Exception:
        log("GEE not initialized -- attempting authentication...")
        ee.Authenticate()
        ee.Initialize(project=gee_project)

    # Compute date range
    ign_date = datetime.strptime(ign_date_str, "%m/%d/%Y")
    search_start = (ign_date + timedelta(days=cfg["post_fire_buffer_days"])).strftime(
        "%Y-%m-%d"
    )
    max_end = ign_date + timedelta(days=5 * 365)
    now = datetime.now()
    search_end = min(max_end, now).strftime("%Y-%m-%d")

    # Load shapefile
    gdf = gpd.read_file(polygons_shp)
    gdf = gdf.to_crs("EPSG:4326")

    for field in [FIELD_EVENT_DATE, FIELD_START, FIELD_END]:
        if field not in gdf.columns:
            gdf[field] = None
    for field in [FIELD_CONFIDENCE]:
        if field not in gdf.columns:
            gdf[field] = ""
    for field in [FIELD_PRECIP_MM, FIELD_CHG_SCORE]:
        if field not in gdf.columns:
            gdf[field] = np.nan

    indices_to_process = list(range(len(gdf)))

    log(f"Processing {len(indices_to_process)} polygons")
    log(f"Date range: {search_start} to {search_end}")

    for count, idx in enumerate(indices_to_process):
        row = gdf.iloc[idx]
        geom = row.geometry

        if progress_callback:
            progress_callback(count, len(indices_to_process))

        # Local reference baseline
        ref_std = get_local_reference_baseline(
            geom, gdf, idx, ign_date, search_start, cfg
        )
        if ref_std is not None:
            ref_threshold = ref_std * cfg["baseline_multiplier"]
            log(
                f"  Polygon {idx}: local baseline std {ref_std:.4f} -> threshold {ref_threshold:.4f}"
            )
        else:
            ref_threshold = cfg["fallback_abs_threshold"]
            log(f"  Polygon {idx}: using fallback threshold {ref_threshold:.4f}")

        # Get time series (retry at coarser scales)
        ts = get_gee_timeseries(geom, search_start, search_end, cfg)
        if ts is None:
            ts = get_gee_timeseries(geom, search_start, search_end, cfg, scale=30)
        if ts is None:
            ts = get_gee_timeseries(geom, search_start, search_end, cfg, scale=60)
        if ts is None:
            log(f"  Polygon {idx}: No data available")
            continue

        # Detect events
        all_events = detect_change_event(ts, cfg, ref_std=ref_std, geom=geom)
        event = all_events[0] if all_events else None

        if event is None:
            log(f"  Polygon {idx}: No valid event detected")
            continue

        event_date, start_date, end_date, change_score, precip_mm, confidence = event

        if all_events and len(all_events) > 1:
            other_dates = [e[0].strftime("%Y-%m-%d") for e in all_events[1:]]
            log(
                f"  Polygon {idx}: Event {event_date.strftime('%Y-%m-%d')} "
                f"(score: {change_score:.4f}, precip: {precip_mm:.1f}mm, conf: {confidence}) "
                f"[+{len(all_events)-1} more: {', '.join(other_dates)}]"
            )
        else:
            log(
                f"  Polygon {idx}: Event {event_date.strftime('%Y-%m-%d')} "
                f"(score: {change_score:.4f}, precip: {precip_mm:.1f}mm, conf: {confidence})"
            )

        gdf.at[idx, FIELD_EVENT_DATE] = event_date.strftime("%Y-%m-%d")
        gdf.at[idx, FIELD_START] = start_date.strftime("%Y-%m-%d")
        gdf.at[idx, FIELD_END] = end_date.strftime("%Y-%m-%d")
        gdf.at[idx, FIELD_CONFIDENCE] = confidence
        gdf.at[idx, FIELD_PRECIP_MM] = (
            round(precip_mm, 1) if precip_mm >= 0 else None
        )
        gdf.at[idx, FIELD_CHG_SCORE] = round(change_score, 4)

    if progress_callback:
        progress_callback(len(indices_to_process), len(indices_to_process))

    # Save
    os.makedirs(os.path.dirname(output_shp), exist_ok=True)
    gdf.to_file(output_shp)

    detected = gdf[FIELD_EVENT_DATE].notna().sum()
    log(f"Events detected: {detected} / {len(indices_to_process)} polygons")
    log(f"Saved to: {output_shp}")

    return output_shp
