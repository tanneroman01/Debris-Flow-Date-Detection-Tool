"""
Step 3: Debris flow date detection using Google Earth Engine (Landsat variant).

Pulls harmonized Landsat 5/7/8 time series over each deposit polygon, computes
composite change scores (NBR + NDVI + RED), and detects the first significant
change event.

Landsat coverage begins ~1984, making this suitable for fires from ~2010 onward
(where Sentinel-2, starting mid-2015, is too late).
"""

import os
from datetime import datetime, timedelta
from dateutil import parser
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import ee

# ---------- Default detection parameters ----------
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
    "post_fire_buffer_days": 270,
    "local_ref_inner_buffer_m": 50,
    "local_ref_outer_buffer_m": 500,
    "local_ref_scale": 30,
    "local_ref_min_area_m2": 50000,
    "refine_events": True,
    "refine_margin_days": 15,
    "refine_scale": 30,
    # PRISM precipitation validation
    "prism_dataset": "OREGONSTATE/PRISM/AN81d",
    "precip_min_mm": 10.0,
    "precip_window_pad_days": 5,
}

# Output field names
FIELD_EVENT_DATE = "EVENT_DATE"
FIELD_START = "DATE_START"
FIELD_END = "DATE_END"
FIELD_CONFIDENCE = "CONFIDENCE"
FIELD_CHG_SCORE = "CHG_SCORE"

# Landsat band mappings: satellite-native names → common names
# TM/ETM+ (Landsat 5, 7)
_TM_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
# OLI (Landsat 8, 9)
_OLI_BANDS = ["SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
# Common target names (order matches both lists above)
_COMMON_NAMES = ["GREEN", "RED", "NIR", "SWIR1", "SWIR2"]

# Landsat C2 surface reflectance scale factor
_SCALE_FACTOR = 0.0000275
_SCALE_OFFSET = -0.174


def _mask_clouds_qa(image):
    """Mask clouds and cloud shadows using Landsat QA_PIXEL band."""
    qa = image.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 3).eq(0)
    cloud = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud_shadow.And(cloud))


def _prep_tm(image):
    """Prepare a Landsat 5/7 TM/ETM+ image: mask clouds, rename, scale."""
    masked = _mask_clouds_qa(image)
    scaled = masked.select(_TM_BANDS, _COMMON_NAMES).multiply(_SCALE_FACTOR).add(_SCALE_OFFSET)
    return scaled.copyProperties(image, ["system:time_start", "system:time_end"])


def _prep_oli(image):
    """Prepare a Landsat 8/9 OLI image: mask clouds, rename, scale."""
    masked = _mask_clouds_qa(image)
    scaled = masked.select(_OLI_BANDS, _COMMON_NAMES).multiply(_SCALE_FACTOR).add(_SCALE_OFFSET)
    return scaled.copyProperties(image, ["system:time_start", "system:time_end"])


def _build_landsat_collection(search_start, search_end, ee_polygon, cfg):
    """
    Build a merged, harmonized Landsat collection spanning L5 + L7 + L8.

    All images are cloud-masked, band-renamed to common names, and scaled
    to surface reflectance (approximately 0–1).
    """
    cloud_max = cfg["cloud_cover_max"]
    collections = []

    # Landsat 5 TM: operational through ~2012-05
    if search_start < "2012-05-01":
        l5_end = min(search_end, "2012-05-05")
        l5 = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterDate(search_start, l5_end)
            .filterBounds(ee_polygon)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
            .map(_prep_tm)
        )
        collections.append(l5)

    # Landsat 7 ETM+: operational 1999–present (SLC-off striping from 2003,
    # mitigated by median compositing over 30-day windows)
    l7 = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        .filterDate(search_start, search_end)
        .filterBounds(ee_polygon)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
        .map(_prep_tm)
    )
    collections.append(l7)

    # Landsat 8 OLI: operational 2013-03-18+
    if search_end > "2013-03-18":
        l8_start = max(search_start, "2013-03-18")
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(l8_start, search_end)
            .filterBounds(ee_polygon)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
            .map(_prep_oli)
        )
        collections.append(l8)

    # Merge all available collections
    merged = collections[0]
    for c in collections[1:]:
        merged = merged.merge(c)

    return merged


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
            composite.normalizedDifference(["NIR", "SWIR2"]).rename("NBR"),
            ee.Image.constant(0).rename("NBR"),
        )
        ndvi = ee.Algorithms.If(
            count.gt(0),
            composite.normalizedDifference(["NIR", "RED"]).rename("NDVI"),
            ee.Image.constant(0).rename("NDVI"),
        )
        ndsi = ee.Algorithms.If(
            count.gt(0),
            composite.normalizedDifference(["GREEN", "SWIR1"]).rename("NDSI"),
            ee.Image.constant(-1).rename("NDSI"),
        )

        selected = ee.Algorithms.If(
            count.gt(0),
            composite.select(["RED", "NIR", "SWIR2"])
            .addBands(ee.Image(nbr))
            .addBands(ee.Image(ndvi))
            .addBands(ee.Image(ndsi)),
            ee.Image.constant([0, 0, 0, 0, 0, -1]).rename(
                ["RED", "NIR", "SWIR2", "NBR", "NDVI", "NDSI"]
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
                "B04": stats.get("RED"),
                "B08": stats.get("NIR"),
                "B12": stats.get("SWIR2"),
                "NBR": stats.get("NBR"),
                "NDVI": stats.get("NDVI"),
                "NDSI": stats.get("NDSI"),
                "count": count,
            },
        )

    results = ee.FeatureCollection(intervals.map(calculate_interval_stats))
    return results.getInfo()["features"]


def _shapely_to_ee_geometry(geom):
    """Convert a shapely Polygon or MultiPolygon to an ee.Geometry."""
    if geom.geom_type == "Polygon":
        return ee.Geometry.Polygon(
            [[list(c) for c in geom.exterior.coords]]
            + [[list(c) for c in ring.coords] for ring in geom.interiors]
        )
    if geom.geom_type == "MultiPolygon":
        parts = []
        for part in geom.geoms:
            parts.append(
                [[list(c) for c in part.exterior.coords]]
                + [[list(c) for c in ring.coords] for ring in part.interiors]
            )
        return ee.Geometry.MultiPolygon(parts)
    raise TypeError(f"Unsupported geometry type for GEE: {geom.geom_type}")


def get_gee_timeseries(geom, search_start, search_end, cfg, scale=30):
    try:
        ee_polygon = _shapely_to_ee_geometry(geom)

        collection = _build_landsat_collection(
            search_start, search_end, ee_polygon, cfg
        )

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

    # NBR and NDVI are normalized differences (-1 to 1) -- same as Sentinel-2
    nbr_norm = (nbr_vals + 1.0) / 2.0
    ndvi_norm = (ndvi_vals + 1.0) / 2.0
    # Landsat SR is already scaled to ~0-1 reflectance; red in burned areas ~0.03-0.15
    b04_norm = 1.0 - np.clip(b04_vals / 0.15, 0, 1)

    composite = (
        cfg["weight_dnbr"] * nbr_norm
        + cfg["weight_ndvi"] * ndvi_norm
        + cfg["weight_b04"] * b04_norm
    )

    for i, r in enumerate(results):
        r["score"] = composite[i]

    return results


def _fetch_individual_acquisitions(ee_polygon, start_str, end_str, cfg):
    """Pull per-acquisition polygon-mean NBR/NDVI/RED/NDSI (no compositing)."""
    collection = _build_landsat_collection(start_str, end_str, ee_polygon, cfg)

    def per_image_stats(image):
        nbr = image.normalizedDifference(["NIR", "SWIR2"]).rename("NBR")
        ndvi = image.normalizedDifference(["NIR", "RED"]).rename("NDVI")
        ndsi = image.normalizedDifference(["GREEN", "SWIR1"]).rename("NDSI")
        combined = (
            image.select(["RED"]).addBands(nbr).addBands(ndvi).addBands(ndsi)
        )
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_polygon,
            scale=cfg["refine_scale"],
            maxPixels=1e7,
        )
        return ee.Feature(
            None,
            {
                "date": image.date().format("YYYY-MM-dd"),
                "B04": stats.get("RED"),
                "NBR": stats.get("NBR"),
                "NDVI": stats.get("NDVI"),
                "NDSI": stats.get("NDSI"),
            },
        )

    feats = collection.map(per_image_stats).getInfo()["features"]
    return [f["properties"] for f in feats]


def refine_event_window(geom, coarse_iv_start, coarse_iv_end, cfg):
    """
    Tighten a coarse event window using individual Landsat acquisitions.

    Pulls every cloud-free acquisition in the coarse window (extended by
    ``refine_margin_days`` on both sides), scores each one with the same
    composite weights used for the coarse pass, and returns the adjacent
    pair with the largest jump in score.
    """
    try:
        ee_polygon = _shapely_to_ee_geometry(geom)
        margin = cfg.get("refine_margin_days", 15)
        start_str = (coarse_iv_start - timedelta(days=margin)).strftime("%Y-%m-%d")
        end_str = (coarse_iv_end + timedelta(days=margin)).strftime("%Y-%m-%d")

        acquisitions = _fetch_individual_acquisitions(
            ee_polygon, start_str, end_str, cfg
        )
        if not acquisitions or len(acquisitions) < 2:
            return None

        scored = []
        for acq in acquisitions:
            date_str = acq.get("date")
            if not date_str:
                continue
            nbr = acq.get("NBR")
            ndvi = acq.get("NDVI")
            b04 = acq.get("B04")
            ndsi = acq.get("NDSI")
            if nbr is None and ndvi is None:
                continue
            if ndsi is not None and ndsi > cfg["ndsi_snow_threshold"]:
                continue
            nbr_v = nbr if nbr is not None else 0.0
            ndvi_v = ndvi if ndvi is not None else 0.0
            b04_v = b04 if b04 is not None else 0.0
            nbr_norm = (nbr_v + 1.0) / 2.0
            ndvi_norm = (ndvi_v + 1.0) / 2.0
            b04_norm = 1.0 - max(0.0, min(1.0, b04_v / 0.15))
            score = (
                cfg["weight_dnbr"] * nbr_norm
                + cfg["weight_ndvi"] * ndvi_norm
                + cfg["weight_b04"] * b04_norm
            )
            try:
                dt = parser.parse(date_str)
            except Exception:
                continue
            scored.append((dt, score))

        if len(scored) < 2:
            return None

        scored.sort(key=lambda x: x[0])

        max_drop = 0.0
        max_idx = -1
        for i in range(len(scored) - 1):
            drop = scored[i][1] - scored[i + 1][1]  # positive when score drops
            if drop > max_drop:
                max_drop = drop
                max_idx = i

        if max_idx < 0:
            return None

        pre_dt = scored[max_idx][0]
        post_dt = scored[max_idx + 1][0]
        event_dt = pre_dt + (post_dt - pre_dt) / 2
        return pre_dt, post_dt, event_dt

    except Exception:
        return None


def _validate_precip(geom, coarse_iv_start, coarse_iv_end, cfg):
    """Check whether significant precipitation occurred in a coarse event window.

    Returns True if any day in the window has precip >= precip_min_mm,
    or True on query failure (fail-open).
    """
    try:
        centroid = geom.centroid
        ee_point = ee.Geometry.Point([centroid.x, centroid.y])

        pad = cfg.get("precip_window_pad_days", 5)
        start_str = (coarse_iv_start - timedelta(days=pad)).strftime("%Y-%m-%d")
        end_str = (coarse_iv_end + timedelta(days=pad + 1)).strftime("%Y-%m-%d")

        collection = (
            ee.ImageCollection(cfg["prism_dataset"])
            .filterDate(start_str, end_str)
            .select("ppt")
        )

        def per_image_ppt(image):
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_point,
                scale=4000,
                maxPixels=1e6,
            )
            return ee.Feature(None, {"ppt": stats.get("ppt")})

        feats = collection.map(per_image_ppt).getInfo()["features"]
        if not feats:
            return True  # fail-open

        min_mm = cfg.get("precip_min_mm", 10.0)
        for f in feats:
            ppt = f["properties"].get("ppt")
            if ppt is not None and ppt >= min_mm:
                return True

        return False

    except Exception:
        return True  # fail-open


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


def compute_confidence(change_score, threshold, obs_count):
    strong_count = 0
    if change_score > threshold * 2:
        strong_count += 1
    if obs_count is not None and obs_count >= 3:
        strong_count += 1

    if strong_count >= 2:
        return "High"
    elif strong_count >= 1:
        return "Medium"
    else:
        return "Low"


def detect_change_event(ts, cfg, ref_std=None, geom=None):
    scores = compute_change_scores(ts, cfg)
    if scores is None or len(scores) < 3:
        return None

    raw_scores = np.array([s["score"] for s in scores])
    diffs = np.diff(raw_scores)

    if ref_std is not None and ref_std > 1e-6:
        threshold = ref_std * cfg["baseline_multiplier"]
    else:
        threshold = cfg["fallback_abs_threshold"]

    # Collect coarse candidates (DOWN exceedances) before refinement
    candidates = []
    for i in range(len(diffs)):
        if diffs[i] >= 0 or abs(diffs[i]) < threshold:
            continue
        coarse_iv_start = parser.parse(scores[i]["interval_start"])
        coarse_iv_end = parser.parse(scores[i + 1]["interval_end"])
        candidates.append((i, coarse_iv_start, coarse_iv_end))

    if not candidates:
        return None

    # Precipitation validation: accept candidates with significant rainfall,
    # fall back to first candidate if none pass.
    validated = candidates
    if geom is not None and cfg.get("precip_min_mm", 0) > 0:
        passed = [
            c for c in candidates
            if _validate_precip(geom, c[1], c[2], cfg)
        ]
        if passed:
            validated = passed

    # Build final events from validated candidates (with refinement)
    all_events = []
    for i, coarse_iv_start, coarse_iv_end in validated:
        candidate_start = coarse_iv_start
        candidate_end = coarse_iv_end
        candidate_date = scores[i + 1]["end_dt"]
        refined = False

        change_score = float(abs(diffs[i]))
        obs_count = scores[i + 1].get("count", 0)

        # Two-pass refinement: find the specific adjacent pair of clear
        # acquisitions that straddle the jump. Localizes the event from
        # ~60 days to the Landsat revisit cadence (~8-16 days).
        if geom is not None and cfg.get("refine_events", True):
            r = refine_event_window(geom, coarse_iv_start, coarse_iv_end, cfg)
            if r is not None:
                pre_dt, post_dt, event_dt = r
                candidate_start = pre_dt
                candidate_end = post_dt
                candidate_date = event_dt
                refined = True

        confidence = compute_confidence(change_score, threshold, obs_count)

        all_events.append(
            (
                candidate_date,
                candidate_start,
                candidate_end,
                change_score,
                confidence,
                refined,
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
    gee_credentials: str = None,
    params: dict = None,
    log=print,
    progress_callback=None,
) -> str:
    """
    Run debris flow date detection using harmonized Landsat 5/7/8 imagery.

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
    for field in [FIELD_CHG_SCORE]:
        if field not in gdf.columns:
            gdf[field] = np.nan

    indices_to_process = list(range(len(gdf)))

    log(f"Processing {len(indices_to_process)} polygons (Landsat)")
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
            ts = get_gee_timeseries(geom, search_start, search_end, cfg, scale=60)
        if ts is None:
            ts = get_gee_timeseries(geom, search_start, search_end, cfg, scale=120)
        if ts is None:
            log(f"  Polygon {idx}: No data available")
            continue

        # Detect events
        all_events = detect_change_event(ts, cfg, ref_std=ref_std, geom=geom)
        event = all_events[0] if all_events else None

        if event is None:
            log(f"  Polygon {idx}: No valid event detected")
            continue

        event_date, start_date, end_date, change_score, confidence, refined = event
        window_days = (end_date - start_date).days
        tag = "refined" if refined else "coarse"

        if all_events and len(all_events) > 1:
            other_dates = [e[0].strftime("%Y-%m-%d") for e in all_events[1:]]
            log(
                f"  Polygon {idx}: Event {event_date.strftime('%Y-%m-%d')} "
                f"[{tag}, +/-{window_days}d] "
                f"(score: {change_score:.4f}, conf: {confidence}) "
                f"[+{len(all_events)-1} more: {', '.join(other_dates)}]"
            )
        else:
            log(
                f"  Polygon {idx}: Event {event_date.strftime('%Y-%m-%d')} "
                f"[{tag}, +/-{window_days}d] "
                f"(score: {change_score:.4f}, conf: {confidence})"
            )

        gdf.at[idx, FIELD_EVENT_DATE] = event_date.strftime("%Y-%m-%d")
        gdf.at[idx, FIELD_START] = start_date.strftime("%Y-%m-%d")
        gdf.at[idx, FIELD_END] = end_date.strftime("%Y-%m-%d")
        gdf.at[idx, FIELD_CONFIDENCE] = confidence
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
