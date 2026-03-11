"""
Pipeline orchestrator -- wires Steps 1-4 together.
"""

import os
import json
import tempfile

from pipeline import kml_to_shp, attributer, time_detect, merger


def run_full_pipeline(
    kml_path: str,
    fire_boundary_path: str,
    roads_shp: str,
    template_shp: str,
    fire_defaults_path: str,
    fire_key: str,
    obs_date: str,
    gee_project: str,
    output_dir: str,
    detection_params: dict = None,
    log=print,
    progress_callback=None,
) -> str:
    """
    Run the complete debris flow pipeline.

    Args:
        kml_path: Path to uploaded KML file
        fire_boundary_path: Path to fire boundary shapefile
        roads_shp: Path to OSM roads shapefile
        template_shp: Path to CDOT shapefile template
        fire_defaults_path: Path to fire_defaults.json
        fire_key: Key in fire_defaults.json (e.g. "DECKER2019")
        obs_date: Observation date as YYYY-MM-DD
        gee_project: GEE cloud project ID
        output_dir: Directory for all outputs
        detection_params: Optional dict overriding TIME detection parameters
        log: Logging function
        progress_callback: Optional callable(step, step_name, pct) for UI updates

    Returns:
        Path to final merged_points.shp
    """
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(work_dir, exist_ok=True)

    # Load fire defaults
    with open(fire_defaults_path, "r") as f:
        fire_db = json.load(f)

    if fire_key not in fire_db:
        raise KeyError(
            f"'{fire_key}' not found in fire_defaults.json. "
            f"Available: {[k for k in fire_db if not k.startswith('_')]}"
        )

    fire_entry = fire_db[fire_key]
    if fire_entry.get("IGN_DATE") == "TODO":
        raise ValueError(f"IGN_DATE for '{fire_key}' is still TODO -- fill it in first.")

    defaults = {"OBS_DATE": obs_date, **fire_db["_constants"], **fire_entry}

    def update_progress(step, name, pct=None):
        if progress_callback:
            progress_callback(step, name, pct)

    # ── Step 1: KML to Shapefile ──
    update_progress(1, "Converting KML to shapefile")
    log("=" * 50)
    log("STEP 1: KML to Shapefile")
    log("=" * 50)
    polygons_shp = kml_to_shp.run(kml_path, work_dir, log=log)

    # ── Step 2: Build Attributes ──
    update_progress(2, "Computing spatial attributes")
    log("\n" + "=" * 50)
    log("STEP 2: Build Attributes")
    log("=" * 50)
    polys_out, points_shp = attributer.run(
        polygons_shp=polygons_shp,
        fire_boundary_shp=fire_boundary_path,
        roads_shp=roads_shp,
        template_shp=template_shp,
        fire_defaults=defaults,
        output_dir=work_dir,
        log=log,
    )

    # ── Step 3: Date Detection ──
    update_progress(3, "Detecting debris flow dates (GEE)")
    log("\n" + "=" * 50)
    log("STEP 3: Date Detection")
    log("=" * 50)
    time_shp = os.path.join(work_dir, "timepolygons.shp")

    def time_progress(current, total):
        if progress_callback:
            pct = current / total if total > 0 else 0
            progress_callback(3, f"Processing polygon {current}/{total}", pct)

    time_detect.run(
        polygons_shp=polygons_shp,
        output_shp=time_shp,
        ign_date_str=fire_entry["IGN_DATE"],
        gee_project=gee_project,
        params=detection_params,
        log=log,
        progress_callback=time_progress,
    )

    # ── Step 4: Merge ──
    update_progress(4, "Merging results")
    log("\n" + "=" * 50)
    log("STEP 4: Merge & Finalize")
    log("=" * 50)
    merged_shp = os.path.join(output_dir, "merged_points.shp")
    merger.run(
        points_shp=points_shp,
        time_shp=time_shp,
        output_shp=merged_shp,
        log=log,
    )

    update_progress(4, "Complete", 1.0)
    log("\n" + "=" * 50)
    log("PIPELINE COMPLETE")
    log(f"Output: {merged_shp}")
    log("=" * 50)

    return merged_shp
