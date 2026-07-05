"""
Validate time_detect output against a known-date polygon shapefile.

Expected input: a polygon shapefile carrying ``EVENT_DATE`` (ground truth,
ISO ``YYYY-MM-DD``), ``FIRE``, and ``FIRE_YEAR`` columns -- e.g. the output
of ``prep_known_dates.py``.

For each unique (FIRE, FIRE_YEAR) group the driver:
    1. Looks up the ignition date (from ``tools/known_fire_dates.json`` or
       ``data/fire_defaults.json`` fallback).
    2. Picks Sentinel-2 vs Landsat based on the ignition year.
    3. Writes the polygon subset to a scratch shapefile.
    4. Calls ``time_detect.run`` / ``time_detect_landsat.run`` directly.
    5. Reads the detection output back and joins predicted date/window to
       the known date by row order.

At the end it prints a per-polygon table and summary stats: detection
rate, |delta_days|, and whether the known date falls inside the reported
``[DATE_START, DATE_END]`` window.

Example:
    python tools/validate_known_dates.py \\
        --input "C:/.../known_dates_polygons.shp" \\
        --project my-gee-project \\
        --credentials ~/.config/earthengine/credentials \\
        --work-dir ./validation_runs
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

import geopandas as gpd
import pandas as pd

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _APP_DIR)

from pipeline import time_detect, time_detect_landsat  # noqa: E402

DEFAULT_FIRE_DEFAULTS = os.path.join(_APP_DIR, "data", "fire_defaults.json")
DEFAULT_KNOWN_DATES = os.path.join(
    os.path.dirname(__file__), "known_fire_dates.json"
)


def load_ignition_lookup(known_dates_path, fire_defaults_path):
    """Build FIRENAME|FIRE_YEAR -> 'MM/DD/YYYY' from two sources."""
    lookup = {}

    if os.path.exists(fire_defaults_path):
        with open(fire_defaults_path, "r") as f:
            db = json.load(f)
        for k, v in db.items():
            if k.startswith("_"):
                continue
            name = v.get("FIRENAME")
            year = str(v.get("FIRE_YEAR", "")).strip()
            ign = v.get("IGN_DATE")
            if name and year and ign and ign != "TODO":
                lookup[f"{name}|{year}"] = ign

    if os.path.exists(known_dates_path):
        with open(known_dates_path, "r") as f:
            user_db = json.load(f)
        for k, v in user_db.items():
            if k.startswith("_"):
                continue
            if v and v not in ("TODO", "todo"):
                lookup[k] = v

    return lookup


def pick_sensor(ign_date_str):
    ign = datetime.strptime(ign_date_str, "%m/%d/%Y")
    return "sentinel2" if ign >= datetime(2014, 10, 1) else "landsat"


def load_credentials(path):
    if not path:
        return None
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        return f.read()


def run_fire_group(group_gdf, ign_date, project, credentials, work_dir, log):
    """Write the group to a scratch shapefile and run the appropriate detector."""
    os.makedirs(work_dir, exist_ok=True)
    input_shp = os.path.join(work_dir, "input.shp")
    output_shp = os.path.join(work_dir, "detected.shp")

    # Write the subset with a stable integer ID so we can align the detection
    # output back to the input rows regardless of how time_detect reorders.
    subset = group_gdf.copy().reset_index(drop=True)
    subset["VID"] = range(len(subset))
    subset.to_file(input_shp)

    sensor = pick_sensor(ign_date)
    module = time_detect_landsat if sensor == "landsat" else time_detect
    log(f"  sensor={sensor}  ign_date={ign_date}  polygons={len(subset)}")

    module.run(
        polygons_shp=input_shp,
        output_shp=output_shp,
        ign_date_str=ign_date,
        gee_project=project,
        gee_credentials=credentials,
        log=log,
    )

    detected = gpd.read_file(output_shp)
    return subset, detected, sensor


def _parse_iso(s):
    if s is None:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def compare(known_subset, detected):
    """Row-aligned comparison of known vs detected dates for one fire group."""
    merged = pd.DataFrame({
        "VID": known_subset["VID"].values,
        "FIRE": known_subset["FIRE"].values,
        "FIRE_YEAR": known_subset["FIRE_YEAR"].values,
        "KNOWN_DATE": known_subset["EVENT_DATE"].values,
        "DETECTED": detected.get("EVENT_DATE", pd.Series([None] * len(known_subset))).values,
        "DATE_START": detected.get("DATE_START", pd.Series([None] * len(known_subset))).values,
        "DATE_END": detected.get("DATE_END", pd.Series([None] * len(known_subset))).values,
        "CONFIDENCE": detected.get("CONFIDENCE", pd.Series([None] * len(known_subset))).values,
    })

    deltas = []
    in_window = []
    for _, row in merged.iterrows():
        k = _parse_iso(row["KNOWN_DATE"])
        d = _parse_iso(row["DETECTED"])
        s = _parse_iso(row["DATE_START"])
        e = _parse_iso(row["DATE_END"])
        if k is None or d is None:
            deltas.append(None)
        else:
            deltas.append((d - k).days)
        if k is not None and s is not None and e is not None:
            in_window.append(s <= k <= e)
        else:
            in_window.append(None)
    merged["DELTA_DAYS"] = deltas
    merged["IN_WINDOW"] = in_window
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="known-date polygon shapefile")
    p.add_argument("--project", required=True, help="Google Earth Engine project ID")
    p.add_argument("--credentials", default=None, help="Path to GEE credentials JSON")
    p.add_argument(
        "--fire-dates",
        default=DEFAULT_KNOWN_DATES,
        help="JSON of FIRENAME|FIRE_YEAR -> ignition date (MM/DD/YYYY)",
    )
    p.add_argument(
        "--fire-defaults",
        default=DEFAULT_FIRE_DEFAULTS,
        help="Pipeline fire_defaults.json fallback",
    )
    p.add_argument(
        "--work-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "validation_runs"),
        help="Where to write per-fire scratch files and combined comparison",
    )
    p.add_argument(
        "--only-fire",
        default=None,
        help="Optional 'FIRENAME|FIRE_YEAR' filter to run a single group",
    )
    args = p.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    credentials = load_credentials(args.credentials)
    lookup = load_ignition_lookup(args.fire_dates, args.fire_defaults)

    known = gpd.read_file(args.input)
    required = {"EVENT_DATE", "FIRE", "FIRE_YEAR"}
    missing = required - set(known.columns)
    if missing:
        sys.exit(f"Input shapefile missing required columns: {missing}")

    all_results = []
    for (fire, year), group in known.groupby(["FIRE", "FIRE_YEAR"]):
        key = f"{fire}|{year}"
        if args.only_fire and args.only_fire != key:
            continue

        print(f"\n=== {key}  ({len(group)} polygons) ===")
        ign_date = lookup.get(key)
        if not ign_date:
            print(f"  SKIP: no ignition date in lookup (fill in {args.fire_dates})")
            continue

        fire_work = os.path.join(work_dir, key.replace("|", "_").replace(" ", "_"))
        try:
            subset, detected, sensor = run_fire_group(
                group, ign_date, args.project, credentials, fire_work, log=print
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        df = compare(subset, detected)
        df["SENSOR"] = sensor
        df["IGN_DATE"] = ign_date
        all_results.append(df)

        found = df["DETECTED"].notna().sum()
        in_win = df["IN_WINDOW"].eq(True).sum()
        abs_deltas = df["DELTA_DAYS"].dropna().abs()
        med = abs_deltas.median() if len(abs_deltas) else float("nan")
        print(
            f"  detected {found}/{len(df)}  "
            f"|delta| median={med:.0f}d  "
            f"in-window {in_win}/{len(df)}"
        )

    if not all_results:
        print("\nNo fire groups were processed.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    out_csv = os.path.join(work_dir, "comparison.csv")
    combined.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(combined.to_string(index=False))
    total = len(combined)
    detected = combined["DETECTED"].notna().sum()
    in_window = combined["IN_WINDOW"].eq(True).sum()
    abs_deltas = combined["DELTA_DAYS"].dropna().abs()
    print()
    print(f"detection rate : {detected}/{total} ({100*detected/total:.0f}%)")
    if len(abs_deltas):
        print(
            f"|delta| days   : median={abs_deltas.median():.0f}  "
            f"mean={abs_deltas.mean():.0f}  "
            f"min={abs_deltas.min():.0f}  "
            f"max={abs_deltas.max():.0f}"
        )
    print(f"known in window: {in_window}/{total} ({100*in_window/total:.0f}%)")
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
