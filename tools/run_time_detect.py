"""
Standalone driver for Step 3 only (date detection).

Runs either pipeline.time_detect or pipeline.time_detect_landsat against a
polygon shapefile and writes the detection output to another shapefile.
Lets us iterate on time_detect.py without running KML conversion,
attribution, or merging.

Example:
    python tools/run_time_detect.py \\
        --input path/to/polygons.shp \\
        --ign-date 08/10/2020 \\
        --project my-gee-project \\
        --sensor auto \\
        --output path/to/detected.shp
"""

import argparse
import os
import sys
from datetime import datetime

# Make the pipeline package importable when run as a script
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _APP_DIR)

from pipeline import time_detect, time_detect_landsat  # noqa: E402


def pick_sensor(ign_date_str, user_choice):
    if user_choice != "auto":
        return user_choice
    # Sentinel-2 SR collection begins ~2015-06-23. Give it a cushion: require
    # imagery available ~9 months after the fire (matches post_fire_buffer_days
    # default), so ignitions before ~2014-10 should use Landsat.
    ign = datetime.strptime(ign_date_str, "%m/%d/%Y")
    return "sentinel2" if ign >= datetime(2014, 10, 1) else "landsat"


def load_credentials(path):
    if not path:
        return None
    with open(path, "r") as f:
        return f.read()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Polygon shapefile")
    p.add_argument("--ign-date", required=True, help="Fire ignition date MM/DD/YYYY")
    p.add_argument("--project", required=True, help="Google Earth Engine project ID")
    p.add_argument(
        "--sensor",
        choices=["auto", "sentinel2", "landsat"],
        default="auto",
        help="Satellite to use (auto picks based on ignition date)",
    )
    p.add_argument(
        "--credentials",
        default=None,
        help="Path to a JSON credentials file (contents of ~/.config/earthengine/credentials)",
    )
    p.add_argument("--output", required=True, help="Output detection shapefile")
    p.add_argument(
        "--event-selection",
        choices=["first", "max_score", "max_precip"],
        default="first",
        help="How to pick the reported event from multiple candidates",
    )
    args = p.parse_args()

    sensor = pick_sensor(args.ign_date, args.sensor)
    module = time_detect_landsat if sensor == "landsat" else time_detect
    params = {"event_selection": args.event_selection}

    print(f"[run_time_detect] sensor={sensor}  ign_date={args.ign_date}  event_selection={args.event_selection}  input={args.input}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    module.run(
        polygons_shp=args.input,
        output_shp=args.output,
        ign_date_str=args.ign_date,
        gee_project=args.project,
        gee_credentials=load_credentials(args.credentials),
        params=params,
        log=print,
    )
    print(f"[run_time_detect] wrote {args.output}")


if __name__ == "__main__":
    main()
