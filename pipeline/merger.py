"""
Step 4: Merge attributes and dates into final output.

Joins date information from TIME onto attribute-rich centroid points
from ATTRIBUTER, producing one record per deposit with all fields.
"""

import os
import geopandas as gpd
import pandas as pd


def run(
    points_shp: str,
    time_shp: str,
    output_shp: str,
    points_name_field: str = "PT_TYPE",
    time_name_field: str = "Name",
    log=print,
) -> str:
    """
    Merge points (Step 2) with time polygons (Step 3) into final shapefile.

    Returns:
        Path to merged output shapefile
    """
    pts = gpd.read_file(points_shp)
    log(f"Loaded {len(pts)} points from ATTRIBUTER")

    time_gdf = gpd.read_file(time_shp)
    log(f"Loaded {len(time_gdf)} polygons from TIME")

    # Extract date columns
    date_fields = ["EVENT_DATE", "DATE_START", "DATE_END"]
    extra_fields = ["CONFIDENCE", "PRECIP_MM", "CHG_SCORE"]
    available = [f for f in date_fields + extra_fields if f in time_gdf.columns]

    keep = [time_name_field] + available
    time_df = pd.DataFrame(time_gdf[[c for c in keep if c in time_gdf.columns]])

    if time_name_field != points_name_field:
        time_df = time_df.rename(columns={time_name_field: points_name_field})

    # Check for duplicates to decide join strategy
    dupe_time = time_df[points_name_field].duplicated().sum()
    dupe_pts = pts[points_name_field].duplicated().sum()

    if dupe_time or dupe_pts:
        log("Duplicate names detected -- using positional join (row order)")
        pts_reset = pts.reset_index(drop=True)
        time_reset = time_df.reset_index(drop=True)
        date_cols = [c for c in time_reset.columns if c != points_name_field]
        merged = pts_reset.copy()
        for col in date_cols:
            merged[col] = time_reset[col]
    else:
        log(f"Joining on '{points_name_field}'...")
        merged = pts.merge(time_df, on=points_name_field, how="left")
        if available:
            matched = merged[available[0]].notna().sum()
            log(f"  {matched} / {len(merged)} points matched to a date record")

    # change HAZ_TYPE value based on name in PT_TYPE
    LANDSLIDE_KEYWORDS = ["landslide", "land", "slide", "ls"]
    def classify_haz_type(row):
        name = str(row[points_name_field]).lower()
        if any(kw in name for kw in LANDSLIDE_KEYWORDS):
            return "Landslide-Generated Debris Flow"
        return "Channelized Sediment Flow"
    merged["HAZ_TYPE"] = merged.apply(classify_haz_type, axis=1)

    # Add lat/lon
    wgs = merged.to_crs("EPSG:4326")
    merged["LONGITUDE"] = wgs.geometry.x.round(6)
    merged["LATITUDE"] = wgs.geometry.y.round(6)

    os.makedirs(os.path.dirname(output_shp), exist_ok=True)
    merged.to_file(output_shp)
    log(f"Merged output saved to: {output_shp}")

    return output_shp
