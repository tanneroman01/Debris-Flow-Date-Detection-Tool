"""
Step 2: Build spatial attributes and centroid points.

Computes ROAD_REL (within 100m of road) and DEPO_AREA (sq m) for each
deposit polygon, then creates centroid points with CDOT template fields.
"""

import os
import json
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid


def to_utm(gdf):
    utm_crs = gdf.estimate_utm_crs()
    return gdf.to_crs(utm_crs), utm_crs


def compute_deposit_area(gdf, name_field="Name"):
    gdf_utm, _ = to_utm(gdf)
    areas = gdf_utm.geometry.area
    gdf["DEPO_AREA"] = None
    mask = gdf[name_field] == "Deposit"
    gdf.loc[mask, "DEPO_AREA"] = areas[mask].round(2)
    return gdf


def run(
    polygons_shp: str,
    fire_boundary_shp: str,
    roads_shp: str,
    template_shp: str,
    fire_defaults: dict,
    output_dir: str,
    name_field: str = "Name",
    log=print,
) -> tuple:
    """
    Run the attributer step.

    Args:
        polygons_shp: Path to polygons.shp from Step 1
        fire_boundary_shp: Path to fire boundary shapefile
        roads_shp: Path to OSM roads shapefile
        template_shp: Path to CDOT shapefile template
        fire_defaults: Dict with fire metadata (FIRENAME, FIRE_YEAR, IGN_DATE, etc.)
        output_dir: Where to write outputs
        name_field: Field name containing polygon names (default "Name")
        log: Logging function

    Returns:
        (polygons_roadsarea_path, points_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read polygons
    polys = gpd.read_file(polygons_shp)
    if polys.crs is None:
        polys = polys.set_crs("EPSG:4326")
    log(f"Loaded {len(polys)} polygons")

    # Buffer fire boundary by 2 miles, clip roads
    fire_bndy = gpd.read_file(fire_boundary_shp)
    if fire_bndy.crs is None:
        fire_bndy = fire_bndy.set_crs("EPSG:4326")

    fire_utm, utm_crs = to_utm(fire_bndy)
    fire_buffer_2mi = fire_utm.buffer(2 * 1609.34)
    fire_buffer_gdf = gpd.GeoDataFrame(geometry=fire_buffer_2mi, crs=utm_crs).to_crs(polys.crs)

    roads = gpd.read_file(roads_shp)
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    roads = roads.to_crs(polys.crs)
    clipped_roads = gpd.clip(roads, fire_buffer_gdf)
    log(f"Clipped {len(clipped_roads)} road segments within 2-mile fire buffer")

    # Buffer clipped roads by 100m
    clipped_roads_utm = clipped_roads.to_crs(utm_crs)
    roads_100m = clipped_roads_utm.buffer(100)
    roads_100m_gdf = gpd.GeoDataFrame(geometry=roads_100m, crs=utm_crs).to_crs(polys.crs)

    # Spatial join for ROAD_REL
    joined = gpd.sjoin(polys.copy(), roads_100m_gdf, how="left", predicate="intersects")
    near_road_indices = joined.dropna(subset=["index_right"]).index.unique()
    polys["ROAD_REL"] = "No"
    polys.loc[polys.index.isin(near_road_indices), "ROAD_REL"] = "Yes"
    log(f"ROAD_REL: {polys['ROAD_REL'].value_counts().to_dict()}")

    # Deposit area
    if "DEPO_AREA" not in polys.columns:
        polys["DEPO_AREA"] = None
    polys = compute_deposit_area(polys, name_field=name_field)

    # Repair geometry
    polys["geometry"] = polys["geometry"].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )

    # Save polygons with road/area attributes
    polys_out = os.path.join(output_dir, "polygons_roadsarea.shp")
    polys.to_file(polys_out)
    log(f"Saved polygons with attributes to {polys_out}")

    # Create centroid points with template schema
    template = gpd.read_file(template_shp)
    template_fields = [c for c in template.columns if c != "geometry"]

    centroids = polys.copy()
    centroids["geometry"] = polys.geometry.representative_point()

    pt_type_field = "PT_TYPE"
    points_data = {}
    points_data[pt_type_field] = centroids[name_field].values

    valid_default_keys = [k for k in fire_defaults.keys() if k in template_fields]
    for key in valid_default_keys:
        points_data[key] = [fire_defaults[key]] * len(centroids)

    for field in ("ROAD_REL", "DEPO_AREA"):
        if field in template_fields and field in centroids.columns:
            points_data[field] = centroids[field].values

    for field in template_fields:
        if field not in points_data:
            points_data[field] = [None] * len(centroids)

    points_gdf = gpd.GeoDataFrame(
        points_data, geometry=centroids["geometry"].values, crs=polys.crs
    )
    points_gdf = points_gdf.set_crs("EPSG:4326", allow_override=True)

    points_out = os.path.join(output_dir, "points.shp")
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = os.path.splitext(points_out)[0] + ext
        if os.path.exists(p):
            os.remove(p)

    points_gdf.to_file(points_out)
    log(f"Created {len(points_gdf)} centroid points at {points_out}")

    return polys_out, points_out
