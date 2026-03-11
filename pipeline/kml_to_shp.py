"""
Step 1: KML to Shapefile conversion.

Parses a KML file, extracts all geometry types (including MultiGeometry),
converts points and lines to 50m buffer polygons, and writes a single
polygons.shp output.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any
import geopandas as gpd
from shapely.geometry import shape

NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2",
}


def parse_coordinates(text: str) -> List[Tuple[float, float]]:
    coords = []
    if not text:
        return coords
    parts = text.strip().split()
    for p in parts:
        vals = p.split(",")
        if len(vals) >= 2:
            try:
                coords.append((float(vals[0]), float(vals[1])))
            except Exception:
                continue
    return coords


def close_ring_if_needed(ring):
    if not ring:
        return ring
    if ring[0] != ring[-1]:
        return ring + [ring[0]]
    return ring


def sanitize_properties(props):
    clean = {}
    for k, v in (props or {}).items():
        if v is None:
            clean[k] = ""
        else:
            try:
                clean[k] = str(v)
            except Exception:
                clean[k] = ""
    return clean


def extract_placemark_properties(pm_elem):
    props = {}
    name = pm_elem.find("kml:name", NS)
    if name is not None and name.text:
        props["Name"] = name.text

    desc = pm_elem.find("kml:description", NS)
    if desc is not None and desc.text:
        props["description"] = desc.text

    for data in pm_elem.findall(".//kml:ExtendedData//kml:Data", NS):
        key = data.get("name") or (
            data.find("kml:displayName", NS).text
            if data.find("kml:displayName", NS) is not None
            else None
        )
        val_elem = data.find("kml:value", NS)
        val = val_elem.text if val_elem is not None else None
        if key:
            props[str(key)] = val

    for sd in pm_elem.findall(".//kml:SchemaData//kml:SimpleData", NS):
        key = sd.get("name")
        val = sd.text
        if key:
            props[str(key)] = val

    return props


def extract_geometries_from_placemark(pm_elem):
    points, lines, polygons = [], [], []
    props = sanitize_properties(extract_placemark_properties(pm_elem))

    for p in pm_elem.findall(".//kml:Point", NS):
        coord_elem = p.find("kml:coordinates", NS)
        coords = parse_coordinates(coord_elem.text if coord_elem is not None else "")
        if coords:
            lon, lat = coords[0]
            points.append({"geometry": {"type": "Point", "coordinates": (lon, lat)}, "properties": props})

    for ls in pm_elem.findall(".//kml:LineString", NS):
        coord_elem = ls.find("kml:coordinates", NS)
        coords = parse_coordinates(coord_elem.text if coord_elem is not None else "")
        if coords and len(coords) >= 1:
            lines.append({"geometry": {"type": "LineString", "coordinates": coords}, "properties": props})

    for poly in pm_elem.findall(".//kml:Polygon", NS):
        outer_elem = poly.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", NS)
        if outer_elem is None:
            outer_elem = poly.find(".//kml:LinearRing/kml:coordinates", NS)
        outer_coords = parse_coordinates(outer_elem.text if outer_elem is not None else "")
        if not outer_coords:
            continue
        outer_coords = close_ring_if_needed(outer_coords)

        inner_coords_list = []
        for inner in poly.findall(".//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates", NS):
            ic = parse_coordinates(inner.text or "")
            if ic:
                inner_coords_list.append(close_ring_if_needed(ic))

        geo_coords = [outer_coords] + inner_coords_list
        polygons.append({"geometry": {"type": "Polygon", "coordinates": geo_coords}, "properties": props})

    return points, lines, polygons


def collect_all_geometries(kml_path: str):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    placemarks = root.findall(".//kml:Placemark", NS)

    all_points, all_lines, all_polys = [], [], []
    for pm in placemarks:
        pts, lns, polys = extract_geometries_from_placemark(pm)
        all_points.extend(pts)
        all_lines.extend(lns)
        all_polys.extend(polys)

    return all_points, all_lines, all_polys


def buffer_features_to_polygons(features, geom_type):
    if not features:
        return []

    geometries = [shape(f["geometry"]) for f in features]
    props_list = [f.get("properties", {}) for f in features]

    gdf = gpd.GeoDataFrame(props_list, geometry=geometries, crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)

    if geom_type == "LineString":
        gdf_utm["geometry"] = gdf_utm.geometry.interpolate(0.5, normalized=True)

    gdf_utm["geometry"] = gdf_utm.geometry.buffer(50)
    gdf_wgs84 = gdf_utm.to_crs("EPSG:4326")

    buffered = []
    prop_cols = [c for c in gdf_wgs84.columns if c != "geometry"]
    for _, row in gdf_wgs84.iterrows():
        buffered.append({
            "geometry": row.geometry.__geo_interface__,
            "properties": {col: row[col] for col in prop_cols},
        })
    return buffered


def write_shapefile(features, out_path):
    if not features:
        return
    geometries = [shape(f["geometry"]) for f in features]
    props_list = [f.get("properties", {}) for f in features]
    gdf = gpd.GeoDataFrame(props_list, geometry=geometries, crs="EPSG:4326")

    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = os.path.splitext(out_path)[0] + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    gdf.to_file(out_path, driver="ESRI Shapefile")


def run(kml_path: str, output_dir: str, log=print) -> str:
    """Convert KML to polygons.shp. Returns path to output shapefile."""
    os.makedirs(output_dir, exist_ok=True)

    pts, lns, polys = collect_all_geometries(kml_path)
    log(f"Parsed KML: {len(pts)} points, {len(lns)} lines, {len(polys)} polygons")

    all_polys = (
        polys
        + buffer_features_to_polygons(pts, "Point")
        + buffer_features_to_polygons(lns, "LineString")
    )

    out_path = os.path.join(output_dir, "polygons.shp")
    write_shapefile(all_polys, out_path)
    log(f"Wrote {len(all_polys)} polygon features to {out_path}")
    return out_path
