"""
Debris Flow Detection Tool -- Streamlit Web App

Provides a user-friendly interface to the full debris flow mapping pipeline:
KML -> attribute addition -> GEE date detection -> merged output.
"""

import os
import sys
import json
import tempfile
import zipfile
from datetime import date
from io import BytesIO

import streamlit as st

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import orchestrator

# page configuration
st.set_page_config(
    page_title="Debris Flow Date Detection Tool",
    page_icon="🏔️",
    layout="wide",
)

# paths to bundled data (defaults, template, roads)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
FIRE_DEFAULTS_PATH = os.path.join(DATA_DIR, "fire_defaults.json")
TEMPLATE_SHP = os.path.join(DATA_DIR, "template", "CDOT_ShapefileTemplate.shp")
ROADS_SHP = os.path.join(DATA_DIR, "roads", "gis_osm_roads_free_1.shp")

# helper functions for fetching defaults, saving uploads, and creating download zips
def load_fire_defaults():
    if os.path.exists(FIRE_DEFAULTS_PATH):
        with open(FIRE_DEFAULTS_PATH, "r") as f:
            return json.load(f)
    return None


def save_uploaded_file(uploaded_file, dest_dir, filename=None):
    """Save an uploaded file to disk and return the path."""
    fname = filename or uploaded_file.name
    path = os.path.join(dest_dir, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def save_uploaded_shapefile(uploaded_files, dest_dir, base_name):
    """Save all components of an uploaded shapefile (.shp, .shx, .dbf, .prj, etc.)."""
    paths = []
    for uf in uploaded_files:
        path = os.path.join(dest_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        paths.append(path)
    shp_path = os.path.join(dest_dir, base_name + ".shp")
    if os.path.exists(shp_path):
        return shp_path
    for p in paths:
        if p.endswith(".shp"):
            return p
    return None


def create_download_zip(output_dir):
    """Create a zip of all shapefile components in the output directory."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    buf.seek(0)
    return buf


# load the fire defaults database to populate the fire selection dropdown
fire_db = load_fire_defaults()
fire_keys = [k for k in fire_db.keys() if not k.startswith("_")] if fire_db else []

# ── Header ──
st.title("Debris Flow Date Detection Tool")
st.markdown(
    "Post-fire debris flow inventory pipeline. Upload your KML with mapped polygons for debris flow/landslide deposits, outlets, initiation points, or other features. Also upload the fire boundary shapefile."
    "Provide fire metadata (select from dropdown) and provide your Google Earth Enginge Project ID, and get back a fully attributed shapefile with detected event dates."
)

# ── Sidebar: GEE project ID,  ──
with st.sidebar:
    st.header("Set Up")

    gee_project = st.text_input(
        "GEE Cloud Project ID",
        value="",
        help="Your Google Earth Engine cloud project ID (e.g., 'my-gee-project-12345')",
    )

    st.divider()
    st.subheader("Fire Selection")

    use_existing = st.checkbox("Use existing fire from database", value=bool(fire_keys))

    if use_existing and fire_keys:
        fire_key = st.selectbox("Select fire", fire_keys)
        fire_entry = fire_db[fire_key]
        st.info(
            f"**{fire_entry['FIRENAME']}** ({fire_entry['FIRE_YEAR']})\n\n"
            f"Ignition: {fire_entry['IGN_DATE']}"
        )
        fire_name = fire_entry["FIRENAME"]
        fire_year = fire_entry["FIRE_YEAR"]
        ign_date_str = fire_entry["IGN_DATE"]
    else:
        fire_key = st.text_input(
            "Fire key",
            help="Unique identifier, e.g. MYFIRE2023 (used for folder names)",
        )
        fire_name = st.text_input("Fire name", help="e.g. Cameron Peak")
        fire_year = st.text_input("Fire year", value=str(date.today().year))
        ign_date_input = st.date_input("Ignition date")
        ign_date_str = ign_date_input.strftime("%m/%d/%Y") if ign_date_input else ""

    obs_date = st.date_input("Date Mapped", value=date.today())
    obs_date_str = obs_date.strftime("%Y-%m-%d")

# ── Main area: File uploads ──
st.header("Input Files")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Debris-Flow Feature KML")
    kml_file = st.file_uploader(
        "Upload KML with mapped polygons",
        type=["kml"],
        help="Export from Google Earth with your digitized debris flow polygons",
    )

with col2:
    st.subheader("2. Fire Boundary")
    fire_boundary_files = st.file_uploader(
        "Upload fire boundary shapefile",
        type=["shp", "shx", "dbf", "prj", "cpg"],
        accept_multiple_files=True,
        help="Upload all shapefile components (.shp, .shx, .dbf, .prj)",
    )

# Raise warnings if required inputs are missing or if the shapefile upload doesn't include a .shp file
ready = True
issues = []

if not gee_project:
    issues.append("Enter your GEE Cloud Project ID in the sidebar")
    ready = False
if not kml_file:
    issues.append("Upload a KML file with deposit polygons")
    ready = False
if not fire_boundary_files:
    issues.append("Upload fire boundary shapefile components")
    ready = False
if not fire_key:
    issues.append("Select or enter a fire key")
    ready = False
if not ign_date_str:
    issues.append("Provide an ignition date")
    ready = False

has_shp = any(f.name.endswith(".shp") for f in fire_boundary_files) if fire_boundary_files else False
if fire_boundary_files and not has_shp:
    issues.append("Fire boundary upload must include a .shp file")
    ready = False

if not os.path.exists(ROADS_SHP):
    issues.append(f"Roads shapefile not found at: {ROADS_SHP}")
    ready = False
if not os.path.exists(TEMPLATE_SHP):
    issues.append(f"CDOT template not found at: {TEMPLATE_SHP}")
    ready = False

if issues:
    for issue in issues:
        st.warning(issue)

# A button to run the tool, disabled if all required inputs aren't there.  
st.divider()

if st.button("Run Pipeline", type="primary", disabled=not ready, use_container_width=True):
    # Create temp working directory
    work_dir = tempfile.mkdtemp(prefix="dbflow_")
    upload_dir = os.path.join(work_dir, "uploads")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(upload_dir)
    os.makedirs(output_dir)

    # Save uploaded files
    kml_path = save_uploaded_file(kml_file, upload_dir)
    fire_bndy_path = save_uploaded_shapefile(fire_boundary_files, upload_dir, "fire_boundary")

    if fire_bndy_path is None:
        st.error("Could not find .shp file in fire boundary upload")
        st.stop()

    # If fire key not in database, create a temporary fire_defaults with user inputs
    if use_existing and fire_db:
        fd_path = FIRE_DEFAULTS_PATH
    else:
        temp_db = {
            "_constants": fire_db.get("_constants", {}) if fire_db else {
                "OBS_USER": "",
                "COUNTRY": "United States",
                "STATE": "",
                "HAZ_TYPE": "Channelized Sediment Flow",
                "ID_METHOD": "Satellite Imagery",
                "FIRE_SRC": "MTBS",
            },
            fire_key: {
                "FIRENAME": fire_name,
                "FIRE_YEAR": fire_year,
                "IGN_DATE": ign_date_str,
            },
        }
        fd_path = os.path.join(upload_dir, "fire_defaults.json")
        with open(fd_path, "w") as f:
            json.dump(temp_db, f, indent=2)

    # Progress display
    progress_bar = st.progress(0, text="Starting process...")
    log_container = st.container()
    log_lines = []

    def ui_log(msg):
        log_lines.append(msg)
        with log_container:
            st.code("\n".join(log_lines[-30:]), language="text")

    step_weights = {1: 0.05, 2: 0.15, 3: 0.70, 4: 0.10}
    step_starts = {1: 0.0, 2: 0.05, 3: 0.20, 4: 0.90}

    def progress_callback(step, name, pct=None):
        base = step_starts.get(step, 0)
        weight = step_weights.get(step, 0.1)
        if pct is not None:
            total_pct = base + weight * pct
        else:
            total_pct = base
        progress_bar.progress(min(total_pct, 1.0), text=f"Step {step}: {name}")

    try:
        merged_path = orchestrator.run_full_pipeline(
            kml_path=kml_path,
            fire_boundary_path=fire_bndy_path,
            roads_shp=ROADS_SHP,
            template_shp=TEMPLATE_SHP,
            fire_defaults_path=fd_path,
            fire_key=fire_key,
            obs_date=obs_date_str,
            gee_project=gee_project,
            output_dir=output_dir,
            log=ui_log,
            progress_callback=progress_callback,
        )

        progress_bar.progress(1.0, text="Process complete!")
        st.success("Process finished successfully!")

        # Create downloadable zip
        zip_buf = create_download_zip(output_dir)
        st.download_button(
            label="Download Results (ZIP)",
            data=zip_buf,
            file_name=f"{fire_key}_debris_flow_results.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

        # Preview results
        import geopandas as _gpd

        if os.path.exists(merged_path):
            result_gdf = _gpd.read_file(merged_path)
            st.subheader("Results Preview")

            preview_cols = [
                c for c in result_gdf.columns if c != "geometry"
            ]
            st.dataframe(result_gdf[preview_cols], use_container_width=True)

            detected = result_gdf["EVENT_DATE"].notna().sum() if "EVENT_DATE" in result_gdf.columns else 0
            total = len(result_gdf)
            st.metric("Detection Rate", f"{detected} / {total} polygons")

    except Exception as e:
        progress_bar.progress(0, text="Error")
        st.error(f"Pipeline failed: {e}")
        st.exception(e)


# ── Footer ──
st.divider()
st.caption(
    "Debris Flow Detection Tool | CDOT Project | "
    "Powered by Google Earth Engine, Sentinel-2, CHIRPS, Streamlit, OpenStreetMap, and more. Developed by Tanner Oman"
)
