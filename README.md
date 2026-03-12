# Debris Flow Date Detection Tool

A Streamlit web app for detecting post-fire debris flow event dates from Sentinel-2 satellite imagery and CHIRPS precipitation data via Google Earth Engine.

See [docs/methodology.md](docs/methodology.md) for full methodology documentation.

---

## Running Locally

Running locally is recommended for large fires (>~10 polygons), as cloud deployments may time out during long GEE processing runs.

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or Miniconda
- A [Google Earth Engine](https://earthengine.google.com/) account with a registered cloud project

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/tanneroman01/Debris-Flow-Date-Detection-Tool.git
cd Debris-Flow-Date-Detection-Tool
```

**2. Create and activate the conda environment**
```bash
conda create -n debrisflow python=3.11
conda activate debrisflow
pip install -r requirements.txt
```

**3. Authenticate with Google Earth Engine**
```bash
earthengine authenticate
```
This opens a browser window. Sign in with the Google account linked to your GEE project and follow the prompts. You only need to do this once.

**4. Run the app**
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## Using the App

### Inputs

| Input | Description |
|---|---|
| **GEE Cloud Project ID** | Your GEE project ID (e.g. `my-project-123456`). Find it at [console.cloud.google.com](https://console.cloud.google.com) or run `earthengine project list`. |
| **GEE Credentials JSON** | Only required when using the hosted web app. Paste the contents of `~/.config/earthengine/credentials` (Windows: `C:\Users\<you>\.config\earthengine\credentials`). Leave blank when running locally. |
| **KML file** | Exported from Google Earth with your mapped debris flow polygons. Supported feature types: deposits, initiation zones, outlets, landslide scarps. Polygons are recommended — points and linestrings will be buffered to 50m. |
| **Fire boundary shapefile** | Upload all components (.shp, .shx, .dbf, .prj). Used to clip the road network for ROAD_REL attribution. |
| **Fire** | Select from the built-in Colorado fire database (MTBS fires) or enter custom fire metadata manually. |

### Output

A ZIP file containing a shapefile of centroid points with the following fields:

| Field | Description |
|---|---|
| `PT_TYPE` | Feature type from KML name |
| `FIRENAME` | Fire name |
| `FIRE_YEAR` | Fire year |
| `IGN_DATE` | Fire ignition date |
| `ROAD_REL` | Yes/No — feature within 100m of a road |
| `DEPO_AREA` | Deposit area in m² (Deposit features only) |
| `EVENT_DATE` | Detected debris flow date |
| `DATE_START` | Start of detection interval |
| `DATE_END` | End of detection interval |
| `CONFIDENCE` | High / Medium / Low |
| `PRECIP_MM` | Total precipitation (mm) in 30-day window before event |
| `CHG_SCORE` | Composite spectral change score |
| `LATITUDE` | Centroid latitude |
| `LONGITUDE` | Centroid longitude |

---

## Notes

- Processing time is approximately 20–40 minutes for fires with ~50 polygons, depending on GEE server load.
- Detection is limited to the active debris flow season (April–November) and skips snow-covered intervals (NDSI > 0.4).
- A post-fire buffer of ~9 months is applied before searching for events, to avoid detecting the fire itself.
- The tool currently includes fire boundary data for Colorado MTBS fires. For fires outside this database, use the manual fire entry option in the sidebar.
