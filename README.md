# Debris Flow Date Detection Tool (SCAR v2)

A Streamlit web app for detecting post-fire debris flow event dates from Sentinel-2 satellite imagery via Google Earth Engine.

v2 replaces the rule-based change detector with a **HistGradientBoosting classifier** applied to per-scene Sentinel-2 index time series (NDVI, NDRE, NBR, NDSI, RECI, B04, composite score). Each scene is scored with deseasonalized anomaly features (2-harmonic fit residuals, local level-shift statistics, day-of-year encoding); the scene with the highest event probability is the detected date. Working at the native 3-5 day revisit replaces the v1 30-day composite windows, so `DATE_START`/`DATE_END` narrow to the gap between consecutive clear scenes. The model is trained on known SW-USA debris flow event dates (see `tools/train_model.py`); a pre-trained model ships at `data/model/hgb_date_model.joblib`.

**Sentinel-2 only**: fires ignited before mid-2015 are not supported — use the v1 app with the Landsat backend for those.

See [docs/methodology.md](docs/methodology.md) for the v1 process; steps 1, 2 and 4 are unchanged in v2.

## Retraining the model

```bash
python tools/train_model.py --csv <timeseries_all_polygons.csv>
```

The training CSV is produced by `Spectral_Separability_Analysis/pull_timeseries.py` from known-date polygons: one row per polygon-scene with all indices plus the known event window (`Before_Scn`/`After_Scn`). The script prints out-of-fold hit rates (GroupKFold grouped by event) before fitting the final model, so you can compare against the current model's `oof_metrics` (stored inside the artifact).

---

## Running Locally

Running locally is recommended for large fires (>~10 polygons), as the [Streamlit cloud deployment of this tool](https://debris-flow-date-detection-tool-sjmnanym4tw2rvc7hh23s7.streamlit.app) times out during long GEE runs (free tier issues).

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
This opens a browser window. Sign in with the Google account linked to your GEE project and follow the prompts. You only need to do this once, in subsequent runs the tool references the .config file.

**4. Run the app**
```bash
streamlit run app.py
```
The app will open in your browser at this port: `http://localhost:8501`.

---

## Using the App

### Inputs

| Input | Description |
|---|---|
| **GEE Cloud Project ID** | Your GEE project ID (e.g. `tannersproject_123456`). Find it at [console.cloud.google.com](https://console.cloud.google.com) or run `earthengine project list`. |
| **GEE Credentials JSON** | Only required when using the hosted web app. Paste the contents of `~/.config/earthengine/credentials` (Windows: `C:\Users\<you>\.config\earthengine\credentials`). Leave blank when running locally. |
| **KML file** | Exported from Google Earth with your mapped debris flow polygons. Polygons are recommended — points and linestrings will be buffered to 50m. |
| **Fire boundary shapefile** | Upload all components (.shp, .shx, .dbf, .prj). Used to clip the road network for ROAD_REL attribution. |
| **Fire** | Select from the built-in Colorado fire database (MTBS fires) or enter custom fire metadata manually. Ignition date must be after mid-2015 (Sentinel-2 era). |

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
| `EVENT_DATE` | Detected debris flow date (first scene showing the anomaly) |
| `DATE_START` | Last clear scene before the event |
| `DATE_END` | First scene showing the anomaly (same as EVENT_DATE) |
| `CONFIDENCE` | High / Medium / Low (binned model probability) |
| `CHG_SCORE` | Model event probability (0-1) |
| `LATITUDE` | Centroid latitude |
| `LONGITUDE` | Centroid longitude |

---

## Notes

- Scenes with more than 20% cloud/shadow cover over the polygon (Sentinel-2 SCL) or polygon-mean NDSI > 0.6 (snow) are dropped before scoring.
- A post-fire buffer of ~9 months (adjustable in the sidebar) is applied before searching for events, to keep the burn itself out of the candidate scenes. Reduce it for fires with first-monsoon debris flows.
- Polygons with fewer than 10 usable scenes are reported as undetected.
