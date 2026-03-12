# Debris Flow Date Detection Tool Methodology

## Overview 
This document outlines the backend methodology involved in the production of the attribute-rich shapefile output of the debris flow date detection tool. As a disclaimer, this is a **naive** implementation designed to help constrain event dates to avoid manually looking at >1000 days of imagery per debris flow event. This is in no way optimized for processing speed, and not designed to handle extremely large data. Many changes could be made to optimize feature engineering and improve emperical grounding.
## 1. Input Processing 

The tool accepts a kml file as the input containing polygons of mapped debris flow features (e.g. deposit, initiation, landslide scarp, etc.) exported from google earth. The first step in the pipeline is to convert the kml to a shp file that can be handled by the other scripts that use shapely and geopandas (arcpy and pyQGIS libraries were avoided to needing multiple environments). This conversion is handled by the kml_to_shp.py script where the raw text is read and converted to coordinate tuples, polygon rings are closed if needed (last point appended to first point), and linestrings or placemarks are buffered to create polygons (placemark or midpoint of linestring buffered out to 50m). However, using linestrings and placemarks for mapping is not recommended to avoid data leakage of the non-feature background into the composit change score. The final output is an ESRI shapefile with all components.

## 2. Spatial Attribution

The tool then adds feature values for identifying fields ( e.g. COUNTRY, STATE, FIRENAME, IG_DATE) that are inputed by the user, or pulled from a reference json file containing the mtbs metada of all the colorado fire. Two (potentially useful) new features are calculated: ROAD_REL and DEPO_AREA. Deposit areas are calculated as the geodesic area in square meters. The ROAD_REL attribute is populated with boolean (Yes/No) values based on wether the debris flow threatens a road, based on intersect of a 100m buffered road shapefile. The road data is downloaded from OpenStreetMap, and contains **all** roads (forest, highways, county roads, private, etc.). Issues that need addressed in this step are: depo area is calculated for all features irregardless of PT_TYPE name (deposit area of initiations and outlets is irrelevant) and depo areas are still calculated for polygons that are potentially buffered from points or linestrings (these would all be the same value and contain no useful information). I plan on adding some keyword-to-type mappings that prevent these problems. 

## 3. Date Detection

The bulk of the compute time in the process is consumed here. Most of the work is happening server side with Google Earth Engine, but it still takes ~30 minutes to process a large fire with >~50 polygons. This step pulls Sentinel-2 satellite imagery from Google Earth Engine, creates a median composite image for each polygon and interval with cloud masking, and calculates spectral indices for each median composite image, from which "events" are detected. Improvements in the temporal resolution of this process could be made. The revisit time for sentinel is 3-5 days, making it necessary to produce median composites of many days (30 in this case) if using cloud masking. This means that detected "events" occurred sometime within a 30-day interval. One potential solution for next iterations of this tool would be to use Planet's Dove Constellation, which has a daily revisit time, although fewer bands are available and costs could be high. Another solution may be to pull high-resolution gridded precip data (e.g. NEXRAD, Prism) within your detected interval and assume the highest magnitude event in a given interval produced the debris flow. 

### 3.1 Spectral Indices

Three spectral indices are used to calculate a composite score per-interval per-polygon. If the mean NDSI of a composite in an interval is greater than 0.4, the interval is skipped. Additionally, intervals from Dec-March are excluded from analysis as it is assumed debris flows aren't occuring during snow covered months. 

1. NBR - Normalized Burn Ratio 

    $ \dfrac{NIR-SWIR}{NIR + SWIR}$
2. NDVI - Normalized Difference Vegetation Index 

    $ \dfrac{NIR-Red}{NIR + Red} $
3. Red- Raw 

3. NDSI - Normalized Difference Snow Index (masking only)

    $ \dfrac{Green - SWIR}{Green + SWIR} $


### 3.2 Composite Change Score
The composite change score is calculated per-polygon per-interval after NBR, NVDI, and Red are rescaled to [0,1] by $Index = (value + 1)/2 $ and in the case of red $Red = (value /3000) $ after which np.clip is applied. The composite score takes the form

$ Composite = 0.35(NBR) + 0.45(NVDI) + 0.2(Red)$

A time series of first differences of the composite score is then constructed to represent change over time. A convolution with a uniform kernel could be helpful to smooth erraneous spikes, but I did't think the current 30-day interval width warranted this. 

### 3.3 Baseline Reference 
A baseline with which to compare the changes in the composite score for is also constructed per-interval per-polygon, making out detection self calibrating per polygon. A "donut" is constructed around the feature with an inner buffer that excluded the feature itself and an out buffer of 500m that captures nearby terrain. The "donut" feature is passed to the same functions as the polygon features, and a time series of composite scores is produced. From there, we calculate the standard deviation of first differences, and set a "detection threshold": 

$ Threshold = RefStdev * 0.75 $

The first interval where the composite score exceeds the detection threshold is marked as the "event", which is then passed to the precip validation functions.

### 3.4 Precipitation Validation
The purpose here is to reject spectral change events that lack a rainfall trigger. The CHIRPS dataset is queried for precip values for each polygon and sums all daily precip values for a 30-day window preceding the candidate event date. If total precip over the preceding 30 days is below 10mm, the event is discarded. 

### 3.5 Detection Parameters

I performed 10 runs on a set of polygons with known dates from the Cameron Peak and Grizzly Creek fires (from the USGS data release), each with a different combinations of parameters. The current list of parameters in use represents the best run out of these ten. However, this was an informal tuning on a small test set with no hold-out validation. Building a larger test set, defining formal evaluation metrics (e.g. F1), and performing a grid search would likely improve this tool markedly. 

## 4. Output Merging

Finally the outputs of the spatial attribution script and the time detection script are merged into a shapefile output of points with feature fields populated.

## 5. Next Steps for Data Pre-Processing

The gold-standard for optimizing this data collection process would be to produce an algorithm that accuratetly **locates** candidate debris flow events from imagery **and** detects dates for said events. This however seems difficult given the ambiguity of spectral data, lack of labeled inventories, and regional bias preventing good generalization. Next steps for improving this tool, or producing better similar tools, may look like using imagery products with higher satellite revisit times, higher resolution gridded precip products, and incorporating more indicies, etc. As a final output, a more feature rich dataset that includes delineated basins and precipitation data (both as a feature and to further refine date selection) will also be helpful.
