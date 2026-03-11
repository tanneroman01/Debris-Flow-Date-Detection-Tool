# Debris Flow Date Detection Tool Methodology

## Overview 
This document outlines the backend methodology involved in the production of the attribute-rich shapefile output of the debris flow date detection tool. As a disclaimer, this is a **naive** implementation designed to help constrain event dates to avoid manually looking at >1000 days of imagery per debris flow event. This is in no way optimized for processing speed, and not designed to handle extremely large data. Many changes could be made to optimize feature engineering and improve scientific-basis grounding. 

## 1. Input Processing 

The tool accepts a kml file as the input containing polygons of mapped debris flow features (e.g. deposit, initiation, landslide scarp, etc.) exported from google earth. The first step in the pipeline is to convert the kml to a shp file that can be handled by the other scripts that use shapely and geopandas (arcpy and pyQGIS libraries were avoided to needing multiple environments). This conversion is handled by the kml_to_shp.py script where the raw text is read and converted to coordinate tuples, polygon rings are closed if needed (last point appended to first point), and linestrings or placemarks are buffered to create polygons (placemark or midpoint of linestring buffered out to 50m). However, using linestrings and placemarks for mapping is not recommended to avoid data leakage of the non-feature background into the composit change score. The final output is an ESRI shapefile with all components.

## 2. Spatial Attribution

The tool then adds feature values for identifying fields ( e.g. COUNTRY, STATE, FIRENAME, IG_DATE) that are inputed by the user, or pulled from a reference json file containing the mtbs metada of all the colorado fire. Two (potentially useful) new features are calculated: ROAD_REL and DEPO_AREA. Deposit areas are calculated as the geodesic area in square meters. The ROAD_REL attribute is populated with boolean (Yes/No) values based on wether the debris flow threatens a road, based on intersect of a 100m buffered road shapefile. The road data is downloaded from OpenStreetMap, and contains **all** roads (forest, highways, county roads, private, etc.). Issues that need addressed in this step are: depo area is calculated for all features irregardless of PT_TYPE name (deposit area of initiations and outlets is irrelevant) and depo areas are still calculated for polygons that are potentially buffered from points or linestrings (these would all be the same value and contain no useful information). I plan on adding some keyword-to-type mappings that prevent these problems. 

## 3. Date Detection

The bulk of the compute time in the process is consumed here. Most of the work is happening server side with Google Earth Engine, but it still takes ~30 minutes to process a large fire with >~50 polygons. This step pulls Sentinel-2 satellite imagery from goog

### 3.1 Spectral Indices


### 3.2 Composite Change Score

### 3.3 Baseline Reference 

### 3.4 Precipitation Validation

### 3.5 Detection Parameters

## 4. Output Merging

## 5. Next Steps for Data Pre-Processing

## References