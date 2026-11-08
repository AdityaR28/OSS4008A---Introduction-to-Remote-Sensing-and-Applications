# Hospital Infrastructure in Canada

**Hamza Arab — OSS 4008 \| November 2026**

This project is about identifying hospital infrastructure in overhead imagery and showing the results on an interactive map of Canada. I have a working geographic reference viewer and the initial remote-sensing data workflow. The image based detector is the next major stage and has not been trained yet.

The current map can search Canadian cities and towns, display known hospitals from Statistics Canada's ODHF v2.0 dataset, and adjust the study radius. It also includes basic map-navigation commands. These hospital markers are published location records, **not predictions from a trained model**.

## Run the map

Open [`01_Notebooks/04_canada_web_demo.ipynb`](https://drive.google.com/file/d/10Sv_k6-VXAP0gSSI7NIlcDM8N-23stDv/view) in **Google Colab** and run all cells. It prepares the reference data and launches a temporary browser link. Notebook 04 runs on its own; you don't need to run notebooks 01–03 first.

## Where to find things

- **`00_Documentation`** — detailed README, progress log, November 9 update, dataset notes and meeting/demo notes.  
- **`01_Notebooks`** — Earth Engine, reference data, map, annotation and training notebooks.  
- **`02_App`** — reusable Python files for the current web interface.  
- **`03_Data`** — geographic data formats and future annotation records.  
- **`04_Results`** — reserved for actual model evaluations and result figures.

## Current status

The first four notebooks cover setup, exploratory reference data, candidate building geometries and the map. The annotation-conversion and detector-training notebooks are prepared but still need licensed high-resolution images and verified labels. There are no valid detection accuracy results yet.

For details, read `00_Documentation/README.md`.&nbsp;
