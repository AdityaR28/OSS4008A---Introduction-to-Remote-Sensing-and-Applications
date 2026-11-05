# Data sources and access

The map notebook uses saved satellite display layers and metadata in
../results/satellite_data/. Large raw satellite scenes are accessed
through Google Earth Engine.

## Primary satellite collections

- [Sentinel-2 harmonized surface reflectance](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Sentinel-1 GRD radar imagery](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- [Cloud Score+ observation-quality data](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED)

## Event details

The saved catalogue, wildfire manifest and individual event.json files
record the relevant imagery, dates, processing information, source
references and limitations. Consult these files for event-specific
ancillary datasets and attribution.

## Reproducing the maps

Run code/Canadian_Disasters_Monitor_Reproducible.ipynb in Google Colab.
It retrieves the committed display inputs and verifies their checksums.

## Obtaining new satellite observations

Use the collection links above and the Python modules in
code/processing/. This separate workflow requires an Earth Engine
account, authentication, a configured project and event parameters.
