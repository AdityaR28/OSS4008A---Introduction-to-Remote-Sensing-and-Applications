# Data sources

## Statistics Canada — Open Database of Healthcare Facilities, v2.0

This is the main source for the current map. Version 2.0 was released in September 2026, and the app filters the dataset to records classified as hospitals. The latest Colab run returned 678 hospital reference records.

Dataset page: https://www150.statcan.gc.ca/n1/pub/13-26-0001/132600012020001-eng.htm

Data archive used by the notebook: https://www150.statcan.gc.ca/n1/pub/13-26-0001/2020001/zip/odhf_v2_geojson.zip

The original records may use point or polygon geometries. Neither is necessarily a verified outline of the hospital roof. The dataset is a location reference and is not a complete inventory of every hospital or a source of image-detector ground truth. Attribute and position quality still need checking.

## Place search and map tiles

The app uses the Open-Meteo Geocoding API to look up a Canadian city or town entered by the user: https://open-meteo.com/en/docs/geocoding-api

Map tiles come from CartoDB/OpenStreetMap, with attribution displayed in the interface. Earlier notebooks also explored OpenStreetMap hospital features for Halifax.

## Satellite and aerial imagery

Google Earth Engine is used in the separate imagery notebooks. The web app does not yet download or analyze satellite pixels. High-resolution imagery is still needed for actual model training, and its licence should be checked before including images in GitHub or a published training dataset.

The ODHF database and the imagery source have different roles: the former helps find known sites, while the latter provides the pixels a computer-vision model would analyze.
