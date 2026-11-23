# History of Wildfires in Canada
Project #23 | OSS4008A - Introduction to Remote Sensing and Applications

## Project description
An interactive map of Canada showing burned area over time, with a time slider
(2001-2023) and, later, an AI model that detects burned areas in satellite images.
It will become one button in the shared course dashboard.

## Folder contents
- code/ : Colab notebook (Earth Engine map, yearly chart, button + slider UI)
- data/ : data notes (sources and download details)
- results/ : yearly burned-area CSV and chart

## Data
MODIS MCD64A1 burned area (500 m) via Google Earth Engine. Canada first.
Planned: Sentinel-2 / Landsat for burn-scar annotation, NRCan fire database for validation.

## Progress log
### Nov 9-12, 2026
- Set up GitHub and Colab push workflow
- Connected Google Earth Engine
- Built 2023 burned-area map of Canada
- Built yearly burned-area chart (2001-2023)
- Built a Wildfires button and year slider (UI prototype)
- Next: annotation or existing burn-scar dataset, model training, connect to dashboard
