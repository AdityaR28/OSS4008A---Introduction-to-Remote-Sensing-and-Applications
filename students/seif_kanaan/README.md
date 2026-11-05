# Canadian Disasters Monitor

**Student:** Seif Kanaan  
**Course:** OSS4008A — Introduction to Remote Sensing and Applications

## Project objective

Investigate satellite-observed changes associated with selected Canadian
wildfires, floods, earthquakes and tornadoes, and display the results
through interactive maps.

## Run in Google Colab

[Open the reproducible notebook](https://colab.research.google.com/github/AdityaR28/OSS4008A---Introduction-to-Remote-Sensing-and-Applications/blob/main/students/seif_kanaan/code/Canadian_Disasters_Monitor_Reproducible.ipynb)

1. Open the notebook in Colab and connect to a runtime.
2. Select Runtime → Run all.
3. Use the event dropdown and map layer controls to explore the results.

The notebook downloads the repository, installs the required dependencies,
verifies the saved satellite layers and generates four interactive maps.

## Data and methods

- Sentinel-2 optical imagery for wildfire and vegetation-change analysis.
- Sentinel-1 radar imagery for flood and earthquake-related landscape change.
- Before-and-after comparisons, quality masks and exploratory change indicators.
- Python, Folium and ipywidgets for map generation and interaction.

| Case study | Verified image layers |
|---|---:|
| McDougall Creek wildfire, 2023 | 6 |
| Lower Fraser flood, 2021 | 7 |
| Hubbard Glacier earthquake context, 2025 | 8 |
| Law Lake tornado context, 2025 | 7 |
| Total | 28 |

Results are exploratory observations. They do not establish confirmed
disaster damage or independently validated hazard boundaries.

## Reproducibility scope

The main notebook verifies previously exported satellite layers and
rebuilds their interactive maps.

A new satellite acquisition and numerical processing run requires the
separate Earth Engine workflow, authentication and a configured project.
Processing modules are retained in code/processing/.

## Project organization

- code/: Python processing modules and Colab notebooks.
- data/: source-data links and access instructions.
- results/satellite_data/: saved display layers and event metadata.
- results/maps/: generated interactive HTML maps.

## Weekly progress log

### 2026-11-04

- Reviewed the prototype against the course requirements.
- Established the required student folder structure.
- Imported Python processing modules and satellite display results.
- Installed dependencies and checked module imports.
- Built and tested interactive maps for all four events.
- Uploaded the project files to GitHub.

### 2026-11-05

- Tested the reproducible notebook in a fresh Colab runtime.
- Successfully verified four events and 28 image layers.
- Confirmed that map generation and the event dropdown work.
- Documented the workflow, data sources and reproducibility scope.
- Next: expand the methods discussion and assess the interpretation
  and limitations of each case study.
