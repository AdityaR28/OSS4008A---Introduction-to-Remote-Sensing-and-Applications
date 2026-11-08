# Hospital Infrastructure in Canada

**OSS 4008 — Remote Sensing | Hamza Arab**  
**Updated:** November 8, 2026

## Project overview

For my OSS 4008 project, I'm building a tool to explore hospital infrastructure across Canada using geographic data and overhead imagery. The intended final system will let someone search for a location, select a study area, run a trained hospital-detection model on suitable imagery, and view the results on an interactive map.

I've started with the map and data workflow. The web interface is already running in Google Colab: I can search for Canadian cities and towns, display known hospital locations from Statistics Canada, change the study radius, and use a simple map assistant to navigate between places. The hospital markers are **existing database records, not results from an image-detection model**.

## What works today

- Google Earth Engine authentication and an initial satellite-imagery notebook run in Colab.
- Statistics Canada's Open Database of Healthcare Facilities (ODHF v2.0) can be downloaded, filtered to hospitals, and displayed on the map. The successful notebook run returned **678 hospital reference records**; that is a source-data count, not a detection result.
- The browser interface uses a city/town search, map layers, a study-radius control and a panel for short location commands.
- The map supports drawing study-area shapes, although those drawings are not connected to imagery analysis yet.
- The web interface can display a GeoJSON prediction file generated separately, once real model results exist.

The latest compact, single-screen layout was updated on November 8. Earlier Gradio versions have been demonstrated in Colab; the most recent layout still needs a visual check in the browser.

## How to run the current demo

Open **`01_Notebooks/04_canada_web_demo.ipynb`** in Google Colab and choose **Runtime → Run all**. It installs the Python packages, prepares the Statistics Canada hospital reference CSV, and launches the web app. Open the temporary `gradio.live` link shown in the final output.

**Notebook 04 is self-contained.** It includes copies of the data-preparation and app source code so it works without mounting Google Drive. The separate `.py` files in `02_App` are maintained for easier editing, reuse and testing. If the app code changes, the copy embedded in Notebook 04 must also be updated. Notebooks 01–03 do not have to run before Notebook 04.

## Folder guide

| Folder | Purpose |
| --- | --- |
| `00_Documentation` | Current progress, dataset notes, demo notes and meeting update. |
| `01_Notebooks` | Colab notebooks for setup, mapping and the planned model pipeline. |
| `02_App` | Reusable Python code, dependencies and basic tests for the web interface. |
| `03_Data` | Small dataset references and GeoJSON formats; later, annotation manifests. |
| `04_Results` | A place for real model evaluation results, maps and figures once produced. |

### Notebook guide

| Notebook | Purpose | Current stage |
| --- | --- | --- |
| `01_earth_engine_setup.ipynb` | Test Earth Engine and preview imagery. | Tested in Colab. |
| `02_halifax_hospital_reference.ipynb` | Explore Halifax hospital references and imagery. | Tested in Colab. |
| `03_hospital_annotation_candidates.ipynb` | Explore candidate hospital sites and building polygons. | Preparation work; candidates require verification. |
| `04_canada_web_demo.ipynb` | Run the Canada-wide web map. | Working demo; latest visual revision needs a check. |
| `05_labelme_to_yolo.ipynb` | Convert manually checked LabelMe bounding boxes into training labels. | Prepared; no verified image/label dataset yet. |
| `06_train_evaluate_export.ipynb` | Train, evaluate and export a detection model. | Prepared scaffold; no model trained or measured accuracy yet. |

The notebook numbers describe the workflow. **Notebook 04 does not call all the other notebooks.** Notebooks 05 and 06 are for later work, once suitable imagery and verified annotations are available.

### Python files

- `app.py`: main map, search, interface and location-command functions.
- `prepare_odhf.py`: download and filter the national healthcare database into a hospital reference CSV.
- `test_app.py`: basic checks of application functions, not a substitute for real model testing.
- `requirements.txt`: packages needed by the app.
- `regions.py`: early Halifax/Ottawa region helper from setup; not required by the current web demo.

## What isn't finished

The central remote-sensing component is still underway. The current interface **does not inspect imagery or find hospitals with AI**. Sentinel-2 imagery is useful for the initial workflow but its 10 m visible-band pixels are generally too coarse for reliable identification of individual hospital buildings. The next substantial task is to obtain suitable licensed high-resolution imagery, verify hospital building labels, and train and evaluate an image detector with independent test data. Predictions then need to be connected to the selected map area.

A hospital's purpose also cannot always be determined from its roof appearance alone; results will need checking against known sites and similar-looking nonhospital buildings.

## Course delivery

Development and demonstrations use Google Colab, as required by the course repository. Final work also needs to be added to `students/hamza_arab/` in the class GitHub repository, with the relevant documentation and deliverables. The final shared frontend framework and any specific imagery/annotation restrictions should be confirmed with the TA.

For the **November 9 progress meeting**, the current deliverable is a working Canadian hospital-reference map and an established data-preparation workflow—not a trained hospital detector. See `00_Documentation/PROGRESS_UPDATE_NOV09.md` for the short meeting summary.

## Data and attribution

Hospital reference data: Statistics Canada, *Open Database of Healthcare Facilities*, version 2.0 (2026). Place search: Open-Meteo Geocoding API. Map tiles: OpenStreetMap/CartoDB with attribution in the interface. Earth Engine is used separately for imagery exploration. See `00_Documentation/DATA_SOURCES.md` for source and usage notes.

Do not upload credentials, oversized imagery, or model weights to the class GitHub repository. Record imagery licences before publishing training images.
