# Project 8 – Where are the airports in Canada?

**Student:** Nifemi Ogunmeru | **Course:** OSS4008A – Introduction to Remote Sensing and Applications (Carleton University)

An interactive map of every active airport, heliport and seaplane base in Canada (2,477 facilities), built in Python on Google Colab, with an **AI plane detector**. A Streamlit dashboard (notebook 02) follows the course's UI draft: AI assistant on the left, interactive map on the right, project buttons at the bottom. **Draw a box around an Ontario airport and a Faster R-CNN model, fine-tuned on 173 hand-labelled planes, finds the planes in open 20 cm aerial imagery and draws them on the map.**

## Objectives
1. Interactive map: airports by type, click for details, legend; then filters, search and summary statistics.
2. AI component: detect aircraft on high-resolution imagery at selected Canadian airports, and measure how image resolution affects detection.
3. Package the map so it plugs into the shared dashboard.

## Data
| Dataset | Used for | Source | Licence |
|---|---|---|---|
| OurAirports `airports.csv` | Airport locations, type, codes, province | https://ourairports.com/data/ | Public domain (per OurAirports) |
| Ontario Imagery Web Map Service (Land Information Ontario) | Aerial imagery (~20 cm) where planes are visible | https://ws.lioservices.lrc.gov.on.ca/arcgis2/rest/services/LIO_Imagery/Ontario_Imagery_Web_Map_Service/MapServer | Open Government Licence – Ontario (to confirm per image collection) |
| **Plane labels (own dataset)**: 272 image chips of 7 Ontario airports, 173 planes boxed by hand in LabelMe | Training and testing the plane detector | Chips: notebook 03 (from the Ontario imagery above); labels: `data/plane_labels/` | Own work; imagery under Open Government Licence – Ontario |
| Sentinel-2 Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`, Google Earth Engine) | 10 m comparison: shows why free satellite imagery cannot resolve planes | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED | Copernicus Sentinel data, free and open |

The CSV in `data/` comes from the 2026-10-18 download. The notebook was last run on 2026-11-07 (86,345 rows worldwide); the Canadian counts were unchanged. OurAirports updates nightly, so counts can change slightly on a new download. The notebook downloads it directly:

```python
URL = 'https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv'
```

Datasets are primarily Canadian, as the course requires.

## How to run
1. Open `code/01_airports_loader_and_map.ipynb` in Google Colab.
2. Replace `YOUR_PROJECT_ID` with your own Google Cloud project ID.
3. Runtime → Run all.
4. On the map: use the layers menu (top right) to switch types and imagery on and off; use the inspector tool and click a point to see its details under "Objects".
5. The last cell builds the whole map with one function, `build_airport_map(airports_fc, center=(lat, lon), zoom=..., s2_compare=...)`.

**Dashboard (notebook 02):**
1. Open `code/02_streamlit_ui.ipynb` in Google Colab (no Earth Engine login needed).
2. Runtime → Run all, then click the `trycloudflare.com` link printed by the last cell. The link works while the Colab session is open.
3. Runtime → Change runtime type → **T4 GPU** first, and allow Google Drive when asked (the trained model is loaded from Drive).
4. Click an airport for details, draw a box (square tool) for an area summary, or ask the assistant (try `YOW`, `how many airports in ON`, `help`, `reset`).
5. **Plane detection:** zoom in to an Ontario airport (e.g. type `YOW`), draw a box about 1 km across or smaller over the aprons, and the planes appear as red boxes. Set the confidence and click Apply.

**Plane dataset and model (notebooks 03 and 04):**
1. `code/03_plane_dataset_chips.ipynb` cuts 1024 × 1024 px chips (~215 m, 0.21 m per pixel) over the aprons of 7 Ontario airports (apron locations from OpenStreetMap) and saves them to Google Drive. The planes were then boxed by hand in LabelMe (free open-source version).
2. `code/04_train_plane_detector.ipynb` (T4 GPU) fine-tunes the detector and tests it on chips it never saw. The trained model (`plane_detector.pt`, ~170 MB) is too large for GitHub and stays in Google Drive; re-running notebook 04 recreates it.

## Folder contents
```
students/nifemi_ogunmeru/
├── README.md                                this file: project description and weekly progress log
├── code/01_airports_loader_and_map.ipynb    load + clean data, build the interactive map
├── code/02_streamlit_ui.ipynb               Streamlit dashboard (course UI draft) with AI plane detection, runs on Colab
├── code/03_plane_dataset_chips.ipynb        cuts the image chips for labelling
├── code/04_train_plane_detector.ipynb       trains and tests the plane detector
├── data/canada_airports_clean.csv           cleaned table: 2,477 facilities
├── data/chips_index.csv                     where each of the 272 image chips is (corner coordinates)
├── data/plane_labels/                       41 LabelMe files with the 173 hand-drawn plane boxes (images stay in Google Drive)
├── docs/PRD.md                              project plan: requirements, decisions, timeline
├── docs/progress2_update.md                 Progress 2 update: work since Oct 19, UI status, buttons and their functions
└── results/
    ├── dashboard.png                      the Streamlit dashboard (overview)
    ├── dashboard_popup.png                clicking an airport: details popup
    ├── dashboard_box_summary.png          drawing a box: area summary in the left panel
    ├── dashboard_yow_chat.png             assistant command "YYZ": zoom to Toronto Pearson, planes visible
    ├── dashboard_detection_yow.png        AI detection at Ottawa airport (draw a box → planes in red)
    ├── dashboard_detection_ygk.png        AI detection at Kingston, an airport not used in training
    ├── plane_detections_test.png          test chips: hand-drawn boxes (green) vs model (red)
    ├── training_loss.png                  training loss per epoch
    ├── test_metrics.json                  test scores
    ├── map_canada_airports.png            the national airport map
    ├── yow_aerial_planes.png              Ottawa airport (YOW) in Ontario aerial imagery (~20 cm): planes visible
    └── yow_sentinel2_vs_aerial.png        same airport in Sentinel-2 (10 m): planes not resolvable
```

## Pipeline (notebook 01)
1. Download OurAirports (86,258 rows worldwide on 2026-10-18).
2. Keep Canada: 3,369 rows.
3. Keep 5 active types (large / medium / small airports, heliports, seaplane bases): 2,479 rows.
4. Remove junk rows, printing each one first: names containing "duplicate", and "large airports" with no IATA code. This removed 2 fake copies of Edmonton (YEG). One was labelled Alberta but placed in Saskatchewan, which a coordinate check alone could not catch.
5. Bounding-box check on coordinates: 0 points outside Canada.
6. Convert to an Earth Engine FeatureCollection and map one coloured layer per type.
7. Add the Ontario aerial imagery layer (planes visible when zoomed in) and a Sentinel-2 comparison layer for YOW.
8. Wrap the map in one function, `build_airport_map()`, so the shared UI can call it.

**Result: 2,477 facilities** (15 large, 233 medium, 1,254 small, 525 heliports, 450 seaplane bases).

## AI: plane detection (notebooks 03, 04 and the dashboard)
**Why aerial imagery:** a Boeing 737 (~40 m) is about 4 pixels in Sentinel-2 (10 m), but about 190 pixels in the Ontario imagery (0.21 m per pixel).

**Dataset:** 272 chips of 1024 × 1024 px over the aprons of 7 Ontario airports (YYZ, YOW, YTZ, YHM, YKF, YXU, YOO). All chips were checked by hand in LabelMe: 41 contain planes, **173 planes** in total (YOO 70, YTZ 31, YYZ 23, YXU 20, YKF 15, YHM 7, YOW 7); chips without planes are used as "no plane" examples. A few chips have grey squares where a tile failed to download; chips with more than 25% missing were dropped.

**Model:** Faster R-CNN (ResNet-50 FPN v2) pretrained on the COCO dataset and fine-tuned for one class, *plane*, in PyTorch on a Colab T4 GPU (20 epochs). Every training chip is also shown rotated and mirrored, because aerial images have no "up".

**Test results** (20 held-out chips with 31 planes the model never saw, IoU ≥ 0.5, confidence ≥ 0.5):

| Precision | Recall | F1 | AP50 | Planes found | False alarms | Missed |
|---|---|---|---|---|---|---|
| 0.88 | 0.97 | 0.92 | 0.99 | 30 of 31 | 4 | 1 |

**In the dashboard:** a drawn box (up to ~1 km across, Ontario only) is cut into the same 1024 px chips, the model runs on each chip, planes split across two chips are joined back into one box, and the results are drawn on the map with their confidence. At Kingston (YGK), an airport **not used in training**, it found 6 planes (average confidence 0.93).

**Limitations:** the test set is small (31 planes) and comes from the same 7 airports as the training chips, so the scores are optimistic for new airports; Oshawa (YOO) supplies 40% of the planes (mostly small private aircraft). Detection only works where open sub-metre imagery exists (Ontario). The model detects anything shaped like an aircraft from above, including static display aircraft (e.g. the jet outside London airport's terminal).
## Milestones
| Date | Milestone | Status |
|---|---|---|
| Oct 19 | Progress 1: topic, objectives, datasets, tools | Done |
| Nov 9 | Progress 2: uniform UI connection | Review done; dashboard built right after (behind schedule, caught up the same day) |
| Nov 23 | Progress 3: AI component and testing | AI integrated in the dashboard; tested on held-out chips and an unseen airport |
| Dec 7 | Final delivery | Upcoming |

## Weekly progress log
<!-- Newest week first. Keep each week to 3–6 bullets: Done / Problems & fixes / Next. -->

### Week of Nov 16 – 22
- **Done:** plane dataset. Notebook 03 cut 272 chips (0.21 m per pixel) over the aprons of 7 Ontario airports, using OpenStreetMap apron outlines to avoid empty fields. All chips were labelled in LabelMe: 173 planes in 41 chips.
- **Done:** trained a Faster R-CNN plane detector (notebook 04). On 20 held-out chips: precision 0.88, recall 0.97, F1 0.92, AP50 0.99.
- **Done:** connected the model to the dashboard. Draw a box on an Ontario airport → the planes appear as red boxes with their confidence, and the count shows in the AI panel. Tested at Kingston (YGK), an airport not in the training data: 6 planes found.
- **Problems & fixes:** LabelMe's paid app was not needed (free open-source version via pip, with Python 3.12 because a dependency did not support Python 3.14 yet). Some chips had missing tiles (filled grey; chips over 25% missing dropped). Planes split across two chips are joined back into one box. Dashboard fixes: deleting a box clears the table and detections; confidence is set with an Apply button (fast repeated changes had frozen the map); `reset` and the project button now reset the whole dashboard.
- **Next:** Progress 3 review (Nov 23); then case studies, user manual and poster for Dec 7.

### Week of Nov 9 – 15
- **Done:** Progress 2 review. The RA accepted an Ontario focus for plane detection if no better combination of open datasets exists, and pointed to the course document's UI draft as the required layout.
- **Done:** dataset survey. Open sub-metre imagery exists for Ontario (~20 cm), Quebec (20 cm), Calgary and Edmonton (city imagery, licences unclear), while BC and Saskatchewan are paid or members-only. US NAIP (0.6 m, public domain) is available in Earth Engine. Decision: deploy on Ontario; train on Ontario plus a few US NAIP airports.
- **Done:** Streamlit dashboard (`code/02_streamlit_ui.ipynb`) laid out from the course's UI draft: assistant and area summary on the left, map with airport layers, popups, Ontario aerial imagery and a draw-a-box tool on the right, one button per class project at the bottom. Runs on Colab through a Cloudflare quick tunnel.
- **Problems & fixes:** the first version was slow, because 2,477 separate markers were rebuilt on every click. It now uses 5 compact GeoJSON layers on a canvas, builds the map once and draws a fresh copy each run, and runs the page once per action. A refresh bug (the map collapsing to a thin strip) was fixed by drawing from a copy of the saved map.
- **Done:** Progress 2 update document with screenshots: [`docs/progress2_update.md`](docs/progress2_update.md) (Nov 12, as requested by the RA).
- **Next:** build the labelled plane dataset (LabelMe), train a detector on a Colab GPU, connect it to the "draw a box" tool.

### Week of Nov 2 – 8
- **Done:** first upload of the project files to the course repository (README, notebook 01, cleaned data, project plan, map screenshot).
- **Done:** added open Ontario aerial imagery (~20 cm) to the map. Parked aircraft are clearly visible at the gates of Ottawa airport (YOW).
- **Done:** Sentinel-2 comparison at YOW (clearest summer 2025 image, Aug 15, ~0% cloud). Planes are only blurry blobs at 10 m, which confirms the RA's point and justifies switching imagery. A moving aircraft shows as a red-green-blue streak, because Sentinel-2 captures colour bands a fraction of a second apart.
- **Done:** packaged the whole map in one function, `build_airport_map()`, ready for the shared UI.
- **Problems & fixes:** no free, current, sub-metre imagery covers all of Canada (Ontario and Quebec publish open 20 cm imagery; BC's is paid). Plane detection becomes a case study on Ontario airports; the map itself stays national.
- **Next:** Progress 2 review; ask the RA about the UI framework and whether open aerial imagery is acceptable; start labelling planes at YOW and YYZ with LabelMe.

### Week of Oct 26 – Nov 1
- **Done:** received the course repository and collaborator access; prepared the project files in the course folder layout.

### Week of Oct 19 – 25
- **Done:** Progress 1 review with a live map demo. The RA approved the project and asked for more work on **airplane detection**, including other (higher-resolution) imagery and other models.
- **Done:** imagery survey for aircraft detection. Sentinel-2 at 10 m shows a 737 as about 4 pixels, which is too coarse. Candidates: Ontario open orthophotos (16–50 cm, Canadian, open licence), Planet SkySat public sample in GEE (0.8 m, 2014–2016), PlanetScope via Planet Education & Research (about 3 m), RarePlanes (labelled aircraft, CC BY-SA 4.0, non-Canadian).
- **Next:** check SkySat coverage of Canadian airports in GEE; view an Ontario orthophoto of Ottawa airport (YOW); decide on imagery and model.

### Week of Oct 12 – 18
- **Done:** project plan (`docs/PRD.md`); notebook 01 built and verified in Colab (runs top to bottom on a fresh runtime, no errors).
- **Problems & fixes:** found 2 fake "large airport" rows for Edmonton and added rules to remove them, printing each removed row. Point-size styling is ignored by geemap, so airport type is shown by colour, with large airports drawn on top.
- **Next:** prepare the Progress 1 review.
