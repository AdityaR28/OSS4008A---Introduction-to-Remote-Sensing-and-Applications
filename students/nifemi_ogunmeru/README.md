# Project 8 – Where are the airports in Canada?

**Student:** Nifemi Ogunmeru | **Course:** OSS4008A – Introduction to Remote Sensing and Applications (Carleton University)

An interactive Google Earth Engine map of every active airport, heliport and seaplane base in Canada, built in Python on Google Colab. It will plug into the class-wide dashboard. A Streamlit dashboard (notebook 02) follows the course's UI draft: AI assistant on the left, interactive map on the right, project buttons at the bottom. Planes are visible on the map through open Ontario aerial imagery (about 20 cm per pixel). The AI component, aircraft detection, will be a case study at selected Ontario airports, following the RA's feedback at the Oct 19 review.

## Objectives
1. Interactive map: airports by type, click for details, legend; then filters, search and summary statistics.
2. AI component: detect aircraft on high-resolution imagery at selected Canadian airports, and measure how image resolution affects detection.
3. Package the map so it plugs into the shared dashboard.

## Data
| Dataset | Used for | Source | Licence |
|---|---|---|---|
| OurAirports `airports.csv` | Airport locations, type, codes, province | https://ourairports.com/data/ | Public domain (per OurAirports) |
| Ontario Imagery Web Map Service (Land Information Ontario) | Aerial imagery (~20 cm) where planes are visible | https://ws.lioservices.lrc.gov.on.ca/arcgis2/rest/services/LIO_Imagery/Ontario_Imagery_Web_Map_Service/MapServer | Open Government Licence – Ontario (to confirm per image collection) |
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
3. Click an airport for details, draw a box (square tool) for an area summary, or ask the assistant (try `YOW`, `how many airports in ON`, `help`).

## Folder contents
```
students/nifemi_ogunmeru/
├── README.md                                this file: project description and weekly progress log
├── code/01_airports_loader_and_map.ipynb    load + clean data, build the interactive map
├── code/02_streamlit_ui.ipynb               Streamlit dashboard (course UI draft), runs on Colab
├── data/canada_airports_clean.csv           cleaned table: 2,477 facilities
├── docs/PRD.md                              project plan: requirements, decisions, timeline
├── docs/progress2_update.md                 Progress 2 update: work since Oct 19, UI status, buttons and their functions
└── results/
    ├── dashboard.png                      the Streamlit dashboard (overview)
    ├── dashboard_popup.png                clicking an airport: details popup
    ├── dashboard_box_summary.png          drawing a box: area summary in the left panel
    ├── dashboard_yow_chat.png             assistant command "YOW": zoom to Ottawa airport, planes visible
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

## Milestones
| Date | Milestone | Status |
|---|---|---|
| Oct 19 | Progress 1: topic, objectives, datasets, tools | Done |
| Nov 9 | Progress 2: uniform UI connection | Review done; dashboard built right after (behind schedule, caught up the same day) |
| Nov 23 | Progress 3: AI component and testing | Upcoming |
| Dec 7 | Final delivery | Upcoming |

## Weekly progress log
<!-- Newest week first. Keep each week to 3–6 bullets: Done / Problems & fixes / Next. -->

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
