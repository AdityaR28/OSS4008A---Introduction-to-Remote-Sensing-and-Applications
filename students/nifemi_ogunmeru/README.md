# Project 8 – Where are the airports in Canada?

**Student:** Nifemi Ogunmeru | **Course:** OSS4008A – Introduction to Remote Sensing and Applications (Carleton University)

An interactive Google Earth Engine map of every active airport, heliport and seaplane base in Canada, built in Python on Google Colab. It will plug into the class-wide dashboard. The AI component is aircraft detection at selected Canadian airports, which is being investigated after the RA's feedback at the Oct 19 review.

## Objectives
1. Interactive map: airports by type, click for details, legend; then filters, search and summary statistics.
2. AI component: detect aircraft on high-resolution imagery at selected Canadian airports, and measure how image resolution affects detection.
3. Package the map so it plugs into the shared dashboard.

## Data
| Dataset | Used for | Source | Licence |
|---|---|---|---|
| OurAirports `airports.csv` | Airport locations, type, codes, province | https://ourairports.com/data/ | Public domain (per OurAirports) |

Downloaded on 2026-10-18. OurAirports updates nightly, so counts can change slightly on a new download. The notebook downloads it directly:

```python
URL = 'https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv'
```

Datasets are primarily Canadian, as the course requires.

## How to run
1. Open `code/01_airports_loader_and_map.ipynb` in Google Colab.
2. Replace `YOUR_PROJECT_ID` with your own Google Cloud project ID.
3. Runtime → Run all.
4. On the map: use the layers menu (top right) to switch types on and off; use the inspector tool and click a point to see its details under "Objects".

## Folder contents
```
students/nifemi_ogunmeru/
├── README.md                                this file: project description and weekly progress log
├── code/01_airports_loader_and_map.ipynb    load + clean data, build the interactive map
├── data/canada_airports_clean.csv           cleaned table: 2,477 facilities
├── docs/PRD.md                              project plan: requirements, decisions, timeline
└── results/map_canada_airports.png          screenshot of the interactive map
```

## Pipeline (notebook 01)
1. Download OurAirports (86,258 rows worldwide on 2026-10-18).
2. Keep Canada: 3,369 rows.
3. Keep 5 active types (large / medium / small airports, heliports, seaplane bases): 2,479 rows.
4. Remove junk rows, printing each one first: names containing "duplicate", and "large airports" with no IATA code. This removed 2 fake copies of Edmonton (YEG). One was labelled Alberta but placed in Saskatchewan, which a coordinate check alone could not catch.
5. Bounding-box check on coordinates: 0 points outside Canada.
6. Convert to an Earth Engine FeatureCollection and map one coloured layer per type.

**Result: 2,477 facilities** (15 large, 233 medium, 1,254 small, 525 heliports, 450 seaplane bases).

## Milestones
| Date | Milestone | Status |
|---|---|---|
| Oct 19 | Progress 1: topic, objectives, datasets, tools | Done |
| Nov 9 | Progress 2: map connected to the shared UI | Upcoming |
| Nov 23 | Progress 3: AI component and testing | Upcoming |
| Dec 7 | Final delivery | Upcoming |

## Weekly progress log
<!-- Newest week first. Keep each week to 3–6 bullets: Done / Problems & fixes / Next. -->

### Week of Nov 2 – 8
- **Done:** first upload of the project files to the course repository (README, notebook 01, cleaned data, project plan, map screenshot).
- **Next:** Progress 2 (Nov 9): connect the map to the shared UI; aircraft-detection feasibility spike.

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
