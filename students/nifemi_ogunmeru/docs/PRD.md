# PRD – Project 8: Where Are the Airports in Canada?

**Course:** OSS4008A – Introduction to Remote Sensing and Applications (Carleton University)
**Owner:** Nifemi Ogunmeru (individual project, no group work)
**Instructor:** M. Hamed Mozaffari | **Research Assistant (progress reviews):** Aditya Rawal
**Version:** 1.3 (draft) | **Date:** 2026-11-04
**Status:** Draft. Scope items marked **[TO CONFIRM]** must be validated with the RA at the Oct 19 meeting.

---

## 1. Summary

Build an interactive, web-ready map of Canadian airports and related aviation facilities on Google Earth Engine (GEE), with an AI component working on satellite imagery, packaged as a module that plugs into the class-wide shared dashboard (one comprehensive tool made of all students' projects).

The project is graded by the instructor's discretion (up to 15% of the final grade) on quality, functionality, presentation, consistent progress and reporting, challenges and solutions, and how clean and understandable the work is for an end user.

## 2. Background and context

- The instructor's brief assigns each student one topic. Mine is **#8: "Where are infrastructures in Canada (interactive Map of Industries) – Airports and airplanes."**
- Every topic follows the same four phases (Backend → GEE map notebook → UI integration → assessment/case studies).
- All projects are merged at the end into one web-based dashboard (map on the right, AI chat/analysis panel on the left, one button per project along the bottom). For this year the map **reloads fully** when a project button is pressed; a shared background map (like windy.com) is a future goal.
- Time window is shorter than it sounds. The course runs about four months, but my **final delivery date is Dec 7, 2026**, which is roughly 7 weeks after the first review on Oct 19.

### 2.1 Decisions made so far
| Date | Decision | Reason |
|---|---|---|
| 2026-10-15 | Scope the project to **airport locations** across Canada and **skip full airplane detection** | Sentinel-2 (10 m) is too coarse to detect individual aircraft; sub-metre imagery is not free at national scale; focus on doing the airport part well |
| 2026-10-15 | Use **OurAirports** open data as the primary airport source | Single clean CSV, public domain per its own statement, includes type, codes, coordinates |
| 2026-10-15 | Clean junk rows (entries labelled "(Duplicate)" and "large airports" without an IATA code) | Found 2 junk rows that would have shown a fake major airport |
| 2026-10-15 | **Propose** a tiered plan (core map, AI component, stretch) **[TO CONFIRM]** | The brief expects an AI model in the pipeline; a plain CSV plot risks being read as incomplete |
| 2026-10-18 | Rebuilt notebook 01 step by step and verified it in Colab (fresh runtime, no errors) | Earlier version had never been run in Colab |
| 2026-10-18 | Show airport type by **colour only**; drop the point-size setting | geemap ignores point size for FeatureCollection layers; keeping layers as FeatureCollections keeps click info (inspector) working. Large airports drawn last so they stay on top |
| 2026-10-18 | Print junk rows before removing them | A future data refresh must not silently drop a real airport |
| 2026-10-19 | **Progress 1 review:** RA approved the project. RA asked to look further into **airplane detection** (part of the project scope) and try other models. Decision #1 (skip airplane detection) is **reopened**: aircraft detection moves from "stretch" to "investigate now" | RA feedback at Oct 19 review |
| 2026-10-27 | Moved the project into the course repository (`students/nifemi_ogunmeru/`, with `code/`, `data/`, `results/`); weekly progress log kept in the folder README | TA email of Oct 27 |
| 2026-10-27 | Datasets must be **primarily Canadian**; non-Canadian data only when no Canadian data exists. Own datasets allowed (annotation with LabelMe) | TA email of Oct 27 |

## 3. Goals and non-goals

### Goals
1. Show every active airport, heliport and seaplane base in Canada on an interactive GEE map, classified by type, with useful information on click.
2. Include a real, honestly scoped **AI component** that uses satellite imagery (see §7).
3. Deliver a module that follows a common standard so it can merge into the shared dashboard without rework.
4. Keep a clean, well-documented, regularly updated GitHub folder, since progress is evaluated through it.
5. Produce all required deliverables on time (see §10).

### Non-goals (this year)
- Canada-wide detection of individual aircraft.
- Real-time flight tracking or live aircraft data.
- A custom shared background map across all projects (future work).
- Any non-Python implementation.

## 4. Users and stakeholders

| Stakeholder | Needs |
|---|---|
| Instructor (Mozaffari) | A functional, clean, mergeable tool; evidence of learning; discretionary grading |
| RA (Aditya Rawal) | Regular GitHub updates; clear progress at each 9-minute review; achievable plan |
| End user (non-expert, e.g. industry or planner) | Simple map, clear legend, easy information, minimal clicks, understandable report |
| Future students | Documented, modular code that can be maintained and extended |
| Me | A project I can explain in detail and put on a résumé (data pipeline, back-end, front-end, GitHub workflow) |

## 5. Scope tiers

| Tier | Content | Status |
|---|---|---|
| **Core (must)** | Cleaned airport dataset → GEE FeatureCollection → interactive map with type layers, click info, legend, summary statistics, filters | Core v1 built (see §11) |
| **Aircraft detection (should, RA request 2026-10-19)** | Detect aircraft at selected Canadian airports. Test (1) a pretrained aerial "plane" detector on open high-resolution orthoimagery and (2) the same on Sentinel-2 at 10 m, to measure what resolution is actually needed | Imagery survey done (§7); feasibility spike next |

## 6. Functional requirements

Priority: **M** = must, **S** = should, **C** = could.

### 6.1 Map and data
| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-1 | Display all active Canadian airports as interactive points, styled and toggled by type | M | Five type layers; large and medium on by default; layer control works |
| FR-2 | Click a point to see name, ident, IATA/ICAO, type, municipality, province, elevation, scheduled service | M | Inspector or popup shows all listed fields |
| FR-3 | Legend explaining colours | M | Legend visible and accurate |
| FR-4 | Filter by type, province, and scheduled service | S | Map updates within a few seconds |
| FR-5 | Search by name or code, then zoom to the result | S | Typing a valid code (e.g. YOW) centres the map |
| FR-6 | Draw a box or pick a location and list/count airports inside | S | Count matches a pandas check on the CSV |
| FR-7 | Summary panel with counts by type and province | S | Numbers match the cleaned CSV |
| FR-8 | Satellite basemap option (e.g. Sentinel-2 mosaic) | S | Toggle on/off without errors |
| FR-9 | Runway geometry from OpenStreetMap `aeroway` data for major airports | C | Runway lines render for selected airports; ODbL attribution included |

### 6.2 AI component (proposed, see §7)
| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-10 | Documented dataset of imagery chips and labels (with licences and sources) | M (if AI is confirmed) | Folder with images and labels plus a data card |
| FR-11 | Train or fine-tune a small PyTorch model in Colab, with metrics reported | M (if AI is confirmed) | Reproducible notebook; metrics on a held-out set |
| FR-12 | Run inference on a user-selected area and overlay results on the map | S | Selecting an airport area shows model output within an acceptable time |
| FR-13 | Compare AI output to the CSV and flag discrepancies | S | List of mismatches with a short explanation |

### 6.3 Integration (shared dashboard)
| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-14 | Expose the map through one entry function with a documented interface (inputs, outputs) | M | Another notebook can import and call it **[interface TO CONFIRM with RA]** |
| FR-15 | Keep UI-independent logic (data, model, map layers) separate from UI code | M | Modules importable without running UI |
| FR-16 | Follow the shared UI layout: map on right, AI/analysis panel on left, project button at bottom | S | Matches the instructor's sketch |

### 6.4 Documentation and process
| ID | Requirement | Pri |
|---|---|---|
| FR-17 | README with data sources, cleaning steps, how to run, status log | M |
| FR-18 | User manual for a non-expert | M |
| FR-19 | Case-study testing results | M |
| FR-20 | Maintenance and future-improvement plan, as a map or diagram | M |
| FR-21 | Printed poster | M |
| FR-22 | GitHub folder updated at least weekly, with papers, datasets and files organised | M |

## 7. AI component design

**Update 2026-10-19:** at the Progress 1 review the RA asked to pursue **aircraft detection** with higher-resolution imagery, since airplanes are part of the project scope. Aircraft detection (old option C) is now the main AI direction; options A and B below are fallbacks if it proves infeasible.

**Why resolution matters:** a Boeing 737 (~40 m) is about 4 pixels in Sentinel-2 (10 m), about 13 pixels at 3 m, and 50+ pixels below 1 m. Detectors need roughly 10+ pixels across an object.

| Candidate imagery / data | Resolution | Canadian? | Licence | Notes |
|---|---|---|---|---|
| Ontario open orthophotos (Geospatial Ontario) | 16–50 cm | Yes | Open Government Licence – Ontario | Aerial, not satellite; settled Ontario incl. Eastern Ontario |
| Planet SkySat public sample (GEE) | 0.8 m | Coverage to check | CC BY 4.0 | 2014–2016 only |
| PlanetScope (Planet Education & Research, Basic) | ~3 m | Yes | Non-commercial research | 3,000 km²/month, 30-day delay |
| RarePlanes (labelled aircraft) | Very high (WorldView-3) | No | CC BY-SA 4.0 | Training data only if no Canadian labelled set exists |
| Own labelled set (LabelMe on Ontario orthophotos) | 16–50 cm | Yes | Own | Encouraged by the TA (Oct 27 email) |

**Earlier options (now fallbacks):**

| Option | Idea | Pros | Cons / risks |
|---|---|---|---|
| **A. Airport-type check** | Classify Sentinel-2 chips around each airport (paved runway / gravel or grass strip / water / not visible) and compare with the CSV's type | Cheap labels, clear story ("verify the database with imagery") | Heliports and tiny strips are only a few pixels at 10 m |
| **B. Runway/airport-surface segmentation** | Train a small segmentation model on Sentinel-2 chips, using OpenStreetMap runway polygons as free (weak) labels | Avoids manual annotation; directly visual on the map | OSM labels contain errors; narrow runways are 3–6 pixels wide at 10 m |
| **C. Aircraft detection at a few big airports (stretch)** | Fine-tune a pretrained detector on an existing annotated aircraft dataset, run on high-resolution imagery | Matches the brief's original idea | Needs high-resolution imagery and licensing; heavy compute; time risk |

**Next step:** feasibility spike: check SkySat coverage of Canadian airports in GEE, view an Ontario orthophoto of Ottawa airport (YOW), and test a pretrained aircraft detector on it versus Sentinel-2. Then choose imagery and model with the RA.

Dataset licences, availability and terms for any external dataset (including annotated aerial/satellite aircraft datasets) must be **verified before use** and documented in the README.

## 8. Non-functional requirements

- **Language/stack:** Python only. GEE Python API, geemap, pandas, PyTorch (AI), Flask or Streamlit for UI integration **[framework TO CONFIRM]**.
- **Colab:** every notebook runs **top to bottom without errors on a fresh runtime**.
- **Code quality:** all code commented; sources cited in comments; meaningful names; no secrets in code (project ID is a placeholder).
- **Data:** open data only, **primarily from Canada** (non-Canadian data only when nothing Canadian exists, with the reason documented); licences recorded; data files versioned with a download date; files under 50 MB, large imagery linked or downloaded by code instead of committed.
- **Modularity:** map, data, and AI code in separable modules so the project merges cleanly.
- **Usability:** clear legend, minimal clicks, readable by a non-expert.
- **Performance:** map must stay responsive; avoid heavy server-side computation at national zoom levels (the brief warns about lag at high altitude). Use aggregation or simplification when zoomed out.
- **Reproducibility:** saved notebooks keep outputs visible (Colab: uncheck "Omit code cell output when saving this notebook").
- **Novelty (bonus ideas from the brief):** multi-language interface (EN/FR is natural for Canada) or multi-platform capability.

## 9. Data requirements

### 9.1 Primary dataset: OurAirports
- Source: https://ourairports.com/data/ (CSV also mirrored on GitHub). Re-check licence terms before the final report.
- Cleaning rules (implemented in `code/01_airports_loader_and_map.ipynb`): keep `iso_country == 'CA'`; keep active types (large, medium, small airports, heliports, seaplane bases); drop "(Duplicate)" rows and large airports with no IATA code; bounding-box check on coordinates.
- Result on the 2026-10-18 download: **2,477** facilities.

| Type | Count |
|---|---|
| Large airport | 15 |
| Medium airport | 233 |
| Small airport | 1,254 |
| Heliport | 525 |
| Seaplane base | 450 |

295 facilities have scheduled passenger service. Counts by province: ON 734, BC 404, AB 347, QC 303, SK 187, MB 153, NT 79, NS 72, NL 53, NU 50, YT 48, NB 42, PE 5.

### 9.2 Candidate secondary datasets
| Dataset | Use | Notes |
|---|---|---|
| Sentinel-2 surface reflectance in GEE (`COPERNICUS/S2_SR_HARMONIZED`, 10 m, free) | Basemap, AI input chips | Confirm the exact asset ID in the GEE catalog before use; cloud property is `CLOUDY_PIXEL_PERCENTAGE` |
| OpenStreetMap `aeroway` features (runways, aprons, terminals) | Runway geometry, weak labels | ODbL licence: attribution required; fetched outside GEE (Overpass API or an extract) |
| Government of Canada open data (airports/aerodromes) | Cross-check of the CSV | Check format and licence |
| Landsat 8/9 Collection 2 Level 2 (`LANDSAT/LC08/C02/T1_L2`) | Optional context imagery | Apply scale factors (×0.0000275, −0.2) |

### 9.3 Known data-quality issues
- OurAirports is crowdsourced; some rows are stale, duplicated or mislabelled (two junk "large airport" rows already found).
- Many small airstrips and heliports have no codes or elevation.
- Facility type is user-contributed and may be wrong; this is a reason for the AI cross-check.

## 10. Deliverables (from the instructor's brief)

A single folder containing:
1. Web-based application: back-end and front-end code, all runnable on Google Colab.
2. User manual.
3. Code uploaded to the course GitHub folder, with a detailed README.
4. Case-study results of testing the tool on the data.
5. Maintenance and future-improvement plan, drawn as a map (AI may help).
6. Printed poster of the work.
7. Licence disclaimer (instructor decides).

If merged into the shared tool, everything must follow the same standard and remain functional.

### Evaluation criteria (project track)
Functionality; usefulness and user-friendliness; bugs and errors; presentation; modularity (can merge with other projects); following the terms with regular progress and an active GitHub repository; novel ideas (e.g. multi-language, multi-platform); value of the data pipeline.

### Mandatory rules
Python only; Colab must run top to bottom; all code commented; deployable with minimal bugs using un-licensed data and code **[exact meaning TO CONFIRM]**; if AI is used I must understand how it works in detail and keep it as simple as possible; no ownership claim unless the instructor specifies.

## 11. Current state (as of 2026-11-04)

| Item | Status |
|---|---|
| Project folder | Course repository: `students/nifemi_ogunmeru/` (`README.md`, `code/`, `data/`, `docs/`, `results/`) |
| `01_airports_loader_and_map.ipynb` | Rebuilt and **verified in Colab**: runs top to bottom on a fresh runtime. Map has 5 type layers, legend and click info (inspector). FR-1, FR-2, FR-3 met |
| `data/canada_airports_clean.csv` | Regenerated from the 2026-10-18 download (2,477 rows) |
| Course GitHub repository | Collaborator access accepted; first upload of project files in progress (week of Nov 2) |
| Progress 1 review (Oct 19) | Done. Project approved; RA asked for further work on airplane detection |
| AI component | Imagery survey done (§7); aircraft-detection feasibility spike next |

## 12. Timeline

Review checkpoints are Mondays 1:30–3:30 PM (online, 9 minutes each).

| Window | Focus | Milestone |
|---|---|---|
| Oct 15–19 | Verify notebook in Colab, set up GitHub folder, prepare talking points | **Progress 1 (Oct 19):** topic, objectives, datasets, tools, next steps |
| Oct 20–26 | Map polish (legend, filters, stats); AI feasibility spike; lock AI option | |
| Oct 27–Nov 2 | AI data preparation and baseline model; define integration interface | |
| Nov 3–9 | Wrap the map into the shared-UI interface; test | **Progress 2 (Nov 9):** uniform UI connection |
| Nov 10–16 | Train and evaluate the AI model | |
| Nov 17–23 | Integrate AI into the map; testing plan | **Progress 3 (Nov 23):** AI and testing |
| Nov 24–Dec 7 | Case studies, user manual, poster, maintenance map, final cleanup | **Delivery (Dec 7)** |

Other courses will compete for the same weeks, so keep buffer in the last two weeks.

## 13. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| AI not feasible at 10 m resolution | Medium | Feasibility spike first; fall back to Option A; be explicit about trade-offs in the report |
| Scope mismatch with the brief | Medium | Confirm tiers with the RA on Oct 19 |
| Short real timeline (about 7 weeks) | High | Core first, AI scoped small, stretch only if ahead |
| UI integration unknown | Medium | Ask for the framework and interface early; keep logic separate from UI |
| GEE quota, auth or lag | Medium | Pre-filter data; aggregate at low zoom; cache results |
| Data quality | Medium | Documented cleaning rules; cross-check with a second source |
| Colab session loss or non-reproducible notebooks | Medium | Commit to GitHub often; run top to bottom before each commit |
| Licence ambiguity | Low–Medium | Record licences; ask the instructor about the "un-licensed" wording |
| Competing coursework | High | Weekly plan; protect two buffer weeks |

## 14. Success metrics

- Notebook(s) run top to bottom on a fresh Colab runtime with no errors.
- Map counts match the CSV exactly (2,477 facilities unless the data is refreshed).
- All FR marked **M** are done by Dec 7; **S** items done as time allows.
- AI model has a reproducible training notebook and reported metrics on a held-out set. Numeric targets set after the spike and agreed with the RA.
- GitHub folder updated at least weekly, with a clear README and status log.
- I can explain every part of the code and the AI method in plain language during reviews.

## 15. Open questions for the RA / instructor

1. ~~Is the tiered scope (map plus an AI component on airport imagery, no full aircraft detection) acceptable?~~ **Answered Oct 19:** overall project approved, but airplane detection should be explored with higher-resolution imagery.
2. Which UI framework and integration interface should my module follow (Flask, Streamlit, other)?
3. What does "un-licensed data and code" mean exactly (open/permissive licences, or no licence at all)?
4. Required structure or naming for the GitHub folder?
5. What evidence of the AI model is expected (metrics, examples, size of dataset)?
6. Poster size and format?
7. Should airports near the border (US side) be included for context?

## 16. Appendix A – Oct 19 meeting outline (9 minutes)

| Time | Content |
|---|---|
| 1 min | Topic: Project 8, interactive airport map of Canada on GEE |
| 2 min | Objectives: airport map; AI component; plug into the shared dashboard |
| 2 min | Datasets: OurAirports, Sentinel-2 in GEE, OSM `aeroway`, optional government data |
| 1.5 min | Tools and methods: Python, GEE API, geemap, pandas, Colab, PyTorch, Flask/Streamlit, GitHub |
| 1.5 min | Next steps and timeline (Nov 9, Nov 23, Dec 7) |
| 1 min | Questions from §15 (especially 1 and 2) |

## 17. Appendix B – Project folder structure (course repository)

Repository: https://github.com/AdityaR28/OSS4008A---Introduction-to-Remote-Sensing-and-Applications

```
students/nifemi_ogunmeru/
├── README.md        project description and weekly progress log
├── docs/
│   └── PRD.md       this document
├── code/            Colab notebooks (.ipynb)
├── data/            small datasets and CSVs
└── results/         figures, map screenshots, outputs
```
