# Project 11: Water Bodies of Canada (Lakes, Rivers, Ponds)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdityaR28/OSS4008A---Introduction-to-Remote-Sensing-and-Applications/blob/main/students/devon_clermont-gauer/code/Dashboard.ipynb)

**Course:** OSS4008A – Introduction to Remote Sensing and Applications
**Student:** Devon Clermont-Gauer
**Instructor:** M. Hamed Mozaffari | **Research Assistant:** Aditya Rawal

An interactive map that identifies lakes, rivers and ponds across Canada from Sentinel-2 satellite imagery. A segmentation model (U-Net) classifies each pixel as water or not water; detected water bodies are then sorted into lakes, rivers and ponds by size and shape.

## Current status

| Part | Status |
|---|---|
| Dashboard UI (chat panel, map, reference layers, AI box detection) | Built |
| Integration-ready module (`water_layers.py`) | Shared-UI entry point and reference filter controls |
| Training dataset (14 regions, 665 tiles) | Exported and tiled |
| Test-set corrections in labelme (40 tiles) | Done |
| NDWI baseline | IoU 0.609 pooled, 0.468 averaged across regions |
| U-Net training and evaluation | IoU 0.891 pooled, 0.840 averaged across regions |
| AI detection in the dashboard (`water_ai.py`) | U-Net on a drawn box; parallel downloads, up to 20,000 km² |
| Lake / river / pond classification | Final delivery |

## Folder structure

```
students/devon_clermont-gauer/
├── code/
│   ├── Dashboard.ipynb                  # interactive dashboard with AI water detection (run this)
│   ├── water_dataset_export.ipynb       # builds the training dataset
│   ├── test_prep_and_baseline.ipynb     # band stats, labelme test-set prep, NDWI baseline
│   ├── unet_training.ipynb              # trains and evaluates the U-Net
│   ├── water_layers.py                  # reference map layers and shared-UI controls
│   ├── water_ai.py                      # AI water detection: runs the U-Net on a drawn area
│   └── band_stats.json                  # band rescaling values used by the model
├── data/          # dataset description (full dataset on Google Drive, see below)
├── docs/
│   └── project_plan.pdf
└── results/
    ├── performance/
    │   └── performance_tests.md
    ├── dashboard/
    │   └── dashboard_preview_V1.png
    │   └── dashboard_preview_V2.png
    ├── baseline/
    │   ├── ndwi_vs_dynamic_world.png
    │   └── ndwi_vs_corrected_masks.png
    ├── training/
    │   ├── training_curves.png
    │   ├── example_predictions.png
    │   └── unet_vs_baselines.png
```

## How to run

**Dashboard:** click the **Open in Colab** badge above, replace `YOUR-PROJECT-ID` in Cell 1 with your Google Earth Engine Cloud project ID, then **Runtime → Run all**. The notebook automatically downloads `water_layers.py`, `water_ai.py` and `band_stats.json` from this repo, and the trained model from Google Drive, so nothing needs to be set up by hand. Draw a box on the map and click **AI: water in drawn box** to run the U-Net (boxes up to 5000 km²). A T4 GPU runtime makes detection faster, but CPU works.

**Other notebooks** (open them in Colab from the `code/` folder):
- **water_dataset_export:** Cell 4 starts 14 Earth Engine exports to your Google Drive. If the exports already exist, run Cells 1–3, skip Cells 4–5, then run Cells 6–9.
- **test_prep_and_baseline:** run Cells 1, 2 and 5. Don't re-run Cell 3 after corrections have started: it overwrites the corrected labelme files. Cell 4 converts corrected labelme files into masks.
- **unet_training:** set **Runtime → Change runtime type → T4 GPU**, then run all cells.

## Method

**Water detection rule:** water absorbs near-infrared (Sentinel-2 B8) and shortwave-infrared (B11) light, so it appears dark in those bands while land appears bright. The Normalized Difference Water Index, NDWI = (B3 − B8) / (B3 + B8), is positive over water (McFeeters, 1996). The U-Net learns this spectral pattern plus shape and context, which helps it reject look-alikes such as shadows, wet soil and asphalt.

**Prediction:** the U-Net runs on overlapping 256 × 256 tiles, which are blended together (weighted towards each tile's centre) to avoid seams at tile edges. Gaps fully surrounded by detected water are filled if they are spectrally water (NDWI > 0); islands (NDWI < 0) remain land.

**Classification rules (after detection, at 10–20 m resolution only):**
- **River:** long, narrow shape. Elongation = (centre-line length)² / area ≥ 10, measured with islands filled in, so island-filled lakes aren't mistaken for rivers.
- **Lake:** compact shape, area ≥ 2 ha
- **Pond:** compact shape, area < 2 ha (Biggs et al., 2005)

Depth is the traditional lake/pond distinction, but it can't be measured from satellite imagery, so surface area is used instead.

## Dataset

Sentinel-2 image tiles paired with water masks, 256×256 pixels at 10 m (2.56 km per side). Summer composites (June–September 2024) to avoid ice. Whole regions are assigned to one split so the model is always tested on places it never trained on.

**Labelling rule:** water = visible open surface water (lakes, rivers, ponds, ocean). Vegetated wetlands and reed-filled potholes are not labelled as water; seasonal wetlands are shown separately by the dashboard's seasonal water layer. Ocean is labelled as water and is excluded later when classifying lakes.

| Split | Regions | Tiles |
|---|---|---|
| Train | algonquin, fraser_delta, lake_louise, mackenzie_delta, peace_athabasca, prairie_potholes, quebec_city, toronto | 396 |
| Validation | north_manitoba, ottawa | 112 |
| Test | iqaluit, kejimkujik, minnedosa, winnipeg | 157 |

**Full dataset (~372 MB):** [Google Drive folder](https://drive.google.com/drive/folders/1QlQR3KWzcUz68h5cVU4suzOqYILWnGtP?usp=sharing)
Each tile is a GeoTIFF: `images/` hold Sentinel-2 bands B2, B3, B4, B8, B11 (blue, green, red, NIR, SWIR); `masks/` hold water masks (1 = water, 0 = not water). The dataset isn't committed to GitHub .

**Trained model:** `unet_resnet34_best.pth` (~100 MB) is shared on [Google Drive](https://drive.google.com/file/d/1ZvIHmtM5tdOvZHpJC5JdmKNEVgkQbkL4/view?usp=sharing) because it exceeds the course's 50 MB file limit. The dashboard downloads it automatically.

## Data sources

| Dataset | Use | Licence |
|---|---|---|
| Sentinel-2 SR Harmonized (Copernicus/ESA) | Satellite imagery | Copernicus open data; attribution required |
| Dynamic World V1 (Google / WRI) | Training labels (water masks) | CC BY 4.0 |
| JRC Global Surface Water v1.4 | "All water" reference layer (country-scale view, not AI) | Free use with attribution |
| HydroSHEDS Free Flowing Rivers (WWF) | "Rivers" reference layer (approximate river courses, not AI) | Free use with attribution |
| LSIB Simple 2017 (US Department of State) | Canada boundary | Public domain |
| National Hydro Network (NRCan) | Planned: validating results against official Canadian data | Open Government Licence – Canada |

Global datasets are used for imagery and labels because no Canadian source offers free 10 m coverage nationwide; Canada's National Hydro Network is used for validation.

## Known limitations

**Detection**
- Water features narrower than ~10 m (small streams, very small ponds) are below the image resolution; rivers only 2–3 pixels wide (20–30 m) are often partly missed.
- Water under tree canopy, very shallow or turbid water, and dark vegetated channels may be missed.
- Gaps can remain in large water bodies where the summer composite shows cloud shadow, sun glint or algae, since these look like land in the bands as well.
- Persistent cloud reduces usable imagery in some northern regions.
- Training labels come from Dynamic World (itself a model output) and contain some errors; the model scores below Dynamic World on the test set (IoU 0.891 vs 0.929), partly because the corrected test masks were drafted from Dynamic World.
- Urban rivers are the weakest case (Winnipeg test region, IoU 0.783).
- The test set is small (40 hand-corrected tiles from 4 regions), so a few tiles can change a region's score noticeably.

**Scale**
- AI detection on areas larger than 400 km² runs at a coarser resolution (up to 60 m); water narrower than about three pixels (e.g. ~180 m wide at 60 m) may be missed, and accuracy at coarse resolution has not been formally evaluated (test scores apply at 10 m).
- AI detection is limited to areas up to 20,000 km²; reference layers cover country-scale viewing.
- Auto AI scans when a pan or zoom ends; Colab can't run timers, so there's no delay before scanning.

**Classification**
- Lakes, rivers and ponds are only classified at 10–20 m resolution (areas up to ~1,600 km²); coarser scans show all water.
- Classification uses shape and size only: rivers flowing into lakes join them into one body, classified by its overall shape.
- Ocean and Great Lake water (e.g. Iqaluit, Fraser Delta, Toronto) are labelled as lakes.
- Water bodies cut off by the edge of the scanned area are flagged as partial; their type may be less reliable.

## Progress log

| Date | Update |
|---|---|
| 2026-10-19 | **Progress 1:** project plan presented (objectives, datasets, tools, water and classification rules). |
| 2026-11-07 | Built the dashboard: chat placeholder, map of Canada, layer buttons (all, permanent, seasonal water, rivers), loading messages; map layers moved into `water_layers.py`. |
| 2026-11-07 | Built the dataset pipeline: 12 regions exported from Earth Engine, tiled into 576 image/mask pairs, split by region. |
| 2026-11-08 | Added dataset summary and preview images; shared the full dataset via Google Drive; set up the GitHub folder and README. |
| 2026-11-09 | **Progress 2:** dashboard UI review. |
| 2026-11-14 | Added two test regions, Minnedosa (prairie potholes, MB) and Winnipeg (urban rivers, MB): test set now 4 regions / 157 tiles; dataset now 14 regions / 665 tiles. Updated the export notebook (regions, previews for new tiles). |
| 2026-11-14 | Set the labelling rule: open surface water only (lakes, rivers, ponds, ocean); vegetated wetlands are not labelled as water. |
| 2026-11-14 | Made `water_layers.py` integration-ready: `show_water()` entry point, `build_water_controls()` filter buttons that work on a shared map, and `PROJECT_INFO`. |
| 2026-11-14 | Created `water_test_prep_and_baseline.ipynb`: band statistics from training tiles; NDWI baseline vs Dynamic World on the test set (IoU 0.546 overall; best Iqaluit 0.708, worst Minnedosa 0.203). |
| 2026-11-15 | Built the labelme correction pipeline (draft polygons → labelme → corrected masks) and verified the 40 tiles to be used. |
| 2026-11-16 | Corrected 40 test tiles in labelme (open water only). NDWI baseline vs corrected masks: IoU 0.609 pooled, 0.468 region average. |
| 2026-11-16 | Trained the U-Net (ResNet-34 encoder, 50 epochs; best validation IoU 0.964 vs Dynamic World labels). Test IoU vs corrected masks: 0.891 pooled, 0.840 region average, far above NDWI; largest gain on prairie ponds (Minnedosa 0.168 → 0.760). Below Dynamic World (0.929 / 0.905), which is expected since the model learns from Dynamic World labels and the corrected masks were drafted from them. Weakest region: Winnipeg (urban rivers, 0.783). |
| 2026-11-16 | Wrote `water_ai.py`: runs the trained U-Net on Sentinel-2 imagery for a drawn box, replacing the Dynamic World placeholder so detection in the tool is done by AI. |
| 2026-11-16 | AI detection now scales resolution with box size: full 10 m detail up to 200 km², coarser (up to 50 m) for boxes up to 5,000 km². A ~5,000 km² box takes about 18 seconds. |
| 2026-11-17 | Simplified the dashboard to the base map, two reference layers (all water, rivers) and AI box detection; removed permanent/seasonal water layers; reference layers labelled as non-AI. |
| 2026-11-17 | Raised AI detection limits after the parallel-download speed-up: boxes up to 20,000 km² (previously 5,000 km²). Full 10 m detail now applies up to 400 km² (previously 200 km²); larger areas run at a coarser resolution, up to 60 m (previously 50 m), reported in the chat. A large scan takes about 5 seconds on a T4 GPU. |
| 2026-11-23 | Dashboard: added Scan view (AI on the visible map area), grouped toolbar, and placeholder Only lakes / rivers / ponds buttons. Auto AI not yet working; Scan view covers it. |
| 2026-11-25 | Fixed classification of island-filled lakes (shape measured with islands filled; previously labelled as rivers). AI prediction now uses overlapping, blended tiles to remove tile-edge seams, and fills interior gaps that are spectrally water (NDWI > 0). Ponds shown in magenta. |
| 2026-11-25 | Auto AI now scans when a pan or zoom ends, and rescans when zooming in gives finer resolution. |

## Next steps

1. Classify AI-detected water into lakes, rivers and ponds (`water_postprocess.ipynb`), and show each type in its own colour.
2. Let users ask the assistant for a specific water type (e.g. "show only lakes").
3. Scan the visible map area with the AI (button, with optional automatic mode).
4. Improve narrow and urban river detection (weakest cases: Winnipeg, narrow boreal channels).
5. Validate results against NRCan's National Hydro Network.
6. Case studies, user manual, poster and maintenance plan for final delivery.

## References

- McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425–1432.
- Biggs, J., et al. (2005). 15 years of pond assessment in Britain. *Aquatic Conservation: Marine and Freshwater Ecosystems*, 15(6), 693–714.
