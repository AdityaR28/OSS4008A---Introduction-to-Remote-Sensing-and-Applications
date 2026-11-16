# Project 11: Water Bodies of Canada (Lakes, Rivers, Ponds)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdityaR28/OSS4008A---Introduction-to-Remote-Sensing-and-Applications/blob/main/students/YOUR-FOLDER/code/Dashboard.ipynb)

**Course:** OSS4008A – Introduction to Remote Sensing and Applications
**Student:** Devon Clermont-Gauer
**Instructor:** M. Hamed Mozaffari | **Research Assistant:** Aditya Rawal

An interactive map that identifies lakes, rivers and ponds across Canada from Sentinel-2 satellite imagery. A segmentation model (U-Net) classifies each pixel as water or not water; detected water bodies are then sorted into lakes, rivers and ponds by size and shape.

## Current status

| Part | Status |
|---|---|
| Dashboard UI (chat panel, map, layer buttons, draw-a-box) | Built (Progress 2) |
| Integration-ready module (`water_layers.py`) | Shared-UI entry point and filter controls |
| Training dataset (14 regions, 665 tiles) | Exported and tiled |
| Test-set corrections in labelme (40 tiles) | Done |
| NDWI baseline | IoU 0.609 pooled, 0.468 averaged across regions |
| U-Net training and evaluation | IoU 0.891 pooled, 0.840 averaged across regions |
| AI detection in the dashboard (`water_ai.py`) | Written; testing |
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
└── results/       # screenshots and result figures
    ├── dashboard.png                          # dashboard screenshot
    ├── ndwi_baseline_vs_dynamic_world.png     # NDWI scores vs Dynamic World masks
    ├── NDWI baseline vs 40 labelme corrected masks.png         # NDWI scores vs hand-corrected masks
    ├── Training curves.png                    # U-Net loss and validation IoU per epoch
    ├── Example predictions.png                # test tiles: image, corrected mask, U-Net prediction
    └── Unet vs baselines.png                  # test results: NDWI vs Dynamic World vs U-Net
```

## How to run

**Dashboard:** click the **Open in Colab** badge above, replace `YOUR-PROJECT-ID` in Cell 1 with your Google Earth Engine Cloud project ID, then **Runtime → Run all**. The notebook automatically downloads `water_layers.py`, `water_ai.py` and `band_stats.json` from this repo, and the trained model from Google Drive, so nothing needs to be set up by hand. Draw a box on the map and click **AI: water in drawn box** to run the U-Net (boxes up to 200 km²). A T4 GPU runtime makes detection faster, but CPU works.

**Other notebooks** (open them in Colab from the `code/` folder):
- **water_dataset_export:** ⚠️ Cell 4 starts 14 Earth Engine exports to your Google Drive (several hours). If the exports already exist, run Cells 1–3, skip Cells 4–5, then run Cells 6–9.
- **test_prep_and_baseline:** run Cells 1, 2 and 5. ⚠️ Don't re-run Cell 3 after corrections have started: it overwrites the corrected labelme files. Cell 4 converts corrected labelme files into masks.
- **unet_training:** set **Runtime → Change runtime type → T4 GPU**, then run all cells.

## Method

**Water detection rule:** water absorbs near-infrared (Sentinel-2 B8) and shortwave-infrared (B11) light, so it appears dark in those bands while land appears bright. The Normalized Difference Water Index, NDWI = (B3 − B8) / (B3 + B8), is positive over water (McFeeters, 1996). The U-Net learns this spectral pattern plus shape and context, which helps it reject look-alikes such as shadows, wet soil and asphalt.

**Classification rules (after detection):**
- **River:** long, narrow shape (low compactness = 4π × area / perimeter²)
- **Lake:** compact shape, area ≥ 2 ha
- **Pond:** compact shape, area < 2 ha (Biggs et al., 2005)

Depth is the traditional lake/pond distinction, but it can't be measured from satellite imagery, so surface area is used instead.

## Dataset

**Trained model:** `unet_resnet34_best.pth` (~100 MB) is shared on [Google Drive](PASTE-MODEL-LINK-HERE) because it exceeds the course's 50 MB file limit. The dashboard downloads it automatically.

Sentinel-2 image tiles paired with water masks, 256×256 pixels at 10 m (2.56 km per side). Summer composites (June–September 2024) to avoid ice. Whole regions are assigned to one split so the model is always tested on places it never trained on.

**Labelling rule:** water = visible open surface water (lakes, rivers, ponds, ocean). Vegetated wetlands and reed-filled potholes are not labelled as water; seasonal wetlands are shown separately by the dashboard's seasonal water layer. Ocean is labelled as water and is excluded later when classifying lakes.

| Split | Regions | Tiles |
|---|---|---|
| Train | algonquin, fraser_delta, lake_louise, mackenzie_delta, peace_athabasca, prairie_potholes, quebec_city, toronto | 396 |
| Validation | north_manitoba, ottawa | 112 |
| Test | iqaluit, kejimkujik, minnedosa, winnipeg | 157 |

**Full dataset (~372 MB):** [Google Drive folder]((https://drive.google.com/drive/folders/1QlQR3KWzcUz68h5cVU4suzOqYILWnGtP?usp=sharing))
Each tile is a GeoTIFF: `images/` hold Sentinel-2 bands B2, B3, B4, B8, B11 (blue, green, red, NIR, SWIR); `masks/` hold water masks (1 = water, 0 = not water). The dataset isn't committed to GitHub .

## Data sources

| Dataset | Use | Licence |
|---|---|---|
| Sentinel-2 SR Harmonized (Copernicus/ESA) | Satellite imagery | Copernicus open data; attribution required |
| Dynamic World V1 (Google / WRI) | Training labels (water masks) | CC BY 4.0 |
| JRC Global Surface Water v1.4 | Dashboard water layers | Free use with attribution |
| HydroSHEDS Free Flowing Rivers (WWF) | Reference river network | Free use with attribution |
| LSIB Simple 2017 (US Department of State) | Canada boundary | Public domain |
| National Hydro Network (NRCan) | Planned: validating results against official Canadian data | Open Government Licence – Canada |

Global datasets are used for imagery and labels because no Canadian source offers free 10 m coverage nationwide; Canada's National Hydro Network is used for validation.

## Known limitations

- Water features smaller than ~10 m (narrow streams, very small ponds) are below the image resolution.
- Water under tree canopy, very shallow or turbid water may be missed.
- Persistent cloud reduces usable imagery in some northern regions.
- Training labels come from Dynamic World (itself a model output) and contain some errors; the test set will be corrected by hand in labelme.
- Some regions (Iqaluit, Fraser Delta, Toronto) include ocean or Great Lake water, which must be excluded when classifying lakes.
- The test set is currently small (68 tiles, 2 regions); prairie and urban test regions will be added.

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

## Next steps

1. Test AI detection in the dashboard and add an example screenshot.
2. Improve urban river detection (weakest region: Winnipeg), e.g. by adding more urban/river training regions.
3. Classify detected water into lakes, rivers and ponds (`water_postprocess.ipynb`).
4. Validate results against NRCan's National Hydro Network.
5. Case studies, user manual, poster and maintenance plan for final delivery.

## References

- McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425–1432.
- Biggs, J., et al. (2005). 15 years of pond assessment in Britain. *Aquatic Conservation: Marine and Freshwater Ecosystems*, 15(6), 693–714.
