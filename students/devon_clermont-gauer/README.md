# Project 11: Water Bodies of Canada (Lakes, Rivers, Ponds)

**Course:** OSS4008A – Introduction to Remote Sensing and Applications
**Student:** Devon Clermont-Gauer
**Instructor:** M. Hamed Mozaffari | **Research Assistant:** Aditya Rawal

An interactive map that identifies lakes, rivers and ponds across Canada from Sentinel-2 satellite imagery. A segmentation model (U-Net) classifies each pixel as water or not water; detected water bodies are then sorted into lakes, rivers and ponds by size and shape.

## Current status

| Part | Status |
|---|---|
| Dashboard UI (chat panel, map, layer buttons) | ✅ Built (Progress 2) |
| Training dataset (12 regions, 576 tiles) | ✅ Exported and tiled |
| Test-set corrections in labelme | ⏳ Next |
| NDWI baseline | ⏳ Next |
| U-Net training and evaluation | ⏳ Progress 3 |
| Lake / river / pond classification | ⏳ Final delivery |

## Folder structure

```
students/devon_clermont-gauer/
├── code/
│   ├── water_dashboard.ipynb          # interactive dashboard (run this to see the map)
│   ├── 01_water_dataset_export.ipynb  # builds the training dataset
│   └── water_layers.py                # reusable map-layer functions used by the dashboard
├── data/          # sample tiles
├── docs/          # project plan
└── results/       # screenshots and figures
```

## How to run

1. Open a notebook in Google Colab.
2. Replace `YOUR-PROJECT-ID` in the first code cell with your Google Earth Engine Cloud project ID.

- **Dashboard:** run all cells top to bottom (**Runtime → Run all**). Shows Canada with buttons for all water, permanent water, seasonal water and rivers. The chat panel is a placeholder until the AI model is connected (Progress 3).
- **Dataset export:** ⚠️ Running Cell 4 starts 12 Earth Engine exports to your Google Drive (several hours). If the exports already exist, run Cells 1–3, skip Cells 4–5, then run Cells 6–9. Cell 6 re-tiles the exports (slow, optional if tiles exist); Cell 9 can run on its own.

## Method

**Water detection rule:** water absorbs near-infrared (Sentinel-2 B8) and shortwave-infrared (B11) light, so it appears dark in those bands while land appears bright. The Normalized Difference Water Index, NDWI = (B3 − B8) / (B3 + B8), is positive over water (McFeeters, 1996). The U-Net learns this spectral pattern plus shape and context, which helps it reject look-alikes such as shadows, wet soil and asphalt.

**Classification rules (after detection):**
- **River:** long, narrow shape (low compactness = 4π × area / perimeter²)
- **Lake:** compact shape, area ≥ 2 ha
- **Pond:** compact shape, area < 2 ha (Biggs et al., 2005)

Depth is the traditional lake/pond distinction, but it can't be measured from satellite imagery, so surface area is used instead.

## Dataset

Sentinel-2 image tiles paired with water masks, 256×256 pixels at 10 m (2.56 km per side). Summer composites (June–September 2024) to avoid ice. Whole regions are assigned to one split so the model is always tested on places it never trained on.

| Split | Regions | Tiles |
|---|---|---|
| Train | algonquin, fraser_delta, lake_louise, mackenzie_delta, peace_athabasca, prairie_potholes, quebec_city, toronto | 396 |
| Validation | north_manitoba, ottawa | 112 |
| Test | iqaluit, kejimkujik | 68 |

**Full dataset (~334 MB):** [Google Drive folder]((https://drive.google.com/drive/folders/1QlQR3KWzcUz68h5cVU4suzOqYILWnGtP?usp=sharing))
Each tile is a GeoTIFF: `images/` hold Sentinel-2 bands B2, B3, B4, B8, B11 (blue, green, red, NIR, SWIR); `masks/` hold water masks (1 = water, 0 = not water). The dataset isn't committed to GitHub (course limit: files under 50 MB) and can be regenerated with `code/01_water_dataset_export.ipynb`.

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

## Next steps

1. Correct test tiles in labelme (~40 tiles).
2. Add prairie and urban test regions.
3. NDWI baseline (`02_water_ndwi_baseline.ipynb`).
4. Train and evaluate the U-Net (`03_water_unet_training.ipynb`) against NDWI and Dynamic World.
5. Connect the model to the dashboard (replaces the placeholder in `detect_in_region()`).

## References

- McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425–1432.
- Biggs, J., et al. (2005). 15 years of pond assessment in Britain. *Aquatic Conservation: Marine and Freshwater Ecosystems*, 15(6), 693–714.
