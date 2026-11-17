# AI detection performance tests (2026-11-17)

Scan time for `detect_water_ai` (dashboard AI box detection), measured in three stages:
**download** (Sentinel-2 pixels from Earth Engine), **model** (U-Net prediction) and **overlay** (drawing the result on the map).
Timings were measured with a temporary timing cell in the dashboard (not committed).

## 1. Baseline: sequential downloads (one chunk at a time)

| Runtime | Box (px) | Download | Model | Overlay |
|---|---|---|---|---|
| CPU | 61 × 89 | 0.9 s | 0.3 s | 0.0 s |
| CPU | 76 × 69 | 1.0 s | 0.3 s | 0.0 s |
| CPU | 150 × 132 | 1.1 s | 0.3 s | 0.0 s |
| CPU | 832 × 585 | 2.9 s | 4.4 s | 0.0 s |
| T4 GPU | 1476 × 1354 | 21.3 s | 1.8 s* | 0.2 s |
| T4 GPU | 1698 × 1177 | 28.5 s | 0.5 s | 0.2 s |

\* Includes one-time GPU warm-up.

**Findings:**
- Small boxes take ~1 s to download regardless of size: Earth Engine's fixed per-request overhead.
- On CPU the model is slow (~0.37 s per 256 × 256 tile); switching to a T4 GPU made it negligible.
- On GPU, downloading is ~95% of scan time, so download speed is the bottleneck.

## 2. Parallel downloads (chunk size 512 px)

| Workers | Box (px) | Download | Model | Overlay | Notes |
|---|---|---|---|---|---|
| 8 | 1665 × 1201 | 4.7 s | 0.7 s | 0.2 s | |
| 8 | (same area, repeat) | 6.7 s | — | — | Run-to-run variation |
| 10 | 1699 × 1176 | 6.7 s | 0.7 s | 0.3 s | |
| 12 | 1518 × 1316 | 12.5 s | 0.7 s | 0.3 s | Warning: "Connection pool is full" (pool size 10) |
| 12 | 1854 × 1078 | 4.5 s | 0.5 s | 0.3 s | |

A ~5,000 km² box (500,000 ha) downloaded in ~4 s with 8 workers.

**Findings:**
- 8 parallel workers cut download time ~5× (from 21–29 s to ~4.5–6.7 s).
- 10 and 12 workers gave no consistent improvement. 12 exceeded the Earth Engine client's 10-connection pool, causing reconnection delays (one run took 12.5 s).
- **Decision:** 8 workers (`WORKERS = 8` in `water_ai.py`).

## 3. Lighter composite test

Same box (1657 × 1206 px, run at 49 m resolution), comparing the full summer composite (median of all clear images, as used for training) with a lighter one (median of the 10 clearest images).

| Composite | Download | Model | Total | Water detected |
|---|---|---|---|---|
| Full (all clear images) | 4.4 s | 0.7 s | 5.3 s | 2,320.72 km² |
| Light (10 clearest) | 9.4 s | 0.4 s | 9.9 s | 2,366.36 km² |

Agreement between the two water masks: **IoU 0.960**.

**Findings:**
- Results were nearly identical, but the lighter composite was ~2× slower: sorting and limiting the image collection added work on Earth Engine's side.
- The full composite also ran first, so any caching would have favoured the lighter one; it was still slower.
- **Decision:** keep the full composite, matching the training data.

## Summary

| Setup | Typical large-box scan |
|---|---|
| Sequential downloads, T4 GPU | ~22–30 s |
| **Parallel downloads (8 workers), T4 GPU** | **~5–7 s** |

**Remaining bottleneck:** Earth Engine computing the composite for each chunk.
**Future improvements:** cache downloaded areas, download only the new edge when panning, pre-computed composites for chosen regions.
