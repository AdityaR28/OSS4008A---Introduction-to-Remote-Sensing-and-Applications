"""Server-side optical graph, with bounded metadata retrieval.

Primary API references (verified 2026-10-19):
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED
https://developers.google.com/earth-engine/guides/resample

B8 and RGB (10 m) are area-mean aggregated to each scene's native B12 (20 m)
grid *before* index calculation. All child pixels must be clear and have all
required input bands. SCL 4/5 only retains vegetation/bare land; water, dark
pixels, shadow, snow, cloud, unclassified and defective pixels are excluded.
The conservative land mask makes the coverage denominator the whole AOI,
including water, rather than silently treating unobserved water as valid land.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from .event import ANALYSIS_CRS, ANALYSIS_TRANSFORM, CLOUD_DATASET, S2_DATASET, Config, Window

LOGGER = logging.getLogger(__name__)
BANDS_10M = ["B4", "B3", "B2", "B8"]


@dataclass
class Analysis:
    aoi: object
    pre: object
    post: object
    pre_nbr: object
    post_nbr: object
    dnbr: object
    candidate: object
    provenance: dict


def _prepare_scene(ee, image, config):
    image = ee.Image(image)
    qa = ee.Image(image.get("cloud_score_image")).select("cs_cdf")
    b12_grid = image.select("B12").projection()
    scl = image.select("SCL")
    land = scl.eq(4).Or(scl.eq(5))
    input_mask = image.select(BANDS_10M).mask().reduce(ee.Reducer.min())
    clear10 = qa.gte(config.cloud_score_threshold).And(input_mask)
    # Missing CS+ samples and source edges are invalid, not a smaller averaging footprint.
    support20 = (
        clear10.unmask(0, sameFootprint=False)
        .reduceResolution(reducer=ee.Reducer.min(), maxPixels=64)
        .reproject(crs=b12_grid)
        .And(land)
        .And(image.select("B12").mask())
    )
    spectral10 = (
        image.select(BANDS_10M).multiply(0.0001)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
        .reproject(crs=b12_grid)
    )
    spectral = spectral10.addBands(image.select("B12").multiply(0.0001))
    return (
        spectral.updateMask(support20)
        .reproject(crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM)
        .copyProperties(image, ["system:time_start", "system:index"])
    )


def _collection(ee, aoi, window: Window, config: Config):
    raw = (
        ee.ImageCollection(S2_DATASET).filterBounds(aoi)
        .filterDate(window.start, window.end_exclusive)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.scene_cloud_limit))
    )
    raw_count = int(raw.size().getInfo())
    if raw_count == 0:
        raise ValueError(f"No Sentinel-2 scenes in {window.start} to {window.end_exclusive} (exclusive).")
    if raw_count > config.max_scenes_per_window:
        raise ValueError(f"{raw_count} scenes exceeds bounded limit {config.max_scenes_per_window}; narrow the window.")
    cloud = ee.ImageCollection(CLOUD_DATASET).filterBounds(aoi).filterDate(window.start, window.end_exclusive)
    joined = ee.ImageCollection(ee.Join.saveFirst("cloud_score_image").apply(
        primary=raw, secondary=cloud,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )).filter(ee.Filter.notNull(["cloud_score_image"])).sort("system:time_start")
    count = int(joined.size().getInfo())
    if not count:
        raise ValueError("No scenes have a matching Cloud Score+ image; refusing an unmasked fallback.")
    # One bounded metadata object, never image pixels or geometries via getInfo().
    metadata = ee.Dictionary({
        "indices": joined.aggregate_array("system:index"),
        "timestamps": joined.aggregate_array("system:time_start"),
    }).getInfo()
    indices, timestamps = metadata["indices"], metadata["timestamps"]
    if len(indices) != count or len(timestamps) != count:
        raise RuntimeError("Scene provenance count mismatch.")
    images = [{
        "id": f"{S2_DATASET}/{index}",
        "cloud_score_id": f"{CLOUD_DATASET}/{index}",
        "acquired_at": datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z"),
    } for index, ms in zip(indices, timestamps, strict=True)]
    prepared = joined.map(lambda image: _prepare_scene(ee, image, config))
    composite = (
        prepared.median().setDefaultProjection(ANALYSIS_CRS, ANALYSIS_TRANSFORM)
        .clip(aoi)
    )
    info = {
        "start": window.start, "end_exclusive": window.end_exclusive,
        "image_count": count, "scene_count_before_cloud_join": raw_count,
        "scenes_missing_cloud_score": raw_count - count,
        "acquisitions": sorted({image["acquired_at"][:10] for image in images}),
        "source_images": images,
        "image_count_definition": "Selected, spatially intersecting granules with Cloud Score+ matches; not unique dates.",
    }
    LOGGER.info("Window %s: %d scenes, %d unique dates (%d unmatched removed)", window.start, count, len(info["acquisitions"]), raw_count-count)
    return composite, info


def _nbr(image):
    nir, swir = image.select("B8"), image.select("B12")
    denominator = nir.add(swir)
    # Explicit arithmetic avoids normalizedDifference's hidden negative-input masking.
    return nir.subtract(swir).divide(denominator).updateMask(denominator.gt(1e-6)).rename("nbr")


def build_analysis(ee, config: Config) -> Analysis:
    aoi = ee.Geometry.Rectangle(list(config.aoi_bbox), proj="EPSG:4326", geodesic=False)
    pre, pre_info = _collection(ee, aoi, config.pre, config)
    post, post_info = _collection(ee, aoi, config.post, config)
    pre_nbr, post_nbr = _nbr(pre), _nbr(post)
    dnbr = pre_nbr.subtract(post_nbr).rename("dnbr")
    candidate = dnbr.gte(config.threshold).rename("candidate").toByte()
    return Analysis(aoi, pre, post, pre_nbr, post_nbr, dnbr, candidate, {"pre": pre_info, "post": post_info})
