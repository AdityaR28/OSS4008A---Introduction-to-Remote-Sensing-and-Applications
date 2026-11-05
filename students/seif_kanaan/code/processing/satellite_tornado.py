"""Predeclared Sentinel-2 canopy-change experiment for Law Lake, Ontario.

Builds an authenticated Earth Engine graph; no image values/results are invented
locally. A pure NumPy audit checks the subsequently received numerical rasters.
This does not infer structural damage, EF rating, treefall percentage or wind speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import math

import numpy as np

S2_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_DATASET = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
FOREST_DATASET = "ESA/WorldCover/v200"
GRID_TRANSFORM = [20, 0, 0, 0, -20, 10000000]
REFLECTANCE_BANDS = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8", "swir2": "B12"}
NDVI_SENSITIVITY = (0.10, 0.15, 0.20)
NBR_SENSITIVITY = (0.05, 0.10, 0.15)
EPOCH = date(1970, 1, 1)
EVENT_SOURCE = "https://www.uwo.ca/ntp/blog/2025/ontarios_first_ef2_tornado_recorded_in_nw_on_on_jul_27.html"


@dataclass(frozen=True)
class TornadoConfig:
    bbox: tuple[float, float, float, float] = (-92.55, 48.765, -92.40, 48.825)
    crs: str = "EPSG:32615"
    scale_m: int = 20
    pre_start: str = "2025-06-15"
    pre_end_exclusive: str = "2025-07-27"
    post_start: str = "2025-07-28"
    post_end_exclusive: str = "2025-08-18"
    cloud_score_threshold: float = 0.65
    min_clear_dates: int = 2
    min_pre_ndvi: float = 0.60
    ndvi_drop_threshold: float = 0.15
    nbr_drop_threshold: float = 0.10
    min_coverage_fraction: float = 0.80
    max_scenes_per_window: int = 64

    def __post_init__(self):
        if self.crs != "EPSG:32615" or self.scale_m != 20:
            raise ValueError("This declared experiment uses the exact 20 m UTM 15N grid")
        west, south, east, north = self.bbox
        if not all(math.isfinite(x) for x in self.bbox) or not (-93 < west < east < -90 and 48 < south < north < 50):
            raise ValueError("AOI must be a finite bounded Ontario UTM 15N rectangle")
        if east - west > 0.20 or north - south > 0.10:
            raise ValueError("AOI exceeds bounded synchronous-export pilot size")
        for start, end, maximum in ((self.pre_start, self.pre_end_exclusive, 42), (self.post_start, self.post_end_exclusive, 21)):
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days
            if not 1 <= span <= maximum:
                raise ValueError("Use positive windows bounded to 42 baseline / 21 comparison days")
        # Entire event day is excluded: same-day Sentinel overpass precedes 23:10 UTC.
        if self.pre_end_exclusive > "2025-07-27" or self.post_start < "2025-07-28":
            raise ValueError("Pre/post windows must exclude the complete tornado day")
        if not ("2025-06-15" <= self.pre_start and self.post_end_exclusive <= "2025-08-25"):
            raise ValueError("This leaf-on experiment does not permit seasonal recovery windows")
        if not 0.6 <= self.cloud_score_threshold <= 1 or not 2 <= self.min_clear_dates <= 5:
            raise ValueError("Conservative Cloud Score+ and at least two distinct clear dates required")
        if not 0 < self.min_coverage_fraction <= 1 or not 1 <= self.max_scenes_per_window <= 64:
            raise ValueError("Invalid coverage/scene bounds")
        if (self.min_pre_ndvi, self.ndvi_drop_threshold, self.nbr_drop_threshold) != (0.60, 0.15, 0.10):
            raise ValueError("Primary exploratory thresholds are predeclared; use the full fixed sensitivity grid")
        if (self.pre_start, self.pre_end_exclusive, self.post_start, self.post_end_exclusive,
                self.min_clear_dates, self.cloud_score_threshold, self.min_coverage_fraction) != (
                "2025-06-15", "2025-07-27", "2025-07-28", "2025-08-18", 2, 0.65, 0.80):
            raise ValueError("Dates, quality criteria and gates are frozen for this declared experiment")


def sensitivity_id(ndvi: float, nbr: float) -> str:
    return f"candidate_ndvi{round(ndvi * 100):03d}_nbr{round(nbr * 100):03d}"


def _prepare_scene(ee, image, config: TornadoConfig):
    image = ee.Image(image)
    qa = ee.Image(image.get("cloud_score_image")).select("cs_cdf")
    native = image.select("B12").projection()
    bands10 = ["B2", "B3", "B4", "B8"]
    all_bands10 = image.select(bands10).mask().reduce(ee.Reducer.min())
    clear10 = qa.gte(config.cloud_score_threshold).And(all_bands10)
    clear20 = clear10.unmask(0, sameFootprint=False).reduceResolution(ee.Reducer.min(), maxPixels=64).reproject(crs=native)
    scl = image.select("SCL")
    land = scl.eq(4).Or(scl.eq(5))  # Keep bare land after disturbance; reject water/cloud/shadow/snow.
    spectrum10 = image.select(bands10).multiply(0.0001).reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(crs=native)
    spectrum = spectrum10.addBands(image.select("B12").multiply(0.0001))
    positive_denominators = spectrum.select("B8").add(spectrum.select("B4")).gt(1e-6).And(spectrum.select("B8").add(spectrum.select("B12")).gt(1e-6))
    physical_reflectance = spectrum.reduce(ee.Reducer.min()).gte(0).And(spectrum.reduce(ee.Reducer.max()).lte(1))
    support = clear20.And(land).And(image.select("B12").mask()).And(positive_denominators).And(physical_reflectance)
    spectrum = spectrum.updateMask(support).reproject(crs=config.crs, crsTransform=GRID_TRANSFORM)
    day = ee.Image.constant(ee.Date(image.get("system:time_start")).difference("1970-01-01", "day").floor()).rename("acquisition_day").toInt32().updateMask(spectrum.select("B8").mask()).reproject(crs=config.crs, crsTransform=GRID_TRANSFORM)
    return spectrum.addBands(day).copyProperties(image, ["system:index", "system:time_start"])


def _window(ee, aoi, start: str, end: str, config: TornadoConfig):
    raw = ee.ImageCollection(S2_DATASET).filterBounds(aoi).filterDate(start, end)
    count_raw = int(raw.size().getInfo())
    if not 1 <= count_raw <= config.max_scenes_per_window:
        raise ValueError(f"Sentinel-2 granule count {count_raw} outside bounded limits for {start}..{end}")
    cloud = ee.ImageCollection(CLOUD_DATASET).filterBounds(aoi).filterDate(start, end)
    joined = ee.ImageCollection(ee.Join.saveFirst("cloud_score_image").apply(primary=raw, secondary=cloud, condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))).filter(ee.Filter.notNull(["cloud_score_image"])).sort("system:index")
    info = ee.Dictionary({"indices": joined.aggregate_array("system:index"), "timestamps": joined.aggregate_array("system:time_start")}).getInfo()
    if (not isinstance(info, dict) or not isinstance(info.get("indices"), list) or not isinstance(info.get("timestamps"), list)
            or len(info["indices"]) != len(info["timestamps"]) or len(info["indices"]) > count_raw):
        raise ValueError("Sentinel source provenance lengths differ")
    if (len(set(info["indices"])) != len(info["indices"]) or
            any(not isinstance(index, str) or not index or "/" in index for index in info["indices"]) or
            any(isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) for ms in info["timestamps"])):
        raise ValueError("Invalid or repeated source acquisition metadata")
    acquired = [datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z") for ms in info["timestamps"]]
    if any(not start <= timestamp[:10] < end for timestamp in acquired):
        raise ValueError("Satellite catalogue returned an out-of-window acquisition")
    days = sorted({timestamp[:10] for timestamp in acquired})
    if len(days) < config.min_clear_dates:
        raise ValueError("Fewer matching catalogue dates than the per-pixel minimum; no unmasked fallback")
    prepared = joined.map(lambda image: _prepare_scene(ee, image, config))
    # One mosaic per UTC day prevents overlapping granules inflating clear-date count.
    day_images = {day: prepared.filterDate(day, (date.fromisoformat(day) + timedelta(days=1)).isoformat()).sort("system:index").mosaic().setDefaultProjection(config.crs, GRID_TRANSFORM).set("system:time_start", ee.Date(day).millis()) for day in days}
    daily = ee.ImageCollection.fromImages(list(day_images.values()))
    counts = daily.select("B8").count().setDefaultProjection(config.crs, GRID_TRANSFORM).rename("clear_count").unmask(0, sameFootprint=False).clip(aoi).toInt16()
    first = daily.select("acquisition_day").min().setDefaultProjection(config.crs, GRID_TRANSFORM).rename("first_day").clip(aoi)
    last = daily.select("acquisition_day").max().setDefaultProjection(config.crs, GRID_TRANSFORM).rename("last_day").clip(aoi)
    median = daily.select(list(REFLECTANCE_BANDS.values())).median().setDefaultProjection(config.crs, GRID_TRANSFORM).clip(aoi)
    valid = counts.gte(config.min_clear_dates).rename("valid").unmask(0, sameFootprint=False).clip(aoi).toByte()
    median = median.updateMask(valid)
    metadata = {
        "start": start, "end_exclusive": end, "raw_granule_count": count_raw,
        "cloud_matched_granule_count": len(acquired), "missing_cloud_score_granules": count_raw - len(acquired),
        "catalogue_dates": days, "catalogue_date_count": len(days),
        "source_images": [{"id": f"{S2_DATASET}/{index}", "cloud_score_id": f"{CLOUD_DATASET}/{index}", "acquired_at": timestamp} for index, timestamp in zip(info["indices"], acquired)],
        "composite": "Per-UTC-day mosaic, stable system:index tie order, then bandwise median of distinct daily observations",
        "pixel_dates": "first_day/last_day are earliest/latest contributing clear UTC days since 1970-01-01, not a fictitious single median acquisition date",
    }
    return median, valid, counts, first, last, metadata, day_images


def _index(image, first, second, label):
    a, b = image.select(first), image.select(second)
    denominator = a.add(b)
    return a.subtract(b).divide(denominator).updateMask(denominator.gt(1e-6)).rename(label).toFloat()


def _date_coverage(ee, forest, before, after, config):
    """One bounded metadata response, no candidate/reference-based selection.

    Uses the common center-defined domain/exact projected cells for consistency.
    Each value is an area after every source quality condition, before the
    two-distinct-date requirement; it cannot by itself identify which filter
    excluded an observation.
    """
    from .satellite_common import analysis_grid, grid_domain
    grid = analysis_grid(config.bbox, config.crs, config.scale_m)
    domain, region = grid_domain(ee, config.bbox, grid)
    transform = list(grid.transform)[:6]
    area = ee.Image.pixelArea().reproject(crs=grid.crs, crsTransform=transform).toFloat().updateMask(domain).updateMask(forest).rename("area_m2")
    def total(image):
        return image.reduceRegion(reducer=ee.Reducer.sum(), geometry=region, crs=grid.crs, crsTransform=transform, maxPixels=1000000, bestEffort=False, tileScale=4).get("area_m2")
    reductions = {"forest_area_m2": total(area)}
    keys = []
    for period, daily in (("pre", before), ("post", after)):
        for day, image in daily.items():
            key = period + "_" + day.replace("-", "_")
            reductions[key] = total(area.updateMask(image.select("B8").mask().gt(0)))
            keys.append((key, period, day))
    if len(keys) > 63:
        raise ValueError("Too many daily diagnostic reductions for fixed 42/21-day windows")
    try:
        returned = ee.Dictionary(reductions).getInfo()
    except Exception as error:
        from .export import safe_provider_error
        return {"status": "unavailable", "failure_message": safe_provider_error(error),
                "scope": "Optional per-date area diagnostic retrieval failed; source inventory and analytical date/count/support rasters remain authoritative. No coverage gate or pixel-quality fallback was applied."}
    if not isinstance(returned, dict) or set(returned) != set(reductions):
        raise ValueError("Daily coverage diagnostic metadata does not match declared acquisitions")
    forest_area = 0 if returned["forest_area_m2"] is None else returned["forest_area_m2"]
    if isinstance(forest_area, bool) or not isinstance(forest_area, (int, float)) or not math.isfinite(forest_area) or forest_area < 0:
        raise ValueError("Invalid forest population area in daily coverage diagnostics")
    rows = []
    for key, period, day in keys:
        value = 0 if returned[key] is None else returned[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > forest_area + max(1.0, forest_area * 1e-6):
            raise ValueError("Invalid daily quality-qualified forest area")
        rows.append({"period": period, "acquired_date": day, "quality_qualified_forest_area_m2": value, "fraction_of_mapped_forest": value / forest_area if forest_area else None})
    return {"status": "available", "forest_area_m2": forest_area, "dates": rows, "scope": "Complete source-quality support per distinct UTC date, before the two-clear-date requirement; diagnostic only, no hazard-result/window tuning", "area_convention": "Common anchored projected grid, full pixelArea for centers inside the declared AOI"}


def build(ee, config: TornadoConfig | None = None):
    from .satellite_common import SatelliteAnalysis
    config = config or TornadoConfig()
    aoi = ee.Geometry.Rectangle(list(config.bbox), proj="EPSG:4326", geodesic=False)
    pre, pre_valid, pre_count, pre_first, pre_last, pre_info, pre_daily = _window(ee, aoi, config.pre_start, config.pre_end_exclusive, config)
    post, post_valid, post_count, post_first, post_last, post_info, post_daily = _window(ee, aoi, config.post_start, config.post_end_exclusive, config)
    pre_ndvi, post_ndvi = _index(pre, "B8", "B4", "pre_ndvi"), _index(post, "B8", "B4", "post_ndvi")
    pre_nbr, post_nbr = _index(pre, "B8", "B12", "pre_nbr"), _index(post, "B8", "B12", "post_nbr")
    shared = pre_valid.And(post_valid).rename("shared_valid").unmask(0, sameFootprint=False).clip(aoi).toByte()
    # Classify the same float32 index/difference values exported for local audit.
    dndvi = pre_ndvi.subtract(post_ndvi).rename("dndvi").updateMask(shared).toFloat()
    dnbr = pre_nbr.subtract(post_nbr).rename("dnbr").updateMask(shared).toFloat()
    worldcover = ee.Image(ee.ImageCollection(FOREST_DATASET).first()).select("Map")
    forest = worldcover.eq(10).unmask(0, sameFootprint=False).reduceResolution(ee.Reducer.min(), maxPixels=64).reproject(crs=config.crs, crsTransform=GRID_TRANSFORM).rename("forest_mask").clip(aoi).toByte()
    original_days = {day: image for day, image in pre_daily.items() if "2025-07-06" <= day < "2025-07-27"}
    if original_days:
        original_count = ee.ImageCollection.fromImages(list(original_days.values())).select("B8").count().setDefaultProjection(config.crs, GRID_TRANSFORM)
    else:
        original_count = ee.Image.constant(0).setDefaultProjection(config.crs, GRID_TRANSFORM)
    original_count = original_count.unmask(0, sameFootprint=False).clip(aoi).rename("original_pre_clear_count").toInt16()
    original_valid = original_count.gte(config.min_clear_dates).rename("original_pre_valid").toByte()
    original_shared = original_valid.And(post_valid).rename("original_shared_valid").toByte()
    date_coverage = _date_coverage(ee, forest, pre_daily, post_daily, config)
    eligible = forest.And(pre_valid).And(pre_ndvi.gte(config.min_pre_ndvi).unmask(0, sameFootprint=False)).rename("eligible").unmask(0, sameFootprint=False).clip(aoi).toByte()
    candidate_support = eligible.And(shared)
    candidate = dndvi.gte(config.ndvi_drop_threshold).And(dnbr.gte(config.nbr_drop_threshold)).rename("candidate").updateMask(candidate_support).toByte()
    rasters = {"pre_ndvi": pre_ndvi, "post_ndvi": post_ndvi, "dndvi": dndvi,
               "pre_nbr": pre_nbr, "post_nbr": post_nbr, "dnbr": dnbr,
               "pre_valid": pre_valid, "post_valid": post_valid, "shared_valid": shared,
               "forest_mask": forest, "eligible": eligible, "candidate": candidate,
               "pre_clear_count": pre_count, "post_clear_count": post_count,
               "pre_first_day": pre_first, "pre_last_day": pre_last,
               "post_first_day": post_first, "post_last_day": post_last,
               "original_pre_clear_count": original_count, "original_pre_valid": original_valid,
               "original_shared_valid": original_shared}
    for period, image in (("pre", pre), ("post", post)):
        for label, band in REFLECTANCE_BANDS.items():
            rasters[f"{period}_{label}"] = image.select(band).rename(f"{period}_{label}")
    for ndvi in NDVI_SENSITIVITY:
        for nbr in NBR_SENSITIVITY:
            key = sensitivity_id(ndvi, nbr)
            rasters[key] = dndvi.gte(ndvi).And(dnbr.gte(nbr)).rename(key).updateMask(candidate_support).toByte()
    palettes = ["#2166ac", "#f7f7f7", "#b2182b"]
    rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0.02, "max": 0.30, "gamma": 1.1}
    displays = [
        {"id": "pre_rgb", "label": "Before: Sentinel-2 median reflectance", "image": pre, "vis": rgb_vis, "legend": [], "units": "surface reflectance"},
        {"id": "post_rgb", "label": "After: Sentinel-2 median reflectance", "image": post, "vis": rgb_vis, "legend": [], "units": "surface reflectance"},
        {"id": "dndvi", "label": "Observed NDVI decrease in eligible forest", "image": dndvi.updateMask(eligible), "vis": {"min": -0.3, "max": 0.3, "palette": palettes}, "legend": [{"color": palettes[0], "label": "NDVI increase"}, {"color": palettes[1], "label": "Little NDVI change"}, {"color": palettes[2], "label": "NDVI decrease"}], "units": "NDVI before minus after"},
        {"id": "dnbr", "label": "Observed NIR/SWIR index decrease in eligible forest", "image": dnbr.updateMask(eligible), "vis": {"min": -0.3, "max": 0.3, "palette": palettes}, "legend": [{"color": palettes[0], "label": "Index increase"}, {"color": palettes[1], "label": "Little index change"}, {"color": palettes[2], "label": "Index decrease (not burn severity)"}], "units": "NBR before minus after"},
        {"id": "candidate", "label": "Exploratory canopy spectral-change indicator", "image": candidate.selfMask(), "vis": {"min": 0, "max": 1, "palette": ["#ffffff", "#e6550d"]}, "legend": [{"color": "#e6550d", "label": "Both predeclared index-decrease criteria met"}], "units": "binary exploratory indicator", "allow_empty": True},
        {"id": "eligible", "label": "Baseline mapped-tree study population", "image": eligible.selfMask(), "vis": {"min": 0, "max": 1, "palette": ["#ffffff", "#238b45"]}, "legend": [{"color": "#238b45", "label": "2021 tree cover, observed baseline NDVI ≥ 0.60"}], "units": "baseline eligibility"},
        {"id": "shared_valid", "label": "Shared quality-qualified observation support", "image": shared.selfMask(), "vis": {"min": 0, "max": 1, "palette": ["#ffffff", "#3182bd"]}, "legend": [{"color": "#3182bd", "label": "At least two distinct clear dates in both windows"}], "units": "observation support"},
    ]
    event = {"id": "tornado-law-lake-2025-satellite-v2", "hazard": "tornado", "title": "Law Lake tornado: Sentinel-2 canopy change (revised baseline)", "event_date": "2025-07-27", "location": "Law Lake (Mine Centre), Ontario", "method": "Explicit v2 42-day baseline / unchanged 21-day comparison; distinct-date Sentinel-2 reflectance medians and predeclared NDVI/NBR decline in baseline mapped-tree cover", "sources": [
        {"label": "NTP event assessment and explicit Sentinel-2 observability evidence", "url": EVENT_SOURCE},
        {"label": "Sentinel-2 harmonized surface reflectance", "url": "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED"},
        {"label": "Cloud Score+ quality data", "url": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED"},
        {"label": "ESA WorldCover 2021 v200 baseline land cover", "url": "https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200"},
        {"label": "Kingfield and de Beurs (2017): tornado spectral response and land-cover/season limitations", "url": "https://doi.org/10.1175/JAMC-D-16-0228.1"},
    ], "limitations": [
        "This is satellite-observed spectral change near a documented tornado, not confirmed canopy-loss fraction, treefall, structural damage, wind speed or EF severity.",
        "NDVI/NBR thresholds are predeclared exploratory diagnostics, not scientifically calibrated damage classes. Report all nine sensitivity combinations; never tune to NTP agreement.",
        "2021 WorldCover tree cover can be stale or misclassified in 2025. The baseline NDVI criterion limits the study population, not all forest or all tornado-affected land.",
        "The revised June 15–July 26 42-day baseline is farther from the event and asymmetric with the 21-day comparison. Phenology/leaf development, moisture, illumination, logging, other wind, fire and cloud/shadow can cause change; this is a stronger temporal confound than the failed original 21-day baseline.",
        "Each 20 m cell mixes vegetation and surroundings. RGB/indices are medians of daily observations, not a simultaneous photograph; per-pixel first/last contributing dates and distinct clear-date counts are retained.",
        "The original July 6–26 baseline failed its unchanged 80% mapped-forest coverage requirement (35.959%). V2 is an explicitly documented one-time baseline revision; quality/count/coverage/indicator thresholds are unchanged, and no search or automatic relaxation occurs.",
        "Unobserved or ineligible land is not unchanged or undamaged. Two clear-date and two coverage gates still fail closed; original-window count/support rasters and per-date forest coverage are diagnostic, not evidence that V2 succeeds.",
        "NTP independently documents the event using satellite review including Sentinel imagery. It is contextual corroboration of observability, not independent pixel-level validation of this experiment.",
        "As checked 22 October 2026, NTP's current public Damage Tracks service does not contain a full Law Lake line. Do not synthesize a line or footprint from published start/end coordinates, maximum width or a map image.",
    ]}
    audit = {"method": "satellite_tornado.audit_arrays", "experiment_version": 2, "pre_start": config.pre_start, "pre_end_exclusive": config.pre_end_exclusive, "post_start": config.post_start, "post_end_exclusive": config.post_end_exclusive, "ndvi_formula": "(nir-red)/(nir+red)", "nbr_formula": "(nir-swir2)/(nir+swir2)", "change_sign": "before minus after", "min_clear_dates": config.min_clear_dates, "min_pre_ndvi": config.min_pre_ndvi, "candidate_thresholds": {"dndvi_gte": config.ndvi_drop_threshold, "dnbr_gte": config.nbr_drop_threshold}, "ndvi_sensitivity": list(NDVI_SENSITIVITY), "nbr_sensitivity": list(NBR_SENSITIVITY), "candidate_population": "eligible & shared_valid", "grid_transform": GRID_TRANSFORM, "min_coverage_fraction": config.min_coverage_fraction}
    provenance = {"experiment_version": 2, "revision": {"reason": "Original 21-day baseline had only 35.959124894% qualified mapped-forest coverage; preserve failure, revise baseline once to fixed 42-day window, retaining all quality/count/coverage/indicator thresholds", "prior_pre": {"start": "2025-07-06", "end_exclusive": "2025-07-27"}, "prior_source_archive_sha256": "9ce9d4c042f31427e12e58658fe1de5b496f9950f32582f06c27a29db0092d02", "result_used_for_revision": "Observation coverage only, not candidate area or reference agreement"}, "pre": pre_info, "post": post_info, "daily_forest_coverage": date_coverage, "dataset": S2_DATASET, "cloud_dataset": CLOUD_DATASET, "cloud_score_band": "cs_cdf", "cloud_score_gte": config.cloud_score_threshold, "scl_classes": [4, 5], "reflectance_range_inclusive": [0, 1], "crs_transform": GRID_TRANSFORM, "forest_source": {"dataset": FOREST_DATASET, "year": 2021, "class": 10, "aggregation": "All contributing WorldCover source pixels must map to tree cover on each 20 m target cell"}, "coverage_gate": {"raster": "eligible", "min_shared_fraction": config.min_coverage_fraction}, "coverage_gates": [{"population": "forest_mask", "valid": "pre_valid", "min_fraction": config.min_coverage_fraction}, {"population": "eligible", "valid": "shared_valid", "min_fraction": config.min_coverage_fraction}], "window_selection": "Explicit V2 recipe fixed before its new computation; no automatic date search, reference/candidate-based tuning or subsequent threshold relaxation"}
    return SatelliteAnalysis(event=event, aoi=aoi, bbox=config.bbox, crs=config.crs, scale_m=config.scale_m, rasters=rasters, displays=displays, provenance=provenance, pre_valid_id="pre_valid", post_valid_id="post_valid", shared_valid_id="shared_valid", candidate_id="candidate", audit=audit)


def audit_arrays(arrays: dict[str, np.ndarray], nodata: float = -9999) -> dict:
    """Independently check received TIFF arrays; synthetic local tests are not results.

    All inputs use the common exact grid. `domain` and `pixel_area_m2` are added by
    the common exporter. Areas are summed from the received pixel-area raster,
    never inferred from the visual PNG or a NTP nominal width.
    """
    required = {"domain", "pixel_area_m2", "pre_valid", "post_valid", "shared_valid", "forest_mask", "eligible", "candidate", "pre_ndvi", "post_ndvi", "dndvi", "pre_nbr", "post_nbr", "dnbr", "original_pre_clear_count", "original_pre_valid", "original_shared_valid"}
    required |= {f"{p}_{band}" for p in ("pre", "post") for band in REFLECTANCE_BANDS}
    required |= {f"{p}_{key}" for p in ("pre", "post") for key in ("clear_count", "first_day", "last_day")}
    required |= {sensitivity_id(a, b) for a in NDVI_SENSITIVITY for b in NBR_SENSITIVITY}
    if not required.issubset(arrays):
        raise ValueError(f"Missing tornado numerical audit rasters: {sorted(required - set(arrays))}")
    values = {name: np.asarray(arrays[name], dtype=np.float64) for name in required}
    shape = values["domain"].shape
    if len(shape) != 2 or any(a.shape != shape for a in values.values()):
        raise ValueError("All audit rasters must be aligned two-dimensional arrays")
    domain = values["domain"] == 1
    if not domain.any() or not np.isin(values["domain"], [0, 1, nodata]).all():
        raise ValueError("Invalid or empty analysis domain")
    known = lambda a: np.isfinite(a) & (a != nodata)
    for name, a in values.items():
        if not np.isfinite(a[domain]).all():
            raise ValueError(f"Nonfinite audit input: {name}")
    for flag in ("pre_valid", "post_valid", "shared_valid", "forest_mask", "eligible", "original_pre_valid", "original_shared_valid"):
        if not np.isin(values[flag][domain], [0, 1]).all():
            raise ValueError(f"Invalid binary population/support flag: {flag}")
    area = values["pixel_area_m2"]
    if not (known(area)[domain].all() and (area[domain] > 0).all()):
        raise ValueError("Positive finite actual pixel areas required")
    masks = {}
    for period, start, end in (("pre", "2025-06-15", "2025-07-27"), ("post", "2025-07-28", "2025-08-18")):
        count = values[f"{period}_clear_count"]
        maximum = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if not ((count[domain] >= 0).all() and (count[domain] <= maximum).all() and (count[domain] == np.floor(count[domain])).all()):
            raise ValueError("Invalid distinct-date count")
        valid = (count >= 2) & domain
        if not np.array_equal(values[f"{period}_valid"][domain] == 1, valid[domain]):
            raise ValueError("Clear-date count/validity mismatch")
        masks[period] = valid
        observed = (count > 0) & domain
        first, last = values[f"{period}_first_day"], values[f"{period}_last_day"]
        for a in (first, last):
            if not np.array_equal(known(a)[domain], observed[domain]):
                raise ValueError("Per-pixel contributing dates must exist exactly where at least one date is observed")
        lo, hi = (date.fromisoformat(start) - EPOCH).days, (date.fromisoformat(end) - EPOCH).days
        if not ((first[observed] >= lo).all() and (last[observed] < hi).all() and (first[observed] <= last[observed]).all() and (first[observed] == np.floor(first[observed])).all() and (last[observed] == np.floor(last[observed])).all()):
            raise ValueError("Contributing dates fall outside fixed window or have invalid order")
        if not (count[observed] <= last[observed] - first[observed] + 1).all():
            raise ValueError("Clear-date counts exceed distinct calendar days in their contributing interval")
        for band in REFLECTANCE_BANDS:
            if not np.array_equal(known(values[f"{period}_{band}"])[domain], valid[domain]):
                raise ValueError("Reflectance support does not equal qualified observation support")
            if not ((values[f"{period}_{band}"][valid] >= 0).all() and (values[f"{period}_{band}"][valid] <= 1).all()):
                raise ValueError("Reflectance outside declared physical quality range")
        nir, red, swir = (values[f"{period}_{name}"] for name in ("nir", "red", "swir2"))
        for key, other in (("ndvi", red), ("nbr", swir)):
            if not ((nir + other)[valid] > 1e-6).all():
                raise ValueError("Invalid normalized-index denominator")
            expected = (nir[valid] - other[valid]) / (nir[valid] + other[valid])
            actual = values[f"{period}_{key}"]
            if not np.array_equal(known(actual)[domain], valid[domain]) or not np.allclose(actual[valid], expected, atol=2e-6, rtol=1e-6):
                raise ValueError(f"{period}_{key} formula/support mismatch")
    shared = masks["pre"] & masks["post"]
    if not np.array_equal(values["shared_valid"][domain] == 1, shared[domain]):
        raise ValueError("Shared observation support mismatch")
    original_count = values["original_pre_clear_count"]
    if not ((original_count[domain] >= 0).all() and (original_count[domain] <= 21).all() and (original_count[domain] == np.floor(original_count[domain])).all() and (original_count[domain] <= values["pre_clear_count"][domain]).all()):
        raise ValueError("Invalid original-window distinct-date diagnostic count")
    original_valid = (original_count >= 2) & domain
    if not np.array_equal(values["original_pre_valid"][domain] == 1, original_valid[domain]) or not np.array_equal(values["original_shared_valid"][domain] == 1, (original_valid & masks["post"])[domain]):
        raise ValueError("Original-window diagnostic support mismatch")
    for key in ("ndvi", "nbr"):
        actual = values[f"d{key}"]
        expected = values[f"pre_{key}"] - values[f"post_{key}"]
        if not np.array_equal(known(actual)[domain], shared[domain]) or not np.allclose(actual[shared], expected[shared], atol=2e-6, rtol=1e-6):
            raise ValueError(f"d{key} sign/formula/support mismatch")
    forest = (values["forest_mask"] == 1) & domain
    eligible = forest & masks["pre"] & (values["pre_ndvi"] >= 0.60)
    if not np.array_equal(values["eligible"][domain] == 1, eligible[domain]):
        raise ValueError("Baseline forest eligibility mismatch")
    population = eligible & shared
    sum_area = lambda mask: float(area[mask & domain].sum())
    forest_area, eligible_area = sum_area(forest), sum_area(eligible)
    if forest_area <= 0 or eligible_area <= 0:
        raise ValueError("No known baseline mapped-tree/eligible population; fail closed")
    pre_coverage = sum_area(forest & masks["pre"]) / forest_area
    shared_coverage = sum_area(population) / eligible_area
    if min(pre_coverage, shared_coverage) < 0.80 - 1e-9:
        raise ValueError("Declared baseline/shared forest coverage gate failed")
    sensitivity = []
    for ndvi in NDVI_SENSITIVITY:
        for nbr in NBR_SENSITIVITY:
            name = sensitivity_id(ndvi, nbr)
            expected = (values["dndvi"] >= ndvi) & (values["dnbr"] >= nbr)
            actual = values[name]
            if not np.array_equal(known(actual)[domain], population[domain]) or not np.isin(actual[population], [0, 1]).all() or not np.array_equal(actual[population] == 1, expected[population]):
                raise ValueError(f"Candidate sensitivity formula/support mismatch: {name}")
            sensitivity.append({"dndvi_gte": ndvi, "dnbr_gte": nbr, "candidate_area_ha": sum_area(population & expected) / 10000})
    candidate = values["candidate"]
    primary = values[sensitivity_id(0.15, 0.10)]
    if not np.array_equal(known(candidate)[domain], population[domain]) or not np.array_equal(candidate[population], primary[population]):
        raise ValueError("Primary candidate differs from predeclared middle sensitivity case")
    return {"status": "passed", "scope": "Numerical TIFF arithmetic, masks, distinct-date bounds, declared coverage gates and all nine exploratory thresholds; not damage accuracy", "domain_area_ha": sum_area(domain) / 10000, "forest_area_ha": forest_area / 10000, "eligible_area_ha": eligible_area / 10000, "baseline_forest_coverage": pre_coverage, "shared_eligible_coverage": shared_coverage, "candidate_area_ha": sum_area(population & (candidate == 1)) / 10000, "sensitivity": sensitivity}
