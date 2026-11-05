"""Authenticated Sentinel-1 candidate surface-inundation graph for Lower Fraser.

This module constructs real Earth Engine expressions and retrieves bounded
catalogue/coverage metadata. It does not authenticate, publish, or create any
observed result offline. Thresholds are declared exploratory before seeing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from typing import Any, Mapping

S1_DATASET = "COPERNICUS/S1_GRD"
WATER_DATASET = "JRC/GSW1_4/YearlyHistory"
LANDCOVER_DATASET = "ESA/WorldCover/v100"
LANDCOVER_CLASSES = (10,20,30,40,50,60,70,80,90,95,100)
DEM_DATASET = "USGS/SRTMGL1_003"
DEFAULTS = {
    "aoi_bbox": (-122.4, 49.0, -121.9, 49.25),
    "analysis_crs": "EPSG:32610", "analysis_transform": (20, 0, 0, 0, -20, 10_000_000),
    "pre_start": "2021-10-15", "pre_end_exclusive": "2021-11-13",
    "post_start": "2021-11-16", "post_end_exclusive": "2021-11-22",
    "post_vv_max_db": -16.0, "decrease_min_db": 3.0,
    "slope_max_degrees": 5.0, "smoothing_radius_m": 30,
    "incidence_min_degrees": 30.0, "incidence_max_degrees": 45.0,
    "permanent_water_year": 2020, "min_pre_observations": 2, "min_post_observations": 1,
    "min_valid_fraction": 0.80, "min_baseline_fraction": 0.80,
    "max_scenes_per_window": 128, "max_orbit_candidates": 16,
}
SOURCES = [
    {"label": "Copernicus Sentinel-1 GRD; Earth Engine catalogue", "url": "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD", "attribution": "Contains modified Copernicus Sentinel data (2021), processed by Google Earth Engine", "license": "Copernicus Sentinel data terms"},
    {"label": "Earth Engine Sentinel-1 preprocessing and filtering", "url": "https://developers.google.com/earth-engine/guides/sentinel1", "attribution": "Google Earth Engine", "license": "Documentation"},
    {"label": "JRC Yearly Water Classification History v1.4", "url": "https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_YearlyHistory", "attribution": "Source: EC JRC/Google; Pekel et al. (2016), Nature 540, 418–422, doi:10.1038/nature20584", "license": "Copernicus Programme; acknowledgement required"},
    {"label": "ESA WorldCover 2020 v100 global land cover", "url": "https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100", "attribution": "© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium; Zanaga et al. (2021), doi:10.5281/zenodo.5571936", "license": "CC BY 4.0"},
    {"label": "ESA WorldCover product documentation and validation", "url": "https://esa-worldcover.org/en/data-access", "attribution": "ESA WorldCover Consortium", "license": "Documentation; global validation does not establish local flood accuracy"},
    {"label": "NASA SAR scattering mechanisms", "url": "https://science.nasa.gov/mission/nisar/get-to-know-sar/", "attribution": "NASA; strong point and corner-reflector returns can be bright", "license": "Method reference"},
    {"label": "NASA SRTM 30 m elevation v3", "url": "https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003", "attribution": "NASA/USGS/JPL-Caltech; Farr et al. (2007), doi:10.1029/2005RG000183", "license": "NASA/JPL data terms"},
    {"label": "UN-SPIDER SAR flood-mapping recommended practice", "url": "https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/in-detail", "attribution": "UNOOSA/UN-SPIDER", "license": "Method reference; this implementation is an independent exploratory adaptation"},
    {"label": "2021 atmospheric-river event context", "url": "https://www.publicsafety.gc.ca/cnt/trnsprnc/brfng-mtrls/prlmntry-bndrs/20211207/02-en.aspx?wbdisable=true", "attribution": "Public Safety Canada; atmospheric-river rainfall began 14 November 2021", "license": "Government of Canada information terms"},
]
LIMITATIONS = [
    "Exploratory candidate open-water inundation, not independently validated flood extent, water depth, damage, exposure, or an emergency warning.",
    "The −16 dB post-event and 3 dB decrease thresholds are predeclared exploratory choices, not locally calibrated universal water thresholds or an accuracy guarantee.",
    "Same relative orbit and pass reduce viewing-geometry differences. Earth Engine GRD is terrain-corrected, but is not radiometrically terrain-flattened; a <5° slope filter is not a layover/shadow correction.",
    "Wet bare soil, smooth fields, land management, freeze/thaw, wind, SAR noise and changing incidence geometry can mimic or hide inundation. Urban or vegetation-covered flooding can increase backscatter or remain undetected.",
    "Temporal medians and 30 m-radius spatial smoothing suppress speckle and also erase narrow, short-lived or heterogeneous flood signals. The 20 m export grid is not a claim of 20 m independent resolving power.",
    "The post window is an event-period composite, not a simultaneous or peak flood snapshot. The pre window may already contain wet or flooded locations; no change is not proof of no flooding.",
    "Baseline support comes from observed, documented ESA WorldCover 2020 land-cover classes, not from filling JRC no-data as dry land. Any contributing WorldCover permanent-water class 80, plus observed JRC2020 class 3, is excluded. Both baseline products can misclassify boundaries and miss changes before the event; seasonal water remains eligible.",
    "WorldCover class 80 is aggregated conservatively: any native permanent-water overlap excludes the full 20 m analysis cell, widening water exclusions along boundaries. WorldCover2020 is a land-cover model, not independent field truth or proof of dry conditions in November 2021.",
    "No-data, missing baseline/DEM, high slope, insufficient observations and permanent water are excluded from candidate evaluation; blank map pixels are not confirmed dry land.",
    "The fixed bounding box covers a Lower Fraser case study. Results must not be extrapolated to the whole British Columbia event. No SAR coherence or earthquake/tornado damage inference is performed by this module.",
]


@dataclass
class FloodConfig:
    aoi_bbox: tuple[float,float,float,float] = (-122.4,49.0,-121.9,49.25)
    analysis_crs: str = "EPSG:32610"
    analysis_transform: tuple[int,...] = (20,0,0,0,-20,10_000_000)
    pre_start: str = "2021-10-15"
    pre_end_exclusive: str = "2021-11-13"
    post_start: str = "2021-11-16"
    post_end_exclusive: str = "2021-11-22"
    post_vv_max_db: float = -16.0
    decrease_min_db: float = 3.0
    slope_max_degrees: float = 5.0
    smoothing_radius_m: int = 30
    incidence_min_degrees: float = 30.0
    incidence_max_degrees: float = 45.0
    permanent_water_year: int = 2020
    min_pre_observations: int = 2
    min_post_observations: int = 1
    min_valid_fraction: float = 0.80
    min_baseline_fraction: float = 0.80
    max_scenes_per_window: int = 128
    max_orbit_candidates: int = 16


def _get(config: Any, key: str, default: Any) -> Any:
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def parameters(config: Any) -> dict[str, Any]:
    """Accept a shared notebook config object/mapping; ignore unrelated fields."""
    params = {key: _get(config, key, value) for key, value in DEFAULTS.items()}
    for period in ("pre", "post"):
        window = _get(config, period, None)
        if window is not None:
            params[f"{period}_start"] = _get(window, "start", params[f"{period}_start"])
            params[f"{period}_end_exclusive"] = _get(window, "end_exclusive", params[f"{period}_end_exclusive"])
        start, end = date.fromisoformat(params[f"{period}_start"]), date.fromisoformat(params[f"{period}_end_exclusive"])
        if not 0 < (end - start).days <= 45:
            raise ValueError("Flood observation windows must be positive and at most 45 days")
    if params["pre_end_exclusive"] > params["post_start"]:
        raise ValueError("Flood pre/post windows must not overlap")
    bounds = list(params["aoi_bbox"])
    if len(bounds) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds):
        raise ValueError("Flood AOI must contain four finite coordinates")
    west, south, east, north = bounds
    if not (-123 <= west < east <= -120 and 48.9 <= south < north <= 50) or east-west > 0.6 or north-south > 0.5:
        raise ValueError("Flood pilot AOI must remain bounded in Lower Fraser / UTM zone 10N")
    if params["analysis_crs"] != "EPSG:32610" or list(params["analysis_transform"]) != [20, 0, 0, 0, -20, 10_000_000]:
        raise ValueError("Flood pilot requires the declared fixed EPSG:32610 20 m grid")
    for key in ("post_vv_max_db", "decrease_min_db", "slope_max_degrees", "incidence_min_degrees", "incidence_max_degrees", "min_valid_fraction", "min_baseline_fraction"):
        if isinstance(params[key], bool) or not isinstance(params[key], (int, float)) or not math.isfinite(params[key]):
            raise ValueError(f"Nonfinite/non-numeric flood parameter: {key}")
    if not (-30 <= params["post_vv_max_db"] <= -8 and 0 < params["decrease_min_db"] <= 10 and 0 < params["slope_max_degrees"] <= 10):
        raise ValueError("Exploratory backscatter/slope thresholds outside bounded supported range")
    if not 20 <= params["incidence_min_degrees"] < params["incidence_max_degrees"] <= 50 or not 0 < params["min_valid_fraction"] <= 1 or not 0 < params["min_baseline_fraction"] <= 1:
        raise ValueError("Invalid flood incidence/coverage gate")
    for key, low, high in (("smoothing_radius_m",10,50),("min_pre_observations",2,8),("min_post_observations",1,8),
                           ("max_scenes_per_window",1,256),("max_orbit_candidates",1,16)):
        if isinstance(params[key], bool) or not isinstance(params[key], int) or not low <= params[key] <= high:
            raise ValueError(f"Invalid bounded integer flood parameter: {key}")
    if params["permanent_water_year"] != 2020:
        raise ValueError("Flood pilot baseline must use pre-event JRC year 2020")
    params["aoi_bbox"] = bounds
    params["analysis_transform"] = list(params["analysis_transform"])
    return params


def _on_grid(image: Any, params: dict) -> Any:
    return image.reproject(crs=params["analysis_crs"], crsTransform=params["analysis_transform"])


def _catalogue(ee: Any, aoi: Any, params: dict, period: str) -> list[dict]:
    collection = (ee.ImageCollection(S1_DATASET).filterBounds(aoi)
                  .filterDate(params[f"{period}_start"], params[f"{period}_end_exclusive"])
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.eq("resolution_meters", 10))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .sort("system:index"))
    count = collection.size().getInfo()
    if isinstance(count, bool) or not isinstance(count, (int,float)) or not 0 < count <= params["max_scenes_per_window"] or int(count) != count:
        raise ValueError(f"No usable or too many Sentinel-1 scenes in {period} window: {count!r}")
    fields = ["system:index", "system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start", "orbitNumber_start", "platform_number"]
    result = ee.Dictionary({field: collection.aggregate_array(field) for field in fields}).getInfo()
    if set(result) != set(fields) or any(not isinstance(result[field],list) or len(result[field]) != count for field in fields):
        raise ValueError("Sentinel-1 bounded catalogue provenance is incomplete")
    rows = []
    for values in zip(*(result[field] for field in fields), strict=True):
        index, timestamp, direction, relative, absolute, platform = values
        if not isinstance(index,str) or not index or "/" in index or direction not in {"ASCENDING","DESCENDING"} or platform not in {"A","B","C","D"}:
            raise ValueError("Invalid Sentinel-1 identity/orbit metadata")
        for value in (timestamp,relative,absolute):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or int(value) != value:
                raise ValueError("Missing/invalid Sentinel-1 numeric orbit/time metadata")
        if not 1 <= relative <= 175 or absolute < 1:
            raise ValueError("Sentinel-1 orbit number outside valid range")
        acquired = datetime.fromtimestamp(timestamp/1000,timezone.utc).isoformat().replace("+00:00","Z")
        if not params[f"{period}_start"] <= acquired[:10] < params[f"{period}_end_exclusive"]:
            raise ValueError("Sentinel-1 catalogue returned an out-of-window scene")
        rows.append({"id":f"{S1_DATASET}/{index}","index":index,"timestamp":int(timestamp),"acquired_at":acquired,
                     "pass":direction,"relative_orbit":int(relative),"absolute_orbit":int(absolute),"platform":platform})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate Sentinel-1 catalogue scene IDs")
    return sorted(rows,key=lambda row:(row["timestamp"],row["id"]))


def track_candidates(pre_rows: list[dict], post_rows: list[dict], params: dict) -> list[dict]:
    """Group shared tracks from metadata, without consulting a change image."""
    pre_tracks = {(row["pass"],row["relative_orbit"]) for row in pre_rows}
    post_tracks = {(row["pass"],row["relative_orbit"]) for row in post_rows}
    candidates = []
    for direction, relative in sorted(pre_tracks & post_tracks):
        before = [row for row in pre_rows if (row["pass"],row["relative_orbit"]) == (direction,relative)]
        after = [row for row in post_rows if (row["pass"],row["relative_orbit"]) == (direction,relative)]
        counts = [len({(row["platform"],row["absolute_orbit"]) for row in group}) for group in (before,after)]
        if counts[0] >= params["min_pre_observations"] and counts[1] >= params["min_post_observations"]:
            candidates.append({"id":f"{direction.lower()}-{relative}","pass":direction,"relative_orbit":relative,
                               "pre":before,"post":after,"pre_acquisition_count":counts[0],"post_acquisition_count":counts[1]})
    if not candidates:
        raise ValueError("No same-pass/same-relative-orbit track has enough independent pre/post acquisitions; do not mix tracks")
    if len(candidates) > params["max_orbit_candidates"]:
        raise ValueError("Too many flood track candidates; refusing an unbounded search")
    return candidates


def track_inventory(pre_rows: list[dict], post_rows: list[dict], params: dict) -> list[dict]:
    """Record even missing/insufficient tracks, so exclusions remain reviewable."""
    available = {(row["pass"],row["relative_orbit"]) for row in pre_rows+post_rows}
    result = []
    for direction,relative in sorted(available):
        row = {"id":f"{direction.lower()}-{relative}","pass":direction,"relative_orbit":relative}
        for period,scenes in (("pre",pre_rows),("post",post_rows)):
            selected = [item for item in scenes if (item["pass"],item["relative_orbit"]) == (direction,relative)]
            row[f"{period}_source_scene_count"] = len(selected)
            row[f"{period}_acquisition_count"] = len({(item["platform"],item["absolute_orbit"]) for item in selected})
        row["eligible_for_coverage_evaluation"] = row["pre_acquisition_count"] >= params["min_pre_observations"] and row["post_acquisition_count"] >= params["min_post_observations"]
        row["reason"] = "Enough same-track independent acquisitions; evaluate shared coverage" if row["eligible_for_coverage_evaluation"] else "Missing or insufficient independent acquisitions in one or both windows; never mixed with another track"
        result.append(row)
    return result


def _prepare_acquisition(ee: Any, rows: list[dict], params: dict) -> Any:
    # Deterministic mosaic of adjacent slices from ONE platform + absolute orbit,
    # before smoothing. This prevents treating overlapping slices as extra dates.
    source = ee.ImageCollection.fromImages([ee.Image(row["id"]) for row in sorted(rows,key=lambda row:row["id"])]).mosaic()
    native = ee.Image(rows[0]["id"]).select("VV").projection()
    source = source.setDefaultProjection(native)
    vv, angle = source.select("VV"), source.select("angle")
    support = vv.mask().reduce(ee.Reducer.min()).And(angle.mask()).And(angle.gte(params["incidence_min_degrees"])).And(angle.lte(params["incidence_max_degrees"]))
    power = ee.Image.constant(10).pow(vv.divide(10)).setDefaultProjection(native).updateMask(support)
    # Zero outside scene support BEFORE neighborhood filtering so edges cannot
    # borrow valid pixels from only one side and appear observed.
    neighborhood_support = (support.unmask(0,sameFootprint=False)
                            .focal_min(radius=params["smoothing_radius_m"],kernelType="circle",units="meters"))
    smoothed = power.focal_mean(radius=params["smoothing_radius_m"],kernelType="circle",units="meters")
    complete20 = _on_grid(neighborhood_support.reduceResolution(reducer=ee.Reducer.min(),maxPixels=64),params)
    power20 = _on_grid(smoothed.updateMask(neighborhood_support).reduceResolution(reducer=ee.Reducer.mean(),maxPixels=64),params)
    return power20.updateMask(complete20).rename("vv_power")


def _composite(ee: Any, rows: list[dict], params: dict, period: str) -> tuple[Any,Any]:
    groups: dict[tuple[str,int],list[dict]] = {}
    for row in rows:
        groups.setdefault((row["platform"],row["absolute_orbit"]),[]).append(row)
    collection = ee.ImageCollection.fromImages([_prepare_acquisition(ee,group,params) for _,group in sorted(groups.items())])
    count = _on_grid(collection.count().rename(f"{period}_observation_count").unmask(0,sameFootprint=False),params)
    composite = _on_grid(collection.median(),params).updateMask(count.gte(params[f"min_{period}_observations"]))
    # Threshold the same float32 values that are exported, avoiding a central
    # mask built from hidden extra precision that cannot reproduce on return.
    db = composite.log10().multiply(10).toFloat().rename(f"{period}_vv_db")
    return db,count


def _coverage(ee: Any, aoi: Any, pre: Any, post: Any, params: dict) -> dict:
    from .satellite_common import analysis_grid, grid_domain
    grid = analysis_grid(params["aoi_bbox"],params["analysis_crs"],20)
    domain,region = grid_domain(ee,params["aoi_bbox"],grid)
    pre_support = domain.where(pre.mask().reduce(ee.Reducer.min()).unmask(0,sameFootprint=False).lt(.5),0)
    post_support = domain.where(post.mask().reduce(ee.Reducer.min()).unmask(0,sameFootprint=False).lt(.5),0)
    transform = list(grid.transform)[:6]
    area = ee.Image.pixelArea().reproject(crs=grid.crs,crsTransform=transform).toFloat().updateMask(domain)
    graph = ee.Image.cat([
        area.rename("aoi_area_m2"), area.updateMask(pre_support).rename("pre_area_m2"),
        area.updateMask(post_support).rename("post_area_m2"),
        area.updateMask(pre_support.And(post_support)).rename("shared_area_m2"),
    ])
    values = graph.reduceRegion(reducer=ee.Reducer.sum(),geometry=region,crs=grid.crs,
                               crsTransform=transform,maxPixels=100_000_000,
                               bestEffort=False,tileScale=4).getInfo()
    keys = {"aoi_area_m2","pre_area_m2","post_area_m2","shared_area_m2"}
    if not isinstance(values,dict) or set(values) != keys:
        raise ValueError("Incomplete Sentinel-1 shared-coverage reduction")
    cleaned = {}
    for key,value in values.items():
        if value is None and key != "aoi_area_m2":
            value = 0.0
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid Sentinel-1 coverage area")
        cleaned[key] = float(value)
    if cleaned["aoi_area_m2"] <= 0 or cleaned["shared_area_m2"] > min(cleaned["pre_area_m2"],cleaned["post_area_m2"]) + 0.01 or max(cleaned["pre_area_m2"],cleaned["post_area_m2"]) > cleaned["aoi_area_m2"] + 0.01:
        raise ValueError("Sentinel-1 coverage violates area subset bounds")
    cleaned["shared_fraction"] = cleaned["shared_area_m2"]/cleaned["aoi_area_m2"]
    return cleaned


def select_track(evaluated: list[dict], minimum: float) -> dict:
    eligible = [row for row in evaluated if row["coverage"]["shared_fraction"] >= minimum]
    if not eligible:
        raise ValueError("No same-track Sentinel-1 comparison passes the shared-coverage gate; no fallback to mixed orbits or wider dates")
    return sorted(eligible,key=lambda row:(-row["coverage"]["shared_area_m2"],row["pass"],row["relative_orbit"]))[0]


def candidate_numpy(pre_db: Any, post_db: Any, shared: Any, water_class: Any, slope: Any,
                    *, baseline_valid: Any=None, worldcover_water_fraction: Any=None,
                    post_vv_max_db: float=-16, decrease_min_db: float=3, slope_max_degrees: float=5) -> tuple[Any,Any]:
    """Small transparent local oracle for auditing exported numerical arrays.

    It does not estimate source pixels, smooth rasters, or produce observed data.
    Invalid numbers are never eligible. Return (candidate boolean, eligible).
    """
    import numpy as np
    before,after,coverage,water,terrain = np.broadcast_arrays(pre_db,post_db,shared,water_class,slope)
    if baseline_valid is None or worldcover_water_fraction is None:
        raise ValueError("Flood v2 requires explicit independent WorldCover baseline support; JRC no-data is never assumed dry")
    baseline,fraction = np.broadcast_arrays(baseline_valid,worldcover_water_fraction)
    eligible = (np.isfinite(before) & np.isfinite(after) & np.isfinite(terrain)
                & (coverage == 1) & (baseline == 1) & np.isfinite(fraction) & (fraction == 0)
                & (water != 3) & (terrain >= 0) & (terrain < slope_max_degrees))
    candidate = eligible & (after <= post_vv_max_db) & ((after-before) <= -decrease_min_db)
    return candidate,eligible


def _baseline(ee: Any, params: dict) -> dict[str,Any]:
    """Independent known-land support; never decode JRC masks as dry land."""
    source = ee.Image(ee.ImageCollection(LANDCOVER_DATASET).first()).select("Map")
    known = source.remap(list(LANDCOVER_CLASSES),[1]*len(LANDCOVER_CLASSES),0).And(source.mask()).unmask(0,sameFootprint=False).setDefaultProjection(source.projection())
    known_fraction = _on_grid(known.reduceResolution(reducer=ee.Reducer.mean(),maxPixels=64),params).toFloat().rename("worldcover_known_fraction")
    baseline = _on_grid(known.reduceResolution(reducer=ee.Reducer.min(),maxPixels=64),params).eq(1).rename("baseline_valid")
    water_fraction = _on_grid(source.eq(80).unmask(0,sameFootprint=False).reduceResolution(reducer=ee.Reducer.mean(),maxPixels=64),params).toFloat().rename("worldcover_water_fraction")
    # Categorical MODE is contextual only. Water exclusion uses any overlap of
    # native class80 (fraction>0), never the mode/nearest sampled category.
    mode = _on_grid(source.updateMask(known).reduceResolution(reducer=ee.Reducer.mode(),maxPixels=64),params).rename("worldcover2020")
    return {"worldcover2020":mode,"worldcover_known_fraction":known_fraction,"worldcover_water_fraction":water_fraction,"baseline_valid":baseline}


def audit_arrays(arrays: Mapping[str,Any], nodata: float=-9999, *, config: Any=None) -> dict:
    """Recompute the flood formula from returned numerical rasters, fail closed.

    Expected arrays are the single bands emitted by build(), plus the common
    exporter domain and pixel_area_m2. Synthetic tests are clearly labelled;
    calling this function never establishes source accuracy or satellite origin.
    """
    import numpy as np
    params = parameters(config or FloodConfig())
    required = {"domain","pre_vv_db","post_vv_db","delta_vv_db","candidate","eligible",
                "pre_valid","post_valid","shared_valid","pre_observation_count","post_observation_count",
                "water_class_2020","permanent_water","slope_degrees","pixel_area_m2",
                "worldcover2020","worldcover_known_fraction","worldcover_water_fraction","baseline_valid"}
    if not required <= set(arrays):
        raise ValueError(f"Missing flood audit rasters: {sorted(required-set(arrays))}")
    values = {}
    for name in required:
        value = np.asarray(arrays[name])
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or not np.issubdtype(value.dtype,np.number):
            raise ValueError(f"Expected numeric single-band raster: {name}")
        values[name] = value
    if len({value.shape for value in values.values()}) != 1:
        raise ValueError("Flood audit raster shapes differ")
    domain = values["domain"] == 1
    if not domain.any() or not np.isin(values["domain"],[0,1,nodata]).all():
        raise ValueError("Invalid/empty flood domain")
    for name,value in values.items():
        if not np.isfinite(value[domain]).all():
            raise ValueError(f"Nonfinite in-domain flood values: {name}")
    for name in ("pre_valid","post_valid","shared_valid","eligible","baseline_valid"):
        if not np.isin(values[name][domain],[0,1]).all():
            raise ValueError(f"Flood flag is not dense binary within domain: {name}")
    pre_flag,post_flag = values["pre_valid"] == 1,values["post_valid"] == 1
    shared = pre_flag & post_flag & domain
    if not np.array_equal(values["shared_valid"][domain],shared[domain].astype(int)):
        raise ValueError("Shared SAR support differs from pre/post intersection")
    backscatter_diagnostics = {}
    for period,flag in (("pre",pre_flag),("post",post_flag)):
        count = values[f"{period}_observation_count"]
        if ((count[domain] < 0).any() or (count[domain] > params["max_scenes_per_window"]).any()
                or not np.equal(count[domain],np.floor(count[domain])).all()):
            raise ValueError("Invalid independent-acquisition observation counts")
        if not np.array_equal(flag[domain],count[domain] >= params[f"min_{period}_observations"]):
            raise ValueError("SAR valid flag does not match minimum independent observations")
        raster = values[f"{period}_vv_db"]
        if not np.array_equal(raster[domain] != nodata,flag[domain]):
            raise ValueError("SAR numerical mask differs from valid flag")
        samples = raster[flag & domain]
        # Sigma0 has no universal +20dB physical ceiling: bright point targets
        # can exceed it. Preserve values and flag for review; do not clip them,
        # alter thresholds or mask by observed outcome. Structural conversion
        # failures remain fatal, independently of any exploratory range flag.
        with np.errstate(over="ignore",under="ignore",invalid="ignore"):
            reconstructed_power = np.power(10.0,samples.astype(np.float64)/10)
        if not np.isfinite(reconstructed_power).all() or not (reconstructed_power > 0).all():
            raise ValueError("Backscatter cannot be represented as finite positive linear power")
        flagged = domain & flag & ((raster < -60) | (raster > 20))
        rows,columns = np.nonzero(flagged)
        backscatter_diagnostics[period] = {
            "min_db":float(samples.min()),"max_db":float(samples.max()),
            "screening_range_db":[-60,20],"screening_is_physical_bound":False,
            "outside_screening_count":int(flagged.sum()),
            "outside_screening_eligible_count":int((flagged & (values["eligible"]==1)).sum()),
            "outside_screening_candidate_count":int((flagged & (values["candidate"]==1)).sum()),
            "outside_screening_rowcol_bbox":([int(rows.min()),int(columns.min()),int(rows.max()),int(columns.max())] if rows.size else None),
            "review_required":bool(flagged.any()),"action":"Retain unchanged raw values; flag strong/weak returns for review, with no post-hoc masking or threshold change",
        }
    delta = values["delta_vv_db"]
    if not np.array_equal(delta[domain] != nodata,shared[domain]):
        raise ValueError("VV-change mask differs from shared SAR support")
    if not np.allclose(delta[shared],(values["post_vv_db"]-values["pre_vv_db"])[shared],rtol=0,atol=3e-5):
        raise ValueError("VV-change formula does not reproduce")
    water = values["water_class_2020"]
    if not np.isin(water[domain],[0,1,2,3,nodata]).all():
        raise ValueError("Unknown JRC yearly water class")
    water_known = domain & np.isin(water,[1,2,3])
    cover = values["worldcover2020"]
    if not np.isin(cover[domain],[*LANDCOVER_CLASSES,nodata]).all():
        raise ValueError("Unknown WorldCover2020 categorical class")
    known_fraction = values["worldcover_known_fraction"]
    water_fraction = values["worldcover_water_fraction"]
    if ((known_fraction[domain] < 0).any() or (known_fraction[domain] > 1).any()
            or (water_fraction[domain] < 0).any() or (water_fraction[domain] > 1).any()):
        raise ValueError("WorldCover native-pixel fractions outside [0,1]")
    baseline = values["baseline_valid"] == 1
    if ((baseline & domain & (known_fraction < 1-1e-6)).any()
            or (baseline & domain & (cover == nodata)).any()
            or ((cover == 80) & domain & (water_fraction <= 0)).any()
            or ((water_fraction == 1) & domain & (cover != 80)).any()):
        raise ValueError("WorldCover support/categorical/fraction consistency failed")
    permanent = values["permanent_water"]
    expected_permanent = (water_fraction > 0) | (water == 3)
    if not np.array_equal(permanent[domain] != nodata,baseline[domain]) or not np.array_equal(permanent[domain & baseline],expected_permanent[domain & baseline].astype(int)):
        raise ValueError("Permanent-water union differs from WorldCover any-water or observed JRC class3")
    slope = values["slope_degrees"]
    valid_slope = domain & (slope != nodata)
    if ((slope[valid_slope] < 0).any() or (slope[valid_slope] > 90).any()):
        raise ValueError("Invalid terrain slope values")
    expected_candidate,expected_eligible = candidate_numpy(values["pre_vv_db"],values["post_vv_db"],shared.astype(int),water,slope,
                         baseline_valid=baseline.astype(int),worldcover_water_fraction=water_fraction,
                         post_vv_max_db=params["post_vv_max_db"],decrease_min_db=params["decrease_min_db"],slope_max_degrees=params["slope_max_degrees"])
    expected_eligible &= domain
    expected_candidate &= domain
    if not np.array_equal(values["eligible"][domain],expected_eligible[domain].astype(int)):
        raise ValueError("Eligible support does not reproduce from SAR/water/terrain masks")
    candidate = values["candidate"]
    if not np.array_equal(candidate[domain] != nodata,expected_eligible[domain]):
        raise ValueError("Candidate no-data mask differs from eligible support")
    if not np.array_equal(candidate[expected_eligible],expected_candidate[expected_eligible].astype(int)):
        # Threshold equality at float32 precision is separately reported below;
        # the declared central mask must still reproduce exactly before import.
        raise ValueError("Candidate inundation formula does not reproduce")
    area = values["pixel_area_m2"].astype(float)
    if (area[domain] <= 0).any() or (area[domain] > 500).any():
        raise ValueError("Invalid physical pixel areas for the 20 m grid")
    baseline_fraction = float(area[domain & baseline].sum()/area[domain].sum())
    if baseline_fraction < params["min_baseline_fraction"]:
        raise ValueError("Known independent WorldCover baseline fails full-domain coverage gate")
    sensitivity = []
    for threshold in (-14,-16,-18):
        for drop in (2,3,4):
            mask,_ = candidate_numpy(values["pre_vv_db"],values["post_vv_db"],shared.astype(int),water,slope,
                                    baseline_valid=baseline.astype(int),worldcover_water_fraction=water_fraction,
                                    post_vv_max_db=threshold,decrease_min_db=drop,slope_max_degrees=params["slope_max_degrees"])
            sensitivity.append({"post_vv_max_db":threshold,"decrease_min_db":drop,"candidate_pixel_count":int(np.count_nonzero(mask & domain)),
                                "candidate_area_ha":float(area[mask & domain].sum()/10_000)})
    return {"status":"numerical_formula_checks_passed","independent_accuracy_validation":False,
            "domain_pixel_count":int(domain.sum()),"shared_pixel_count":int(shared.sum()),
            "eligible_pixel_count":int(expected_eligible.sum()),"candidate_pixel_count":int(expected_candidate.sum()),
            "candidate_area_ha":float(area[expected_candidate].sum()/10_000),
            "shared_area_ha":float(area[shared].sum()/10_000),"eligible_area_ha":float(area[expected_eligible].sum()/10_000),
            "baseline_valid_area_ha":float(area[domain & baseline].sum()/10_000),"baseline_fraction_of_domain":baseline_fraction,
            "jrc_known_fraction_of_domain":float(area[water_known].sum()/area[domain].sum()),
            "baseline_interpretation":"Known pre-event WorldCover2020 categories; JRC missing values are never recoded as dry. Exclude any WorldCover native water overlap or observed JRC class3.",
            "conservative_baseline_exclusions_with_rounded_full_fraction":int((domain & ~baseline & (known_fraction >= 1-1e-6)).sum()),
            "backscatter_diagnostics":backscatter_diagnostics,
            "threshold_sensitivity":sensitivity,"sensitivity_interpretation":"Exploratory threshold stability, not uncertainty bounds, validation or a confidence interval"}


def build(ee: Any, config: Any = None) -> Any:
    from .satellite_common import SatelliteAnalysis
    config = config or FloodConfig()
    params = parameters(config)
    aoi = ee.Geometry.Rectangle(params["aoi_bbox"],proj="EPSG:4326",geodesic=False)
    before,after = _catalogue(ee,aoi,params,"pre"),_catalogue(ee,aoi,params,"post")
    evaluated = []
    graphs = {}
    try:
        tracks = track_candidates(before,after,params)
        for track in tracks:
            pre,pre_count = _composite(ee,track["pre"],params,"pre")
            post,post_count = _composite(ee,track["post"],params,"post")
            coverage = _coverage(ee,aoi,pre,post,params)
            evaluated.append({key:track[key] for key in ("id","pass","relative_orbit","pre_acquisition_count","post_acquisition_count")} | {"coverage":coverage})
            graphs[track["id"]] = (pre,post,pre_count,post_count)
        selected = select_track(evaluated,params["min_valid_fraction"])
    except Exception as error:
        error.satellite_diagnostics = {"parameters":params,"catalogue_pre":before,"catalogue_post":after,"evaluated_tracks":evaluated}
        raise
    selected_track = next(track for track in tracks if track["id"] == selected["id"])
    pre,post,pre_count,post_count = graphs[selected["id"]]
    pre_support = _on_grid(pre.mask().reduce(ee.Reducer.min()).unmask(0,sameFootprint=False),params)
    post_support = _on_grid(post.mask().reduce(ee.Reducer.min()).unmask(0,sameFootprint=False),params)
    shared = pre_support.min(post_support).rename("shared_sar_valid")
    water = _on_grid(ee.ImageCollection(WATER_DATASET).filter(ee.Filter.eq("year",params["permanent_water_year"])).first(),params).select("waterClass").rename("water_class_2020")
    baseline_rasters = _baseline(ee,params)
    baseline_valid = baseline_rasters["baseline_valid"]
    # JRC missing is left missing in its numeric source raster. A false fill
    # here means only "no additional observed JRC class3 exclusion"; independent
    # WorldCover known support, not that fill, establishes baseline eligibility.
    permanent = baseline_rasters["worldcover_water_fraction"].gt(0).Or(water.eq(3).unmask(0,sameFootprint=False)).updateMask(baseline_valid).rename("permanent_water")
    dem = ee.Image(DEM_DATASET).select("elevation")
    # Compute slope at the DEM's native projection; only then sample to the
    # declared grid. Resampling an elevation mosaic before slope changes slopes.
    slope = _on_grid(ee.Terrain.slope(dem),params).toFloat().rename("slope_degrees")
    terrain_support = slope.mask().And(slope.gte(0)).unmask(0,sameFootprint=False)
    eligible = (shared.And(baseline_valid).And(terrain_support).And(permanent.unmask(1,sameFootprint=False).Not())
                .And(slope.unmask(90,sameFootprint=False).lt(params["slope_max_degrees"]))).rename("eligible")
    delta = post.subtract(pre).toFloat().rename("delta_vv_db")
    candidate = post.lte(params["post_vv_max_db"]).And(delta.lte(-params["decrease_min_db"])).updateMask(eligible).rename("candidate")
    rasters = {"pre_vv_db":pre,"post_vv_db":post,"delta_vv_db":delta,"candidate":candidate.toByte(),
               "pre_observation_count":pre_count,"post_observation_count":post_count,
               "water_class_2020":water,"permanent_water":permanent,"slope_degrees":slope,
               "pre_valid":pre_support,"post_valid":post_support,"shared_valid":shared,"eligible":eligible,
               **baseline_rasters}
    rasters = {key:image.clip(aoi) for key,image in rasters.items()}
    display_specs = [
        {"id":"pre_vv_db","raster":"pre_vv_db","label":"Before — Sentinel-1 VV backscatter","vis":{"min":-25,"max":0,"palette":["#08111d","#e4edf2"]},"units":"dB"},
        {"id":"post_vv_db","raster":"post_vv_db","label":"Event period — Sentinel-1 VV backscatter","vis":{"min":-25,"max":0,"palette":["#08111d","#e4edf2"]},"units":"dB"},
        {"id":"delta_vv_db","raster":"delta_vv_db","label":"VV change — event minus before","vis":{"min":-8,"max":8,"palette":["#245fc5","#f1efe6","#c46531"]},"units":"dB"},
        {"id":"candidate","raster":"candidate","label":"Candidate open-water inundation","vis":{"min":0,"max":1,"palette":["#dbe2df","#245fc5"]},"units":"0=eligible noncandidate; 1=candidate"},
        {"id":"permanent_water","raster":"permanent_water","label":"2020 permanent-water union context","vis":{"min":0,"max":1,"palette":["#e8e6d9","#78b7d8"]},"units":"0=other known baseline; 1=any WorldCover permanent-water overlap or observed JRC permanent water"},
        {"id":"baseline_valid","raster":"baseline_valid","label":"Known WorldCover 2020 baseline support","vis":{"min":0,"max":1,"palette":["#5e6671","#50b6a4"]},"units":"0=unknown native land-cover support; 1=all contributing source categories known"},
        {"id":"eligible","raster":"eligible","label":"Eligible evaluation support","vis":{"min":0,"max":1,"palette":["#5e6671","#50b6a4"]},"units":"0=excluded or unobserved; 1=evaluated"},
    ]
    provenance = {
        "method":"Sentinel-1 VV candidate open-water inundation; direct authenticated satellite processing",
        "status":"awaiting_authenticated_pixel_computation_and_returned_export_audit",
        "method_version":"2.0.0",
        "revision":{"previous_event_id":"flood-lower-fraser-2021-satellite",
                    "previous_source_archive_sha256":"9ce9d4c042f31427e12e58658fe1de5b496f9950f32582f06c27a29db0092d02",
                    "reason":"Returned v1 run had JRC2020 no-data at 2,371,570 of 2,534,825 domain cells (93.56%); it could not support an AOI-wide candidate interpretation. A separate observed WorldCover2020 baseline and full-domain known-baseline gate replace the assumption that JRC yearly support defines eligible land. Source JRC no-data remains unchanged.",
                    "range_guard_revision":"Six post pixels exceeded the arbitrary+20dB screening value (maximum21.9404dB); none was v1 eligible. V2 retains and flags finite strong returns for review rather than clipping/rejecting them by an unsupported universalphysicalceiling.",
                    "unchanged":"AOI, analysisgrid, pre/postdates, orbitcoverage-selectioncriterion, smoothing, incidencefilter, observationcountthresholds, slope and centralSARcandidate thresholds"},
        "parameters":params,"datasets":{"sentinel1":S1_DATASET,"jrc_water_context":WATER_DATASET,"known_baseline":LANDCOVER_DATASET,"dem":DEM_DATASET},
        "coverage_gates":[{"population":"domain","valid":"shared_valid","min_fraction":params["min_valid_fraction"]},
                          {"population":"domain","valid":"baseline_valid","min_fraction":params["min_baseline_fraction"]}],
        "selection":{"criterion":"Largest shared SAR valid area only, before examining flood candidate values; ties by pass then relative orbit", "track_inventory":track_inventory(before,after,params),"catalogue_pre":before,"catalogue_post":after,"evaluated_tracks":evaluated,"selected_track":selected},
        "pre":{"start":params["pre_start"],"end_exclusive":params["pre_end_exclusive"],"source_images":selected_track["pre"],"acquisition_count":selected_track["pre_acquisition_count"]},
        "post":{"start":params["post_start"],"end_exclusive":params["post_end_exclusive"],"source_images":selected_track["post"],"acquisition_count":selected_track["post_acquisition_count"]},
        "compositing":"Deterministic mosaic of slices per platform+absolute orbit; native VV dB converted to linear power; circular mean speckle smoothing; mean to fixed 20 m grid with full neighborhood/child-pixel support; temporal median linear power; convert to dB",
        "candidate_definition":"post_vv_db <= post_vv_max_db AND delta_vv_db <= -decrease_min_db, within eligible support only",
        "coverage_definition":"Full physical pixel areas for cells with centres inside the declared WGS84 box, on the anchored 20 m grid, exactly as the common exporter. Shared SAR support has sufficient independent observations in both windows. Independent WorldCover2020 known baseline must also cover at least80% domain. Eligible additionally requires valid low slope and no native WorldCover80 overlap or observed JRC2020class3. JRC no-data remains unknown contextual information and never establishes dry land.",
        "baseline_aggregation":"WorldCover2020 documented categories remapped to known support; minimum support across all contributing native10m cells. Permanent-water fraction is the area-weighted mean of nativeclass80 flags; any positive fraction excludes whole20m cell. Mode land-cover class is context only, not used to exclude water. Raw JRC2020 classifications remain separate.",
        "backscatter_range_policy":"Preserve all finite positive-power representable observations. Returns outside exploratory[-60,+20]dB screening range are retained unchanged and flagged with counts/locations/eligible involvement for review; the range is not a physicalbound and never clips pixels or relaxes flood thresholds.",
        "sources":SOURCES,"limitations":LIMITATIONS,
    }
    displays = []
    for spec in display_specs:
        legend = [{"color":color,"label":str(spec["vis"][key])} for color,key in ((spec["vis"]["palette"][0],"min"),(spec["vis"]["palette"][-1],"max"))]
        if spec["id"] == "candidate":
            legend = [{"color":"#dbe2df","label":"Eligible noncandidate"},{"color":"#245fc5","label":"Candidate surface inundation"}]
        displays.append({"id":spec["id"],"label":spec["label"],"image":rasters[spec["raster"]],"vis":spec["vis"],"legend":legend,"units":spec["units"]})
    return SatelliteAnalysis(
        event={"id":"flood-lower-fraser-2021-satellite-v2","hazard":"flood","title":"Lower Fraser 2021 — Sentinel-1 candidate inundation v2",
               "event_date":"2021-11-14","location":"Sumas Prairie / Lower Fraser Valley, British Columbia",
               "method":"Same-track Sentinel-1 VV backscatter change; independently observed pre-event land-cover support, conservative permanent-water union and terrain exclusions",
               "sources":SOURCES,"limitations":LIMITATIONS},
        aoi=aoi,bbox=tuple(params["aoi_bbox"]),crs=params["analysis_crs"],scale_m=20,
        rasters=rasters,displays=displays,provenance=provenance,
        pre_valid_id="pre_valid",post_valid_id="post_valid",shared_valid_id="shared_valid",candidate_id="candidate",
        audit={"formula":"delta_vv_db=post_vv_db-pre_vv_db; candidate=(post_vv_db<=post_vv_max_db)&(delta_vv_db<=-decrease_min_db), masked to eligible",
               "parameters":params,"module":"processing.satellite_flood","function":"audit_arrays",
               "threshold_sensitivity":{"post_vv_max_db":[-14,-16,-18],"decrease_min_db":[2,3,4],"interpretation":"Threshold stability only, not accuracy or a confidence interval"}},
    )
