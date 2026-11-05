"""Sentinel-1 surface-backscatter change around the 2025 Hubbard earthquake.

This is amplitude change on a conservative, low-slope deposit observation
support. It is NOT InSAR, ground displacement, a landslide inventory, or damage.
The graph requires an authenticated Earth Engine run; offline tests do not
establish scene availability, coverage, or a successful scientific result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import math
import re

import numpy as np


DATASET = "COPERNICUS/S1_GRD"
DEM_DATASET = "COPERNICUS/DEM/GLO30_2024_1"
SOURCE_EVENT = "https://www.usgs.gov/programs/landslide-hazards/science/2025-m70-hubbard-glacier-earthquake-triggered-landslides-and"
SOURCE_CHANGE = "https://doi.org/10.5066/P14H7MVU"
PALETTE = ["#305b9b", "#88b1cf", "#f1f0e8", "#edb873", "#b94b30"]


@dataclass(frozen=True)
class EarthquakeConfig:
    bbox: tuple[float, float, float, float] = (-139.97, 60.45, -139.60, 60.65)
    crs: str = "EPSG:32607"
    scale_m: int = 30
    pre_day: str = "2025-11-28"
    post_day: str = "2025-12-10"
    control_day: str = "2025-11-16"
    relative_orbit: int = 50
    platform: str = "A"
    orbit_pass: str = "ASCENDING"
    min_angle_deg: float = 30.0
    max_angle_deg: float = 45.0
    max_angle_difference_deg: float = 0.5
    max_slope_deg: float = 20.0
    include_control: bool = True
    min_shared_fraction: float = 0.8

    def __post_init__(self):
        if self.scale_m != 30 or self.crs != "EPSG:32607":
            raise ValueError("The pilot is predeclared on the 30 m EPSG:32607 grid")
        if self.bbox != (-139.97, 60.45, -139.60, 60.65):
            raise ValueError("Changing the pilot domain requires a separately reviewed experiment")
        if (self.pre_day, self.post_day, self.control_day) != ("2025-11-28", "2025-12-10", "2025-11-16"):
            raise ValueError("The matched 12-day sequence is fixed; do not silently widen the windows")
        if (self.platform, self.orbit_pass, self.relative_orbit) != ("A", "ASCENDING", 50):
            raise ValueError("The pilot requires Sentinel-1A ascending relative orbit 50")
        if not (0 <= self.min_angle_deg < self.max_angle_deg < 90 and
                0 < self.max_angle_difference_deg <= 1 and 0 < self.max_slope_deg <= 20 and
                0 < self.min_shared_fraction <= 1):
            raise ValueError("Invalid or less-conservative observation-support settings")

    @property
    def transform(self):
        # Fixed global UTM lattice; export bounds determine a bounded window.
        return [30, 0, 0, 0, -30, 10000000]


EXPECTED = {
    "2025-11-16": {"start": "2025-11-16T02:55:33Z", "absolute_orbit": 61897,
                   "prefix": "S1A_IW_GRDH_1SDV_20251116T025533_20251116T025558_061897_"},
    "2025-11-28": {"start": "2025-11-28T02:55:32Z", "absolute_orbit": 62072,
                   "prefix": "S1A_IW_GRDH_1SDV_20251128T025532_20251128T025557_062072_"},
    "2025-12-10": {"start": "2025-12-10T02:55:31Z", "absolute_orbit": 62247,
                   "prefix": "S1A_IW_GRDH_1SDV_20251210T025531_20251210T025556_062247_"},
}


def validate_metadata(metadata: dict, day: str, config: EarthquakeConfig) -> list[dict]:
    """Bounded catalogue validation, independent of the EE pixel graph."""
    expected = EXPECTED[day]
    if metadata.get("count") != 1:
        raise ValueError(f"Expected exactly one matched frame on {day}; inspect catalogue provenance before changing selection")
    fields = ("indices", "timestamps", "platforms", "passes", "relative_orbits", "absolute_orbits")
    if any(not isinstance(metadata.get(key), list) or len(metadata[key]) != 1 for key in fields):
        raise ValueError("Incomplete Sentinel-1 metadata")
    index = metadata["indices"][0]
    if not isinstance(index, str) or not re.fullmatch(re.escape(expected["prefix"]) + r"[0-9A-F]{6}_[0-9A-F]{4}", index):
        raise ValueError(f"Unexpected scene acquisition/frame identity on {day}")
    acquired = datetime.fromtimestamp(metadata["timestamps"][0] / 1000, timezone.utc)
    if abs((acquired - datetime.fromisoformat(expected["start"].replace("Z", "+00:00"))).total_seconds()) > 1:
        raise ValueError("Scene timestamp differs from the independently checked acquisition")
    if (metadata["platforms"][0] != config.platform or metadata["passes"][0] != config.orbit_pass or
            metadata["relative_orbits"][0] != config.relative_orbit or metadata["absolute_orbits"][0] != expected["absolute_orbit"]):
        raise ValueError("Platform, pass or orbit mismatch")
    return [{"id": f"{DATASET}/{index}", "acquired_at": acquired.isoformat().replace("+00:00", "Z"),
             "platform": config.platform, "orbit_pass": config.orbit_pass,
             "relative_orbit": config.relative_orbit, "absolute_orbit": expected["absolute_orbit"]}]


def _collection(ee, aoi, day, config):
    end = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    collection = (ee.ImageCollection(DATASET).filterBounds(aoi).filterDate(day, end)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.eq("resolution_meters", 10))
                  .filter(ee.Filter.eq("platform_number", config.platform))
                  .filter(ee.Filter.eq("orbitProperties_pass", config.orbit_pass))
                  .filter(ee.Filter.eq("relativeOrbitNumber_start", config.relative_orbit))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                  .sort("system:index"))
    metadata = ee.Dictionary({"count": collection.size(),
                              "indices": collection.aggregate_array("system:index"),
                              "timestamps": collection.aggregate_array("system:time_start"),
                              "platforms": collection.aggregate_array("platform_number"),
                              "passes": collection.aggregate_array("orbitProperties_pass"),
                              "relative_orbits": collection.aggregate_array("relativeOrbitNumber_start"),
                              "absolute_orbits": collection.aggregate_array("orbitNumber_start")}).getInfo()
    records = validate_metadata(metadata, day, config)
    return ee.Image(collection.first()), records


def _prepare(ee, image, config):
    """Area-mean linear power on the fixed grid, then convert back to dB.

    A target cell is admitted only if every contributing native sample has
    both polarizations and an accepted incidence angle. This avoids a small
    valid fragment masquerading as a complete 30 m observation.
    """
    angle = image.select("angle").reproject(crs=image.select("VV").projection())
    support_native = (image.select(["VV", "VH"]).mask().reduce(ee.Reducer.min())
                      .And(angle.mask()).And(angle.gte(config.min_angle_deg)).And(angle.lte(config.max_angle_deg)))
    support = (support_native.unmask(0, sameFootprint=False)
               .reduceResolution(reducer=ee.Reducer.min(), maxPixels=64)
               .reproject(crs=config.crs, crsTransform=config.transform).eq(1))
    # Keep the native SAR band's projection (a left-hand constant image would
    # instead introduce its default one-degree projection before aggregation).
    power = image.select(["VV", "VH"]).multiply(math.log(10) / 10).exp().updateMask(support_native)
    mean_power = (power.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
                  .reproject(crs=config.crs, crsTransform=config.transform))
    db = mean_power.log10().multiply(10).updateMask(support)
    mean_angle = (angle.updateMask(support_native).reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
                  .reproject(crs=config.crs, crsTransform=config.transform).updateMask(support))
    return db, mean_angle, support


def build(ee, config=None):
    from .satellite_common import SatelliteAnalysis

    config = config or EarthquakeConfig()
    aoi = ee.Geometry.Rectangle(list(config.bbox), proj="EPSG:4326", geodesic=False)
    pre_source, pre_records = _collection(ee, aoi, config.pre_day, config)
    post_source, post_records = _collection(ee, aoi, config.post_day, config)
    pre, pre_angle, pre_count = _prepare(ee, pre_source, config)
    post, post_angle, post_count = _prepare(ee, post_source, config)
    dem_tiles = ee.ImageCollection(DEM_DATASET).filterBounds(aoi).select("DEM").sort("system:index")
    dem = (dem_tiles.mosaic().setDefaultProjection(ee.Image(dem_tiles.first()).projection())
           .resample("bilinear").reproject(crs=config.crs, crsTransform=config.transform))
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    eligible = slope.lte(config.max_slope_deg).And(slope.mask()).unmask(0, sameFootprint=False).rename("eligible")
    angle_difference = post_angle.subtract(pre_angle).rename("angle_difference_deg")
    geometry_match = angle_difference.abs().lte(config.max_angle_difference_deg).unmask(0, sameFootprint=False)
    # Pre/post flags are intentionally pair-conditioned by angle agreement;
    # pre_count/post_count retain each acquisition's independent availability.
    pre_valid = pre_count.And(eligible).And(geometry_match).unmask(0, sameFootprint=False).rename("pre_valid")
    post_valid = post_count.And(eligible).And(geometry_match).unmask(0, sameFootprint=False).rename("post_valid")
    shared = pre_valid.And(post_valid).rename("shared_valid")
    rasters = {
        "eligible": eligible.toByte(), "slope_deg": slope,
        "pre_count": pre_count.unmask(0, sameFootprint=False).rename("pre_count").toByte(),
        "post_count": post_count.unmask(0, sameFootprint=False).rename("post_count").toByte(),
        "pre_valid": pre_valid.toByte(), "post_valid": post_valid.toByte(), "shared_valid": shared.toByte(),
        "pre_angle_deg": pre_angle.rename("pre_angle_deg"), "post_angle_deg": post_angle.rename("post_angle_deg"),
        "angle_difference_deg": angle_difference,
    }
    for polarization in ("vv", "vh"):
        before = pre.select(polarization.upper()).updateMask(pre_valid).rename(f"pre_{polarization}_db")
        after = post.select(polarization.upper()).updateMask(post_valid).rename(f"post_{polarization}_db")
        rasters[f"pre_{polarization}_db"] = before
        rasters[f"post_{polarization}_db"] = after
        rasters[f"delta_{polarization}_db"] = after.subtract(before).updateMask(shared).rename(f"delta_{polarization}_db")
    control_records = []
    if config.include_control:
        control_source, control_records = _collection(ee, aoi, config.control_day, config)
        control, control_angle, control_count = _prepare(ee, control_source, config)
        control_match = control_angle.subtract(pre_angle).abs().lte(config.max_angle_difference_deg).unmask(0, sameFootprint=False)
        control_valid = control_count.And(shared).And(control_match).unmask(0, sameFootprint=False).rename("control_valid")
        rasters["control_valid"] = control_valid.toByte()
        rasters["control_count"] = control_count.unmask(0, sameFootprint=False).rename("control_count").toByte()
        rasters["control_angle_deg"] = control_angle.rename("control_angle_deg")
        for polarization in ("vv", "vh"):
            baseline = control.select(polarization.upper()).updateMask(control_valid).rename(f"control_{polarization}_db")
            rasters[f"control_{polarization}_db"] = baseline
            rasters[f"control_delta_{polarization}_db"] = (rasters[f"pre_{polarization}_db"].subtract(baseline)
                                                             .updateMask(control_valid).rename(f"control_delta_{polarization}_db"))
    displays = []
    backscatter_legend = [{"color": "#111111", "label": "Lower backscatter"}, {"color": "#eeeeee", "label": "Higher backscatter"}]
    difference_legend = [{"color": PALETTE[0], "label": "Decrease (−7 dB display limit)"},
                         {"color": PALETTE[2], "label": "Approximately no backscatter change"},
                         {"color": PALETTE[-1], "label": "Increase (+7 dB display limit)"}]
    for polarization in ("vv", "vh"):
        for period, label in (("pre", "Baseline"), ("post", "Comparison")):
            key = f"{period}_{polarization}_db"
            displays.append({"id": key, "label": f"{label} {polarization.upper()} backscatter", "image": rasters[key],
                             "vis": {"min": -30 if polarization == "vh" else -25, "max": 0, "palette": ["111111", "eeeeee"]},
                             "legend": backscatter_legend, "units": "dB"})
        key = f"delta_{polarization}_db"
        displays.append({"id": key, "label": f"{polarization.upper()} backscatter difference (comparison − baseline)", "image": rasters[key],
                         "vis": {"min": -7, "max": 7, "palette": [color[1:] for color in PALETTE]},
                         "legend": difference_legend, "units": "dB"})
    if config.include_control:
        displays.append({"id": "control_delta_vv_db", "label": "Pre-event VV repeat difference (28 − 16 November)",
                         "image": rasters["control_delta_vv_db"],
                         "vis": {"min": -7, "max": 7, "palette": [color[1:] for color in PALETTE]},
                         "legend": difference_legend, "units": "dB", "allow_empty": True})
    displays.append({"id": "shared_valid", "label": "Accepted common observation support", "image": shared.selfMask(),
                     "vis": {"min": 0, "max": 1, "palette": ["567c6b", "567c6b"]},
                     "legend": [{"color": "#567c6b", "label": "Accepted low-slope common support"}], "units": "0/1"})
    limitations = [
        "This is measured radar-amplitude/backscatter change, not InSAR displacement, earthquake shaking, confirmed landslides, building damage or losses. GRD contains no phase information for interferometry.",
        "Snowfall, wetness, melt/freeze, glacier motion and viewing/processing effects can change backscatter. Timing and documented regional landslides do not establish the cause of every changed pixel.",
        "Only low-slope support (Copernicus GLO30 slope ≤20°) with matched same-platform ascending orbit, accepted incidence angles and complete source support is admitted. This omits steep landslide source/transport areas and is not a complete event footprint.",
        "Earth Engine terrain-corrects Sentinel-1 GRD geometrically but does not perform radiometric terrain flattening. The low-slope and angle screens reduce some problems; they do not prove freedom from local/distant terrain shadow, layover or foreshortening.",
        "C-band radar responds to surface properties. Positive VV change can be consistent with rougher avalanche deposits on glaciers, but cannot uniquely distinguish rock, snow, ice or other causes. No damage or landslide area is reported.",
        "Each period is one exact frame/date. Three-by-three approximate native-sample power aggregation to a 30 m lattice reduces speckle but does not remove it or create independent observations. Continuous differences and a pre-event repeat must be visually reviewed.",
        "No-data and excluded steep terrain are unknown, not unchanged. A successful coverage gate applies only to the explicitly low-slope eligible denominator; the full domain coverage is reported separately.",
        "The USGS provisional SAR/landslide products used the same acquisitions and are contextual corroboration, not an independent pixel-level accuracy reference. The EE preprocessing and conservative support differ from the USGS product.",
    ]
    provenance = {"config": asdict(config), "dataset": DATASET, "dem_dataset": DEM_DATASET,
                  "pre": {"search_start": config.pre_day, "search_end_exclusive": (date.fromisoformat(config.pre_day) + timedelta(days=1)).isoformat(), "images": pre_records},
                  "post": {"search_start": config.post_day, "search_end_exclusive": (date.fromisoformat(config.post_day) + timedelta(days=1)).isoformat(), "images": post_records},
                  "control": {"images": control_records, "status": "required_and_catalogue_matched" if config.include_control else "not_requested_no_control_result"},
                  "coverage_gate": {"raster": "eligible", "min_shared_fraction": config.min_shared_fraction},
                  "coverage_gates": [{"population": "eligible", "valid": "shared_valid", "min_fraction": config.min_shared_fraction}]
                                    + ([{"population": "shared_valid", "valid": "control_valid", "min_fraction": config.min_shared_fraction}]
                                       if config.include_control else []),
                  "support_definition": "GLO30 slope≤20°, all contributing native VV/VH/angle samples valid, incidence 30–45°, same A/ascending/relative50, pair |angle difference|≤0.5°",
                  "pre_post_valid_pair_conditioned": True, "radiometric_terrain_flattening": False,
                  "local_shadow_or_layover_freedom_demonstrated": False,
                  "native_to_analysis": "Area-weighted mean of linear sigma0 power, then 10log10; fixed30mUTM7N; all-contributor support",
                  "actual_results_status": "Requires authenticated pixel computation, export and independent audit; catalogue metadata alone are not results"}
    return SatelliteAnalysis(
        event={"id": "satellite-earthquake-hubbard-2025", "hazard": "earthquake",
               "title": "2025 Hubbard Glacier earthquake — Mount King George radar surface change",
               "event_date": "2025-12-06", "location": "Mount King George, Kluane region, Yukon, Canada",
               "method": "Matched Sentinel-1A VV/VH amplitude change on conservative low-slope support; pre-event repeat diagnostic",
               "sources": [{"label": "USGS documented landslide response and Yukon reconnaissance", "url": SOURCE_EVENT},
                           {"label": "USGS provisional same-acquisition SAR difference (context, not independent truth)", "url": SOURCE_CHANGE},
                           {"label": "Earth Engine Sentinel-1 preprocessing", "url": "https://developers.google.com/earth-engine/guides/sentinel1"},
                           {"label": "Copernicus GLO30 2024_1 dataset", "url": "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30_2024_1",
                            "attribution": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
                            "licence_notice": "The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30"},
                           {"label": "Official Mount King George location", "url": "https://geonames.nrcan.gc.ca/search-place-names/unique?id=KADIK"}],
               "limitations": limitations},
        aoi=aoi, bbox=config.bbox, crs=config.crs, scale_m=config.scale_m,
        rasters=rasters, displays=displays, provenance=provenance,
        pre_valid_id="pre_valid", post_valid_id="post_valid", shared_valid_id="shared_valid", candidate_id=None,
        audit={"kind": "satellite_earthquake_backscatter", "difference_formula": "delta_vv_db=post_vv_db-pre_vv_db; delta_vh_db=post_vh_db-pre_vh_db",
               "control_formula": "control_delta=pre-control, on control_valid", "candidate_threshold": None,
               "arithmetic_tolerance_db": 0.00005, "validity": "shared_valid=pre_valid AND post_valid; flags pair-conditioned by incidence agreement",
               "damage_or_landslide_area": "not estimated"})


def audit_arrays(arrays, nodata=-9999):
    """Independently audit returned numerical rasters, never infer impacts."""
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    required = ("pre_vv_db", "post_vv_db", "delta_vv_db", "pre_vh_db", "post_vh_db", "delta_vh_db",
                "pre_valid", "post_valid", "shared_valid", "eligible", "pre_count", "post_count", "slope_deg",
                "pre_angle_deg", "post_angle_deg", "angle_difference_deg")
    if any(key not in arrays for key in required):
        raise ValueError("Missing required earthquake numerical rasters")
    shape = arrays["shared_valid"].shape
    if len(shape) != 2 or any(arrays[key].shape != shape for key in arrays):
        raise ValueError("Numerical raster dimensions differ")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite returned numerical raster")
    domain = arrays.get("domain", np.ones(shape)) == 1

    def present(key):
        return (arrays[key] != nodata) & domain

    for key in ("pre_valid", "post_valid", "shared_valid", "eligible", "pre_count", "post_count"):
        if not np.isin(arrays[key][domain], [0, 1]).all():
            raise ValueError(f"Invalid observation flag/count: {key}")
    before = (arrays["pre_valid"] == 1) & domain
    after = (arrays["post_valid"] == 1) & domain
    shared = (arrays["shared_valid"] == 1) & domain
    if not np.array_equal(shared, before & after):
        raise ValueError("Shared validity is not the intersection of period validity")
    if (shared & (arrays["eligible"] != 1)).any() or (before & (arrays["pre_count"] != 1)).any() or (after & (arrays["post_count"] != 1)).any():
        raise ValueError("Observation counts or eligible support disagree with validity")
    if not shared.any():
        raise ValueError("No shared observations; do not manufacture an earthquake result")
    if ((arrays["slope_deg"][shared] < 0) | (arrays["slope_deg"][shared] > 20.0001)).any():
        raise ValueError("Accepted support exceeds the predeclared low-slope screen")
    for key in ("pre_angle_deg", "post_angle_deg"):
        if not present(key)[shared].all() or not ((arrays[key][shared] >= 29.9999) & (arrays[key][shared] <= 45.0001)).all():
            raise ValueError("Accepted incidence angle outside range")
    if (np.abs(arrays["post_angle_deg"][shared] - arrays["pre_angle_deg"][shared]) > .5001).any():
        raise ValueError("Incidence-angle mismatch on accepted support")
    expected_angle = arrays["post_angle_deg"][shared] - arrays["pre_angle_deg"][shared]
    if not np.allclose(arrays["angle_difference_deg"][shared], expected_angle, atol=5e-5, rtol=0):
        raise ValueError("Incidence-angle arithmetic mismatch")
    errors = {}
    for polarization in ("vv", "vh"):
        for period, valid in (("pre", before), ("post", after)):
            key = f"{period}_{polarization}_db"
            if not np.array_equal(present(key), valid):
                raise ValueError(f"Mask mismatch: {key}")
            if ((arrays[key][valid] < -100) | (arrays[key][valid] > 30)).any():
                raise ValueError("Backscatter outside broad physical/sanity bounds")
        key = f"delta_{polarization}_db"
        if not np.array_equal(present(key), shared):
            raise ValueError(f"Mask mismatch: {key}")
        expected = arrays[f"post_{polarization}_db"][shared] - arrays[f"pre_{polarization}_db"][shared]
        error = float(np.max(np.abs(arrays[key][shared] - expected)))
        if error > 5e-5:
            raise ValueError(f"Backscatter difference arithmetic mismatch: {key}")
        errors[key] = error
    if "control_valid" in arrays:
        if not np.isin(arrays["control_valid"][domain], [0, 1]).all():
            raise ValueError("Invalid control validity flag")
        valid = (arrays["control_valid"] == 1) & domain
        if (valid & ~shared).any():
            raise ValueError("Control comparison support exceeds event common support")
        if not all(key in arrays for key in ("control_count", "control_angle_deg", "control_vv_db", "control_vh_db", "control_delta_vv_db", "control_delta_vh_db")):
            raise ValueError("Incomplete pre-event control rasters")
        if (valid & (arrays["control_count"] != 1)).any():
            raise ValueError("Control count mismatch")
        if not np.isin(arrays["control_count"][domain], [0, 1]).all():
            raise ValueError("Invalid control observation count")
        if not present("control_angle_deg")[valid].all() or not ((arrays["control_angle_deg"][valid] >= 29.9999) & (arrays["control_angle_deg"][valid] <= 45.0001)).all():
            raise ValueError("Control incidence angle outside range")
        if (np.abs(arrays["control_angle_deg"][valid] - arrays["pre_angle_deg"][valid]) > .5001).any():
            raise ValueError("Control angle mismatch")
        for polarization in ("vv", "vh"):
            key, difference = f"control_{polarization}_db", f"control_delta_{polarization}_db"
            if not np.array_equal(present(key), valid) or not np.array_equal(present(difference), valid):
                raise ValueError("Control numerical mask mismatch")
            if valid.any():
                expected = arrays[f"pre_{polarization}_db"][valid] - arrays[key][valid]
                error = float(np.max(np.abs(arrays[difference][valid] - expected)))
                if error > 5e-5:
                    raise ValueError("Control difference arithmetic mismatch")
                errors[difference] = error
    return {"status": "numerical_consistency_passed", "shared_pixels": int(shared.sum()),
            "max_arithmetic_error_db": errors, "impact_classification": "not performed"}
