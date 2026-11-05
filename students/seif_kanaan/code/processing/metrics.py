"""Area calculations always use EE pixelArea at the fixed analysis grid.

https://developers.google.com/earth-engine/apidocs/ee-image-pixelarea
https://developers.google.com/earth-engine/apidocs/ee-image-reduceregion
"""

import math
import numpy as np

from .event import ANALYSIS_CRS, ANALYSIS_TRANSFORM

# The authenticated pilot reported 14,250,493 pixels for this reduction,
# exceeding the old 12 million guard. Allow the existing bounded AOI to finish
# at its exact 20 m grid. This changes permission to process pixels, not scale.
# Download limits are separate and remain enforced in export.py.
MAX_REDUCTION_PIXELS = 100_000_000


def nbr_array(nir, swir):
    """Independent local numerical oracle for unit fixtures, not exported observations."""
    nir, swir = np.asarray(nir, dtype=float), np.asarray(swir, dtype=float)
    denominator = nir + swir
    result = np.full(np.broadcast_shapes(nir.shape, swir.shape), np.nan)
    np.divide(nir - swir, denominator, out=result, where=denominator > 1e-6)
    return result


def validate_metrics(values, *, min_valid_fraction):
    keys = ("candidate_area_ha", "valid_area_ha", "aoi_area_ha", "pre_valid_area_ha", "post_valid_area_ha")
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid computed area: {key}={value!r}")
    candidate, valid, aoi = (values[k] for k in keys[:3])
    if aoi <= 0 or valid <= 0:
        raise ValueError("AOI has no valid shared pre/post land observations.")
    if candidate > valid + 1e-6 or valid > aoi + 1e-6:
        raise ValueError("Computed areas violate candidate <= valid <= AOI.")
    for key in ["pre_valid_area_ha", "post_valid_area_ha"]:
        if not valid - 1e-6 <= values[key] <= aoi + 1e-6:
            raise ValueError("Per-period coverage is inconsistent with shared coverage.")
    fraction = valid / aoi
    if fraction < min_valid_fraction:
        raise ValueError(f"Only {fraction:.1%} of AOI has shared valid land observations; required {min_valid_fraction:.0%}. Inspect dates/masks before retrying.")
    return {**values, "valid_fraction": fraction, "no_data_area_ha": aoi-valid,
            "pre_valid_fraction": values["pre_valid_area_ha"]/aoi,
            "post_valid_fraction": values["post_valid_area_ha"]/aoi}


def compute_metrics(ee, analysis, config):
    area = ee.Image.pixelArea().divide(10000)
    shared = analysis.dnbr.mask().reduce(ee.Reducer.min())
    stack = ee.Image.cat([
        area.updateMask(analysis.candidate).rename("candidate_area_ha"),
        area.updateMask(shared).rename("valid_area_ha"),
        area.rename("aoi_area_ha"),
        area.updateMask(analysis.pre_nbr.mask()).rename("pre_valid_area_ha"),
        area.updateMask(analysis.post_nbr.mask()).rename("post_valid_area_ha"),
    ])
    raw = stack.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=analysis.aoi,
        crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM,
        maxPixels=MAX_REDUCTION_PIXELS, bestEffort=False, tileScale=4,
    ).getInfo()
    # No candidates with valid coverage means zero; missing coverage stays a failure.
    if raw.get("candidate_area_ha") is None and raw.get("valid_area_ha", 0):
        raw["candidate_area_ha"] = 0.0
    return validate_metrics(raw, min_valid_fraction=config.min_valid_fraction)
