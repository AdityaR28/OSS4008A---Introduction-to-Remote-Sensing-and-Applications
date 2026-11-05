"""Descriptive overlap with an independent historical perimeter.

This is not a pixel-accuracy assessment. The pilot reference identifies its
creation method as 'Hand sketch of any type'; unburned islands, temporal
differences and unknown boundary precision limit all derived comparisons.
"""

import json
from pathlib import Path
import shutil

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from .export import NODATA, sha256


def compare_reference(directory, reference_path):
    directory, reference_path = Path(directory), Path(reference_path)
    if reference_path.stat().st_size > 5_000_000:
        raise ValueError("Reference must be a bounded GeoJSON under 5 MB.")
    collection = json.loads(reference_path.read_text(encoding="utf-8"))
    if collection.get("type") != "FeatureCollection" or not 1 <= len(collection.get("features", [])) <= 100:
        raise ValueError("Reference must contain 1–100 GeoJSON features.")
    provenance_path = reference_path.with_suffix(".provenance.json")
    if not provenance_path.exists():
        raise ValueError("Pilot reference requires its matching .provenance.json file.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("sha256") != sha256(reference_path):
        raise ValueError("Reference provenance digest does not match GeoJSON.")
    if provenance.get("event_id") != "mcdougall-creek-2023":
        raise ValueError("Reference provenance is not for the configured pilot event.")
    if collection.get("crs", {}).get("properties", {}).get("name", "urn:ogc:def:crs:OGC:1.3:CRS84") not in (
        "urn:ogc:def:crs:OGC:1.3:CRS84", "EPSG:4326", "urn:ogc:def:crs:EPSG::4326"
    ):
        raise ValueError("Reference must use WGS84 longitude/latitude.")
    geometries = []
    for feature in collection["features"]:
        properties = feature.get("properties", {})
        if properties.get("FIRE_NUMBER") != "K52767" or properties.get("FIRE_YEAR") != 2023:
            raise ValueError("Reference must identify the 2023 McDougall Creek fire K52767.")
        if properties.get("CREATION_METHOD") != "Hand sketch of any type":
            raise ValueError("Reference creation method changed; review source metadata and attribution before use.")
        geometry = feature.get("geometry", {})
        if geometry.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("Reference must contain Polygon/MultiPolygon geometry only.")
        geometries.append(geometry)
    with rasterio.open(directory/"rasters/candidate.tif") as source:
        candidate = source.read(1, masked=True)
        transformed = [transform_geom("EPSG:4326", source.crs, item) for item in geometries]
        reference = rasterize(((item, 1) for item in transformed), out_shape=source.shape,
                              transform=source.transform, fill=0, all_touched=False, dtype="uint8").astype(bool)
    with rasterio.open(directory/"rasters/pixel_area_m2.tif") as source:
        area = source.read(1, masked=True).filled(0).astype(np.float64) / 10000
    valid = ~np.ma.getmaskarray(candidate)
    selected = candidate.filled(NODATA) == 1
    reference_valid = reference & valid

    def hectares(mask):
        return float(area[mask].sum())

    intersection, union = hectares(selected & reference_valid), hectares(selected | reference_valid)
    comparison = {
        "status": "descriptive_overlap_only",
        "intersection_area_ha": intersection,
        "candidate_outside_reference_ha": hectares(selected & ~reference),
        "reference_without_candidate_within_valid_coverage_ha": hectares(reference_valid & ~selected),
        "reference_area_in_aoi_ha": hectares(reference),
        "reference_area_with_valid_observations_ha": hectares(reference_valid),
        "intersection_over_union_within_valid_coverage": intersection / union if union > 0 else None,
        "rasterization": "20 m analysis grid, pixel-centre rule; restricted to valid shared land observations for overlap",
        "notes": "Descriptive spatial agreement only. Historical perimeter may contain unburned islands and dates outside the composites. The pilot perimeter is hand sketched, with unverified boundary precision. No pixel-accuracy or damage claim.",
    }
    reference_dir = directory/"reference"
    reference_dir.mkdir(exist_ok=True)
    target = reference_dir/reference_path.name
    shutil.copy2(reference_path, target)
    shutil.copy2(provenance_path, reference_dir/provenance_path.name)
    return {
        "status": "loaded", "url": f"reference/{target.name}",
        "label": "BC Wildfire Service historical perimeter — hand-sketched contextual reference",
        "source_url": "https://catalogue.data.gov.bc.ca/dataset/bc-wildfire-fire-perimeters-historical",
        "sha256": sha256(target),
        "creation_method": "Hand sketch of any type", "comparison": comparison,
    }
