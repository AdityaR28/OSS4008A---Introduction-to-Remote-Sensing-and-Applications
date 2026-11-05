"""Bounded analysis of NRCan EGS flood polygons; no damage/depth inference.

Areas use an ellipsoidal equal-area projection (EPSG:6933), never Web Mercator
Shape_Area fields. Class 1 permanent water and class 6 unknown state are not
flood. Source polygons are unioned before area calculation so overlaps cannot
inflate the result. Display simplification is separate from metric geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pyproj import Transformer
from shapely import GeometryCollection, box, make_valid
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union


AREA_CRS = "EPSG:6933"
CLASS_LABELS = {
    0: "Unclassified", 1: "Permanent water (excluded from flood area)",
    2: "Open-water flood", 3: "Flooded vegetation", 4: "Urban flood",
    5: "River ice (excluded from flood area)", 6: "Unknown state (excluded from flood area)",
}
CLASS_COLORS = {0: "#64748b", 1: "#82bedf", 2: "#2359c4", 3: "#168f7a", 4: "#ab2f88", 5: "#8e63bf", 6: "#b67837"}
FLOOD_CLASSES = frozenset({2, 3, 4})
# Reviewed immutable historical snapshot. Updating this requires reviewing source
# revisions, not just changing a user-editable download ledger after corruption.
EXPECTED_SOURCE_SHA256 = {
    "raw/catalogue.json": "94fc33e7d64072ef4d29efa7aa028a29464cb65fd6d8cec236a371352c794dd0",
    "raw/service.json": "3801d55b122addc5fb30e9e0652db81e15cf329eef6868b6540932f14a59b81f",
    "raw/classes.json": "55f4395b61ef6f9aaabeeec3174d7f28f17d3610890af9d66d0c96a32afc782c",
    "raw/2021-11-17-ids.json": "c9c74887b48e1e96907939e33a68db1b36d0959c57a6f8bfd661ec33e0dc66e1",
    "raw/2021-11-17-polygons.geojson": "ce14174e9f1c0de2f2709f48ed6337c006b2a92e949a86c606a3fc9180b0e08c",
    "raw/2021-11-17-footprint.geojson": "4ac3e18295eacfb1c6cb471aa03c6d29b18f0ffb50d0096d6e456681ae62b469",
    "raw/2021-11-19-ids.json": "f58e6fc38d8bff2913da77554ce369a7abca268be40712f75fdfcedfdc649a2f",
    "raw/2021-11-19-polygons.geojson": "24569dcd60b1e091c05fe8b7e051c4d3cea17df54329a58ea5f134fa1798d597",
    "raw/2021-11-19-footprint.geojson": "c79d0775c4831ae8897c30fef320d70c0552737cc5cd837924edc31c943a6154",
    "raw/2021-11-25-ids.json": "5699fdd9974da89cd0559c45315f7f38c4bac1e7ecb5e18040d36296c30cd35a",
    "raw/2021-11-25-polygons.geojson": "94d54643526fcb4c53b23533d12ed5912b8b6a465364a4026f5b9ede9e6da74c",
    "raw/2021-11-25-footprint.geojson": "ea4b60c58c87dff674758dd67850271ee1efe45cbae39b70835d84b17e9786c2",
}


def polygonal(geometry: Any) -> Any:
    """Repair invalid rings, retain only polygonal members, and reject empties."""
    geometry = make_valid(geometry) if not geometry.is_valid else geometry
    if geometry.is_empty:
        return GeometryCollection()
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    members = [polygonal(part) for part in getattr(geometry, "geoms", [])]
    return unary_union([member for member in members if not member.is_empty])


def project_geometry(geometry: Any, source_crs: str, target_crs: str) -> Any:
    return transform(Transformer.from_crs(source_crs, target_crs, always_xy=True).transform, geometry)


def combined(geometries: Iterable[Any]) -> Any:
    return polygonal(unary_union(list(geometries)))


def bbox_geometry(bbox: list[float] | tuple[float, ...]) -> Any:
    if len(bbox) != 4 or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("Expected finite ordered WGS84 [west, south, east, north] bounds")
    # EPSG:6933 has straight parallels and meridians, so these bbox edges are exact.
    return project_geometry(box(*bbox), "EPSG:4326", AREA_CRS)


@dataclass
class FloodObservation:
    classes: dict[int, Any]
    flood: Any
    permanent_water: Any
    unknown: Any
    aoi: Any
    original_feature_count: int
    repaired_feature_count: int
    flood_permanent_overlap_m2: float

    @property
    def flood_area_km2(self) -> float:
        return self.flood.area / 1_000_000


def analyze_observation(collection: dict[str, Any], bbox: list[float]) -> FloodObservation:
    """Clip and union one acquisition's source class polygons in EPSG:6933."""
    if collection.get("type") != "FeatureCollection" or not collection.get("features"):
        raise ValueError("Expected a nonempty GeoJSON FeatureCollection")
    if collection.get("exceededTransferLimit"):
        raise ValueError("Source response is truncated")
    aoi = bbox_geometry(bbox)
    by_class: dict[int, list[Any]] = {}
    repaired = 0
    for feature in collection["features"]:
        classification = feature.get("properties", {}).get("class")
        if classification not in CLASS_LABELS:
            raise ValueError(f"Unexpected NRCan class: {classification!r}")
        if not feature.get("geometry"):
            raise ValueError("Source feature is missing geometry")
        source = shape(feature["geometry"])
        repaired += not source.is_valid
        projected = polygonal(project_geometry(polygonal(source), "EPSG:4326", AREA_CRS))
        clipped = polygonal(projected.intersection(aoi))
        if not clipped.is_empty:
            by_class.setdefault(classification, []).append(clipped)
    classes = {classification: combined(geometries) for classification, geometries in by_class.items()}
    permanent = classes.get(1, GeometryCollection())
    unknown = classes.get(6, GeometryCollection())
    flood = combined(classes.get(classification, GeometryCollection()) for classification in FLOOD_CLASSES)
    overlap = flood.intersection(permanent).area
    # Explicit conservative exclusion even if source classification boundaries overlap.
    flood = polygonal(flood.difference(permanent).difference(unknown))
    if not flood.is_valid or flood.area > aoi.area + 0.01:
        raise ValueError("Flood geometry failed clipping/validity checks")
    return FloodObservation(classes, flood, permanent, unknown, aoi, len(collection["features"]), repaired, overlap)


def display_feature(geometry: Any, properties: dict[str, Any], tolerance_m: float = 2) -> dict[str, Any] | None:
    if geometry.is_empty:
        return None
    display = polygonal(geometry.simplify(tolerance_m, preserve_topology=True))
    geographic = project_geometry(display, AREA_CRS, "EPSG:4326")
    return {"type": "Feature", "properties": properties, "geometry": mapping(geographic)}


def observation_geojson(observation: FloodObservation, label: str) -> dict[str, Any]:
    features = []
    for geometry, kind, color in [
        (observation.permanent_water, "Permanent water (excluded)", CLASS_COLORS[1]),
        (observation.flood, "Mapped flood (permanent water excluded)", CLASS_COLORS[2]),
        (observation.unknown, "Unknown state (excluded)", CLASS_COLORS[6]),
    ]:
        feature = display_feature(geometry, {"label": f"{label} — {kind}", "color": color, "fill_color": color, "classification": kind})
        if feature:
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def build(directory: Path) -> dict[str, Any]:
    """Rebuild deterministic derived outputs in a directory containing raw/."""
    from scripts.build_flood_data import build as build_outputs
    return build_outputs(directory)
