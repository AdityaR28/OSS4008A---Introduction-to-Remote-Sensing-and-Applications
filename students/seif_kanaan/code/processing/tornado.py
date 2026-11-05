"""Descriptive geometry analysis of the real NTP Didsbury survey centreline.

No path interpolation, buffer footprint, satellite classification or impact model.
Distances are WGS84 ellipsoidal geodesics between supplied vertices (GeographicLib).
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from datetime import date
from pathlib import Path

from geographiclib.geodesic import Geodesic

EVENT_ID = "tornado-didsbury-2023"
LICENSE = "CC BY-NC 4.0 — noncommercial use only; commercial use requires NTP permission"
ATTRIBUTION = "Northern Tornadoes Project, Western University; NTP Storm Survey Team"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
DERIVED_FILES = ("track.geojson", "endpoints.geojson", "endpoint-baseline.geojson", "event.json", "checks.json")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sources(directory: Path) -> list[dict]:
    """Verify every frozen response before reading its geometry or metadata."""
    acquisition = read_json(directory / "raw/acquisition.json")
    inputs = acquisition["inputs"]
    required = {"raw/track.json", "raw/summary.json", "raw/track-item.json", "raw/summary-item.json"}
    if len(inputs) != len(required) or {entry["file"] for entry in inputs} != required:
        raise ValueError("Unexpected or missing tornado source snapshots")
    for entry in inputs:
        path = directory / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Source snapshot hash mismatch: {entry['file']}")
    for filename in ("track-item.json", "summary-item.json"):
        item = read_json(directory / "raw" / filename)
        if item.get("orgId") != "rGKxabTU9mcXMw7k" or item.get("access") != "public":
            raise ValueError("Source must be the public Western University NTP item")
        if "Attribution-NonCommercial 4.0" not in item.get("licenseInfo", ""):
            raise ValueError("NTP source licence changed; review before rebuilding")
    return inputs


def one_feature(response: dict) -> dict:
    if response.get("error") or response.get("exceededTransferLimit"):
        raise ValueError("Incomplete or failed ArcGIS response")
    features = response.get("features", [])
    if len(features) != 1:
        raise ValueError("Expected exactly one matching event feature")
    return features[0]


def coordinates(response: dict) -> list[list[float]]:
    feature = one_feature(response)
    if response.get("spatialReference", {}).get("wkid") != 4326:
        raise ValueError("Expected WGS84 longitude/latitude response")
    if feature.get("attributes", {}).get("Event_Name") != "Didsbury":
        raise ValueError("Unexpected tornado track event")
    paths = feature.get("geometry", {}).get("paths", [])
    # This pilot measures one continuous supplied centreline. Do not silently join gaps.
    if len(paths) != 1 or len(paths[0]) < 2:
        raise ValueError("Expected one continuous surveyed track with at least two vertices")
    points = paths[0]
    for point in points:
        if len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
            raise ValueError("Invalid track coordinate")
        if not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90:
            raise ValueError("Coordinate outside WGS84 range")
    return points


def distance_m(first: list[float], second: list[float]) -> float:
    """GeoJSON lon,lat order converted explicitly to GeographicLib lat,lon."""
    return Geodesic.WGS84.Inverse(first[1], first[0], second[1], second[0])["s12"]


def analyze(track_response: dict, summary_response: dict) -> dict:
    points = coordinates(track_response)
    attributes = one_feature(summary_response)["attributes"]
    if attributes.get("event_name") != "Didsbury" or attributes.get("event_type") != "tornado_over_land":
        raise ValueError("Unexpected tornado survey summary")
    event_date = date(attributes["Year"], attributes["month"], attributes["day"]).isoformat()
    if event_date != "2023-07-01":
        raise ValueError("Summary does not match the selected event date")
    if attributes.get("classification_status") != "final_classification_completed":
        raise ValueError("This pilot requires a finalized NTP event classification")
    reported_km = attributes["track_length"]
    width_m = attributes["max_path_width"]
    if not isinstance(reported_km, (int, float)) or not math.isfinite(reported_km) or reported_km <= 0:
        raise ValueError("Invalid reported track length")
    if not isinstance(width_m, (int, float)) or not math.isfinite(width_m) or width_m <= 0:
        raise ValueError("Invalid reported maximum width")
    length = math.fsum(distance_m(a, b) for a, b in zip(points, points[1:]))
    endpoint = distance_m(points[0], points[-1])
    if endpoint <= 0 or length < endpoint - 1e-6:
        raise ValueError("Degenerate or inconsistent track")
    return {
        "coordinates": points,
        "bbox": [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)],
        "event_date": event_date,
        "vertex_count": len(points),
        "track_length_km": length / 1000,
        "endpoint_distance_km": endpoint / 1000,
        "sinuosity": length / endpoint,
        "reported_track_length_km": reported_km,
        "reported_max_width_m": width_m,
        "difference_from_reported_km": length / 1000 - reported_km,
        "rating": attributes["damage"].upper(),
        "track_objectid": one_feature(track_response)["attributes"]["OBJECTID"],
        "summary_objectid": attributes["objectid"],
    }


def build(directory: Path) -> dict:
    """Build deterministic derived files from checked-in, verified public snapshots."""
    inputs = validate_sources(directory)
    result = analyze(read_json(directory / "raw/track.json"), read_json(directory / "raw/summary.json"))
    points = result["coordinates"]
    attribution = f"{ATTRIBUTION}. {LICENSE}. Geometry converted to GeoJSON; WGS84 distances calculated by Canadian Disasters Monitor."
    track = {
        "type": "FeatureCollection", "name": "NTP Didsbury surveyed centreline",
        "attribution": attribution, "license": LICENSE_URL,
        "features": [{"type": "Feature", "properties": {"label": f"NTP surveyed centreline · {result['track_length_km']:.2f} km calculated", "source_objectid": result["track_objectid"], "color": "#8b5cf6"}, "geometry": {"type": "LineString", "coordinates": points}}],
    }
    endpoints = {
        "type": "FeatureCollection", "attribution": attribution, "license": LICENSE_URL,
        "features": [{"type": "Feature", "properties": {"label": label, "color": color, "radius": 6}, "geometry": {"type": "Point", "coordinates": point}}
                     for label, color, point in [("Start of supplied NTP centreline", "#16a34a", points[0]), ("End of supplied NTP centreline", "#dc2626", points[-1])]],
    }
    # A connection between endpoints is only a geometric baseline for sinuosity.
    chord = {
        "type": "FeatureCollection", "attribution": attribution, "license": LICENSE_URL,
        "features": [{"type": "Feature", "properties": {"label": "Endpoint separation baseline (not a tornado path)", "color": "#64748b"}, "geometry": {"type": "LineString", "coordinates": [points[0], points[-1]]}}],
    }
    reports = {
        "status": "automated_checks_passed",
        "scope": "Source identity/hash, finite/ranged coordinates, single continuous geometry, ellipsoidal distance consistency, reported-data comparison. Not independent field validation.",
        "inputs_verified": len(inputs), "source_features": 1, "result": {k: v for k, v in result.items() if k != "coordinates"},
        "comparison": "The downloaded mapped centreline and NTP reported track length differ. Both are retained without adjusting geometry or treating the difference as positional accuracy.",
    }
    event = make_event(result, inputs, read_json(directory / "raw/acquisition.json")["retrieved_at"])
    outputs = {"track.geojson": track, "endpoints.geojson": endpoints, "endpoint-baseline.geojson": chord, "event.json": event}
    reports["outputs"] = [{"file": name, "sha256": hashlib.sha256(json_bytes(value)).hexdigest()} for name, value in outputs.items()]
    outputs["checks.json"] = reports
    for name, value in outputs.items():
        (directory / name).write_bytes(json_bytes(value))
    return reports


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def check(directory: Path) -> dict:
    """Recompute into a temporary folder and reject any stale/tampered outputs.

    Never modifies the supplied directory, including when a comparison fails.
    Source checks run before copying. This is a numerical/serialization release
    gate, not independent field validation of the source.
    """
    validate_sources(directory)
    with tempfile.TemporaryDirectory(prefix="tornado-check-") as temporary:
        target = Path(temporary)
        shutil.copytree(directory / "raw", target / "raw")
        report = build(target)
        for filename in DERIVED_FILES:
            if (target / filename).read_bytes() != (directory / filename).read_bytes():
                raise ValueError(f"Derived tornado output differs from frozen-source recomputation: {filename}")
    return {"status": "automated_checks_passed", "derived_files_verified": len(DERIVED_FILES), "outputs": report["outputs"]}


def make_event(result: dict, inputs: list[dict], generated_at: str) -> dict:
    metric = lambda label, value, unit, precision, description: {"label": label, "value": value, "unit": unit, "precision": precision, "description": description}
    layer = lambda id, label, url, color, description: {"id": id, "label": label, "kind": "geojson", "url": url, "description": description, "style": {"color": color, "fillColor": color, "fillOpacity": 0.8, "weight": 3}, "legend": [{"color": color, "label": label}]}
    layers = [layer("surveyed_track", "NTP surveyed centreline", "track.geojson", "#8b5cf6", "Actual complete line supplied by NTP, with all vertices retained. Line thickness is a display symbol, not the tornado's width."), layer("surveyed_endpoints", "Supplied track endpoints", "endpoints.geojson", "#16a34a", "First and last actual line vertices. The green start and red end markers are derived from the geometry."), layer("endpoint_baseline", "Endpoint separation baseline", "endpoint-baseline.geojson", "#64748b", "Geometric endpoint connection for interpreting sinuosity, not a surveyed tornado path. Leaflet renders the two-point line in the map projection; its metric is computed on the ellipsoid.")]
    layers[1]["legend"] = [{"color": "#16a34a", "label": "Start of supplied centreline"}, {"color": "#dc2626", "label": "End of supplied centreline"}]
    return {
        "schema_version": "1.0", "id": EVENT_ID, "hazard": "tornado", "title": "Didsbury tornado · surveyed track", "location": "Didsbury–Carstairs, Alberta", "event_date": result["event_date"],
        "analysis_kind": "surveyed_track",
        "summary": f"How much does the mapped tornado path bend compared with its endpoint separation? WGS84 analysis of the real Northern Tornadoes Project centreline for the July 1, 2023 {result['rating']} tornado. This is a survey geometry analysis; it contains no new satellite damage classification. NTP data: CC BY-NC 4.0, noncommercial use only.",
        "bbox": result["bbox"],
        "temporal": [{"label": "Tornado occurrence", "start": "2023-07-01", "end": "2023-07-01", "description": "NTP summary: 13:45 MDT / 19:45 UTC. Calendar date is read from NTP year/month/day fields; ArcGIS Date_ display time is not used."}, {"label": "NTP ground/drone investigation", "start": "2023-07-01", "end": "2023-07-03", "description": "Survey dates reported in the NTP event summary; NTP also reports satellite review. This project does not ingest NTP's satellite scenes."}],
        "metrics": [
            metric("Calculated centreline length", result["track_length_km"], "km", 2, "Sum of WGS84 ellipsoidal geodesic distances between every vertex in the supplied NTP line."),
            metric("Endpoint separation", result["endpoint_distance_km"], "km", 2, "Shortest WGS84 surface distance between the supplied line endpoints; not an alternative damage path."),
            metric("Track sinuosity", result["sinuosity"], "ratio", 3, "Calculated centreline length divided by endpoint separation. A value above 1 reflects mapped bends, including a small loop; not tornado intensity."),
            metric("NTP reported track length", result["reported_track_length_km"], "km", 1, "Reported attribute in the same NTP survey database. It differs from the downloaded line; this is a descriptive consistency comparison, not validation against independent ground truth."),
            metric("NTP reported maximum width", result["reported_max_width_m"], "m", 0, "Maximum assessed width from the event summary, not mean width or a constant buffer; no affected-area estimate is calculated."),
            metric("Mapped minus reported length", result["difference_from_reported_km"], "km", 2, "Signed difference between calculated geometry length and NTP summary track_length. Source geometry is preserved without forcing agreement."),
        ],
        "layers": layers,
        "methods": [{"title": "Frozen authoritative survey data", "description": "Download a single Didsbury centreline and the matching 2023-07-01 event summary directly from Western University's public NTP ArcGIS services. Keep exact JSON response bytes, item licence/attribution metadata, query URLs, timestamps and SHA-256 hashes."}, {"title": "Ellipsoidal track geometry", "description": "Use GeographicLib 2.1 WGS84 inverse geodesics for consecutive line vertices, sum the segment distances, then divide by endpoint distance for sinuosity. Preserve every supplied vertex and the mapped small loop. No Web Mercator Shape__Length field is treated as a ground distance."}, {"title": "Bounded comparison", "description": "Retain NTP's reported length and maximum width alongside calculated length. Differences can arise from source representation, digitization or assessment definitions; the data do not identify which. A single source compared with itself cannot establish map accuracy."}],
        "limitations": ["Surveyed centreline only: no new satellite change detection, damage mask, area, building exposure or physical impact estimate is produced.", "The map line and endpoint symbols have display thickness/size; they do not show a complete damage footprint or geolocation uncertainty.", "NTP reports EF4 and a maximum assessed width of 620 m. These source assessments do not imply uniform intensity or width along the line.", "Unmapped land is unknown here, not undamaged. A centreline does not describe every affected location, debris trajectory or wind extent.", "Calculated length differs from NTP's reported 15.3 km; this discrepancy is exposed rather than corrected. Source metadata do not provide positional accuracy or an uncertainty interval.", "This is one documented event, not a complete tornado inventory, risk map, prediction or emergency warning.", "NTP data and derivatives are CC BY-NC 4.0. Noncommercial use only; commercial use requires permission from NTP. Attribution and changes are recorded; no endorsement is implied."],
        "sources": [{"label": "NTP Damage Tracks (Open Data)", "url": "https://www.arcgis.com/home/item.html?id=3c401a84ff864d42adcb1ec6fcc79698", "attribution": ATTRIBUTION + "; geometry converted and geodesic metrics calculated by Canadian Disasters Monitor", "license": LICENSE}, {"label": "NTP Didsbury event summary", "url": "https://experience.arcgis.com/experience/04690b8fff5e46d1a35bb6dcfb5f70f7", "attribution": ATTRIBUTION, "license": LICENSE}, {"label": "NTP Event Summaries public service", "url": "https://www.arcgis.com/home/item.html?id=bc14a794dd764b42ba4f025e12910c43", "attribution": ATTRIBUTION, "license": LICENSE}, {"label": "Creative Commons Attribution-NonCommercial 4.0", "url": LICENSE_URL, "attribution": "Creative Commons; NTP source licence", "license": "Licence terms"}],
        "provenance": {"generated_at": generated_at, "inputs": inputs, "software": "processing/tornado.py; scripts/build_tornado.py; GeographicLib 2.1 WGS84 geodesics. Frozen snapshot timestamp retained for deterministic offline rebuilds.", "report_url": "checks.json"},
        "checks": {"status": "automated_checks_passed", "summary": "Source response hashes and public Western/NTP identity verified; exact single continuous geometry, finite WGS84 coordinates, event date/type and ellipsoidal distance checks passed. These checks do not establish independent field or satellite accuracy."},
    }
