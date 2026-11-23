"""Session-owned results for the shared Python dashboard."""
import hashlib
import io
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile, gettempdir


EXAMPLES = {
    "Halifax": "ships_2025-07-26_ccc66456ac88",
    "Vancouver": "ships_2025-08-12_f33c2c479429",
}
PRESETS = {
    "Halifax": [-63.58, 44.615, -63.545, 44.64],
    "Vancouver": [-123.115, 49.285, -123.08, 49.31],
    "Montreal": [-73.55, 45.53, -73.515, 45.555],
    "Churchill": [-94.23, 58.765, -94.195, 58.79],
    "Iqaluit": [-68.54, 63.725, -68.505, 63.75],
    "Prince Rupert": [-130.345, 54.29, -130.31, 54.315],
}
CHECKPOINT_SHA256 = "aa57b80886d48b592209a09e65cbd6b1b657585add4d7dd11f76867268201721"
INFERENCE_LOCK = threading.Lock()


def load_example(project, location):
    """Read immutable published results without updating their measurements."""
    stem = EXAMPLES[location]
    folder = Path(project) / "results"
    files = {extension: (folder / f"{stem}.{extension}").read_bytes()
             for extension in ["json", "geojson", "csv", "png"]}
    metadata = json.loads(files["json"])
    features = json.loads(files["geojson"])
    if metadata["detection_count"] != len(features["features"]):
        raise ValueError("Saved prediction count differs from its metadata.")
    return {"label": location + " saved example", "metadata": metadata, "geojson": features,
            "files": files, "stem": stem, "kind": "saved"}


def drawing_bounds(feature):
    if not isinstance(feature, dict) or feature.get("geometry", {}).get("type") != "Polygon":
        raise ValueError("Draw a rectangle on the map.")
    ring = feature["geometry"]["coordinates"][0]
    longitudes, latitudes = zip(*ring)
    bounds = [min(longitudes), min(latitudes), max(longitudes), max(latitudes)]
    validate_selection(bounds)
    return bounds


def validate_selection(bounds):
    """Apply the existing geographic constraints without loading the model."""
    import math
    west, south, east, north = map(float, bounds)
    if not all(math.isfinite(value) for value in [west, south, east, north]):
        raise ValueError("Coordinates must be finite.")
    if not (-141 <= west < east <= -52 and 41 <= south < north <= 84):
        raise ValueError("Choose a small area within the Canadian map extent.")
    width = (east - west) * 111.32 * math.cos(math.radians((south + north) / 2))
    height = (north - south) * 111.32
    if width * height > 16 or max(width, height) > 6 or min(width, height) < .1:
        raise ValueError("Choose an area of 100 m to 6 km per side and at most 16 km².")
    return [west, south, east, north]


def initialize_imagery(project_id, service_account_json=""):
    import ee
    if service_account_json:
        try:
            account = json.loads(service_account_json)
            credentials = ee.ServiceAccountCredentials(account["client_email"], key_data=service_account_json)
        except (ValueError, KeyError):
            raise ValueError("The host's Earth Engine service account needs attention.") from None
        ee.Initialize(credentials, project=project_id)
    elif os.environ.get("ALLOW_COLAB_AUTH") == "1":
        ee.Initialize(project=project_id)
    else:
        raise ValueError("Live imagery is awaiting the host's Earth Engine connection.")


def load_model(project):
    from ship_backend import load_detector
    path = Path(project) / "results/models/vessel_detector.pt"
    if not path.is_file():
        import requests
        cache = Path(gettempdir()) / "ships-model-cache"
        cache.mkdir(exist_ok=True)
        path = cache / "vessel_detector.pt"
        if not path.is_file():
            url = ("https://raw.githubusercontent.com/AdityaR28/"
                   "OSS4008A---Introduction-to-Remote-Sensing-and-Applications/"
                   "c9fe49ed2d3a9b31801d65baac659384ebe0df73/"
                   "students/oritse-tsegbemi_eyito/results/models/vessel_detector.pt")
            temporary = None
            try:
                with requests.get(url, stream=True, timeout=(10, 90)) as response:
                    response.raise_for_status()
                    digest = hashlib.sha256()
                    size = 0
                    with NamedTemporaryFile(dir=cache, delete=False) as handle:
                        temporary = Path(handle.name)
                        for chunk in response.iter_content(1024 * 1024):
                            size += len(chunk)
                            if size > 42198971:
                                raise ValueError("The checkpoint download exceeded its expected size.")
                            digest.update(chunk)
                            handle.write(chunk)
                    if size != 42198971 or digest.hexdigest() != CHECKPOINT_SHA256:
                        raise ValueError("The deployed checkpoint could not be verified.")
                    temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("The deployed checkpoint could not be verified.")
    return load_detector(path, device="cpu")[:2]


def run_live(project, model, device, bounds, start, end, confidence, cloud_max,
             project_id, service_account_json=""):
    """Reuse the tested detector; each request owns its export directory."""
    from ship_map import fetch_observation, detect_raster, geographic_results, save_detection
    validate_selection(bounds)
    if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
        raise ValueError("The end date must follow the start date and is exclusive.")
    if not .05 <= confidence <= .95 or not 0 <= cloud_max <= 100:
        raise ValueError("Choose a supported confidence and cloud limit.")
    with INFERENCE_LOCK:
        initialize_imagery(project_id, service_account_json)
        rgb, valid, transform, crs, metadata = fetch_observation(bounds, start, end, cloud_max)
        started = time.perf_counter()
        boxes, scores = detect_raster(model, device, rgb, valid, confidence)
        metadata.update(threshold=confidence, detection_count=len(boxes),
                        inference_seconds=time.perf_counter() - started,
                        model_checkpoint_sha256=CHECKPOINT_SHA256,
                        run_utc=datetime.now(timezone.utc).isoformat(),
                        canadian_accuracy_measured=False)
        features = geographic_results(boxes, scores, transform, crs, metadata)
        with TemporaryDirectory(prefix="ships-result-") as directory:
            stem = save_detection(directory, rgb, boxes, scores, features, metadata)
            files = {extension: (Path(directory) / "results" / f"{stem}.{extension}").read_bytes()
                     for extension in ["json", "geojson", "csv", "png"]}
    return {"label": "Live satellite analysis", "metadata": metadata, "geojson": features,
            "files": files, "stem": stem, "kind": "live"}


def result_context(result, filtered_count, province, query, selection):
    metadata = result["metadata"]
    scores = [float(feature["properties"]["confidence"]) for feature in result["geojson"]["features"]]
    return {
        "displayed_run": {"label": result["label"], "kind": result["kind"],
                          "acquired_utc": metadata["acquired_utc"], "scene": metadata["image_id"],
                          "predicted_ships": len(scores), "threshold": metadata["threshold"],
                          "mean_model_confidence": sum(scores) / len(scores) if scores else None,
                          "clear_pixel_fraction": metadata.get("clear_pixel_fraction"),
                          "bounds": metadata["bounds_wgs84"]},
        "facility_filter": {"province": province, "query": query, "matching_references": filtered_count},
        "requested_area": selection,
        "requested_area_has_been_processed": result["kind"] == "live" and selection == metadata["bounds_wgs84"],
        "capabilities": ["show_example", "filter_ports"],
    }


def map_overlay(result):
    """Reproject the actual preview grid before putting it on a geographic map."""
    import numpy as np
    import rasterio.warp
    from rasterio.transform import Affine, array_bounds
    from PIL import Image
    metadata = result["metadata"]
    pixels = np.array(Image.open(io.BytesIO(result["files"]["png"])).convert("RGB"))
    height, width = pixels.shape[:2]
    transform = Affine(*metadata["raster_transform"])
    extent = array_bounds(height, width, transform)
    destination_transform, out_width, out_height = rasterio.warp.calculate_default_transform(
        metadata["raster_crs"], "EPSG:4326", width, height, *extent)
    destination = np.zeros((3, out_height, out_width), dtype=np.uint8)
    rasterio.warp.reproject(
        pixels.transpose(2, 0, 1), destination, src_transform=transform, src_crs=metadata["raster_crs"],
        dst_transform=destination_transform, dst_crs="EPSG:4326",
        resampling=rasterio.warp.Resampling.nearest)
    west, south, east, north = array_bounds(out_height, out_width, destination_transform)
    output = io.BytesIO()
    Image.fromarray(destination.transpose(1, 2, 0)).save(output, format="PNG")
    return output.getvalue(), [[south, west], [north, east]]
