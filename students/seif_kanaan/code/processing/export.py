"""Authenticated exact-grid pixel tiles, durable TIFFs and transparent PNGs.

Use ee.data.computePixels directly instead of thumbnail download URLs. Each
request has an explicit PixelGrid and at most 512 x 512 pixels by default.
Tiles are stitched with raster windows without resampling the analysis grid.
https://developers.google.com/earth-engine/reference/rest/v1/projects.image/computePixels
The API permits 48 MB; this pilot retains stricter 24 MB full-array / 32 MB
response caps and uses much smaller individual requests.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

LOGGER = logging.getLogger(__name__)
NODATA = -9999.0
MAX_REQUEST_BYTES = 24_000_000
MAX_TRANSFER_BYTES = 32_000_000
EXPORT_TILE_SIZE = 512
NBR_COLORS = ["#895d50", "#d1ba91", "#f1eed0", "#a5bd92", "#246957"]
DNBR_COLORS = ["#416ca1", "#ece8cf", "#dfb251", "#d87836", "#a83235"]


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int

    def pixel_grid(self):
        affine = self.transform
        return {"crsCode": self.crs,
                "dimensions": {"width": self.width, "height": self.height},
                "affineTransform": {"scaleX": affine.a, "shearX": affine.b,
                                    "translateX": affine.c, "shearY": affine.d,
                                    "scaleY": affine.e, "translateY": affine.f}}

    def tiles(self, tile_size=EXPORT_TILE_SIZE):
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or not 1 <= tile_size <= 1024:
            raise ValueError("Export tile size must be an integer from 1 to 1024.")
        for row in range(0, self.height, tile_size):
            for column in range(0, self.width, tile_size):
                affine = self.transform
                offset = Affine(affine.a, affine.b, affine.c+affine.a*column+affine.b*row,
                                affine.d, affine.e, affine.f+affine.d*column+affine.e*row)
                yield row, column, Grid(
                    self.crs, offset,
                    min(tile_size, self.width-column), min(tile_size, self.height-row))

    def check_size(self, *, bands: int, bytes_per_sample: int):
        if not (0 < self.width <= 10000 and 0 < self.height <= 10000):
            raise ValueError("Download grid exceeds Earth Engine's 10,000-pixel dimension limit.")
        size = self.width * self.height * bands * bytes_per_sample
        if size > MAX_REQUEST_BYTES:
            raise ValueError(f"Uncompressed request is {size:,} bytes; limit {MAX_REQUEST_BYTES:,}. Narrow AOI or increase display pixel size.")


def grid_for_bbox(bbox, crs, pixel_size):
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    left, top = math.floor(west/pixel_size)*pixel_size, math.ceil(north/pixel_size)*pixel_size
    right, bottom = math.ceil(east/pixel_size)*pixel_size, math.floor(south/pixel_size)*pixel_size
    return Grid(crs, Affine(pixel_size, 0, left, 0, -pixel_size, top),
                int(round((right-left)/pixel_size)), int(round((top-bottom)/pixel_size)))


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class RasterExportError(RuntimeError):
    """A useful provider failure with credentials/URLs removed."""

    def __init__(self, message, *, transient=False):
        super().__init__(message)
        self.transient = transient


def safe_provider_error(error):
    message = str(error)
    message = re.sub(r"https?://[^\s<>\"']+", "[URL redacted]", message)
    message = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", message)
    message = re.sub(r"(?i)(\b(?:access_token|refresh_token|token|api_key|signature)\b\s*[=:]\s*[\"']?)[^\s\"',;]+", r"\1[redacted]", message)
    return " ".join(message.split())[:2000] or type(error).__name__


def _read_tiff(dataset, grid, *, bands, dtype):
    if dataset.crs != rasterio.crs.CRS.from_string(grid.crs):
        raise ValueError("Export CRS does not match requested CRS.")
    if dataset.width != grid.width or dataset.height != grid.height or dataset.count != bands:
        raise ValueError("Unexpected raster size or band count.")
    if not dataset.transform.almost_equals(grid.transform, precision=1e-6):
        raise ValueError("Export raster transform differs from the requested grid.")
    if any(value != dtype for value in dataset.dtypes):
        raise ValueError(f"Unexpected raster dtype: {dataset.dtypes}")
    values = dataset.read()
    if not np.isfinite(values).all():
        raise ValueError("Raster contains nonfinite values.")
    west, south, east, north = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21)
    return values, [[south, west], [north, east]]


def validate_tiff(path, grid, *, bands, dtype):
    with rasterio.open(path) as dataset:
        return _read_tiff(dataset, grid, bands=bands, dtype=dtype)


def _pixel_tile(image, grid, *, bands, dtype, compute_pixels, label):
    grid.check_size(bands=bands, bytes_per_sample=np.dtype(dtype).itemsize)
    params = {"expression": image, "fileFormat": "GEO_TIFF", "grid": grid.pixel_grid()}
    for attempt in range(3):
        try:
            raw = compute_pixels(params)
            if not isinstance(raw, (bytes, bytearray)) or not raw:
                raise ValueError("Earth Engine did not return GeoTIFF bytes.")
            if len(raw) > MAX_TRANSFER_BYTES:
                raise ValueError("Raster response exceeds the configured size cap.")
            if raw.lstrip().startswith(b"{"):
                payload = json.loads(raw)
                raise ValueError(str(payload.get("error", {}).get("message", "Earth Engine returned JSON instead of a raster.")))
            with MemoryFile(raw) as memory, memory.open() as dataset:
                values, _ = _read_tiff(dataset, grid, bands=bands, dtype=dtype)
            return values
        except Exception as error:
            detail = safe_provider_error(error)
            status = getattr(error, "status_code", None) or getattr(getattr(error, "resp", None), "status", None)
            if status:
                detail = f"HTTP {status}: {detail}"
            lower = detail.lower()
            resource_limit = any(text in lower for text in (
                "memory limit", "computation timed out", "deadline exceeded",
                "too many pixels", "total request size", "request exceeds", "reprojection output too large"))
            if resource_limit and max(grid.width, grid.height) > 64:
                size = max(64, math.ceil(max(grid.width, grid.height)/2))
                LOGGER.warning("%s: reducing tile dimensions after provider resource limit: %s", label, detail)
                values = np.empty((bands, grid.height, grid.width), dtype=dtype)
                for row, column, child in grid.tiles(size):
                    values[:, row:row+child.height, column:column+child.width] = _pixel_tile(
                        image, child, bands=bands, dtype=dtype, compute_pixels=compute_pixels,
                        label=f"{label}, sub-tile row={row} col={column}")
                return values
            transient = status in (429, 500, 502, 503, 504) or bool(re.search(r"\b(?:429|500|502|503|504)\b", lower)) or any(text in lower for text in (
                "temporarily unavailable",
                "too many requests", "connection reset", "connection aborted", "connection timed out"))
            if transient and attempt < 2:
                LOGGER.warning("%s: temporary provider failure; retry %d/2: %s", label, attempt+1, detail)
                time.sleep(2**attempt)
                continue
            raise RasterExportError(f"Earth Engine pixel export failed for {label}: {detail}", transient=transient) from None


def download_image(image, destination, grid, *, bands, dtype, tile_size=EXPORT_TILE_SIZE,
                   workers=1, compute_pixels=None):
    """Compute small authenticated tiles, then atomically assemble the exact grid.

    Sequential requests are the conservative default. Tests may inject a pixel
    provider; that does not establish live Earth Engine success.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("Export workers must be an integer from 1 to 4.")
    grid.check_size(bands=bands, bytes_per_sample=np.dtype(dtype).itemsize)
    if compute_pixels is None:
        import ee
        compute_pixels = ee.data.computePixels
    destination = Path(destination)
    temporary = destination.with_name(destination.name+".part")
    tiles = list(grid.tiles(tile_size))
    LOGGER.info("%s: exporting %d exact-grid tiles via authenticated computePixels", destination.name, len(tiles))

    def fetch(tile):
        row, column, child = tile
        return row, column, child, _pixel_tile(
            image, child, bands=bands, dtype=dtype, compute_pixels=compute_pixels,
            label=f"{destination.name}, tile row={row} col={column} ({child.width}x{child.height})")

    try:
        with rasterio.open(temporary, "w", driver="GTiff", width=grid.width, height=grid.height,
                           count=bands, dtype=dtype, crs=grid.crs, transform=grid.transform) as output:
            def write(result, completed):
                row, column, child, values = result
                output.write(values, window=Window(column, row, child.width, child.height))
                if completed == 1 or completed % 4 == 0 or completed == len(tiles):
                    LOGGER.info("%s: tile %d/%d complete", destination.name, completed, len(tiles))
            if workers == 1:
                for completed, tile in enumerate(tiles, 1):
                    write(fetch(tile), completed)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    pending = [executor.submit(fetch, tile) for tile in tiles]
                    try:
                        for completed, future in enumerate(as_completed(pending), 1):
                            write(future.result(), completed)
                    except Exception:
                        for future in pending:
                            future.cancel()
                        raise
        result = validate_tiff(temporary, grid, bands=bands, dtype=dtype)
        temporary.replace(destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def tiff_to_png(tiff, png, grid, *, allow_empty=False):
    array, bounds = validate_tiff(tiff, grid, bands=4, dtype="uint8")
    alpha = array[3]
    if not np.isin(alpha, [0, 255]).all():
        raise ValueError("Expected a binary alpha validity channel.")
    if not allow_empty and not alpha.any():
        raise ValueError("Display has no valid pixels; refusing an empty scientific layer.")
    # RGB may validly be black; alpha alone determines validity.
    rgba = np.moveaxis(array, 0, -1)
    rgba[alpha == 0, :3] = 0
    Image.fromarray(rgba).save(png, optimize=True)
    with Image.open(png, formats=["PNG"]) as check:
        if check.mode != "RGBA" or check.size != (grid.width, grid.height):
            raise RuntimeError("PNG write verification failed.")
    return bounds


def _rgba(ee, image, vis):
    visible = image.visualize(**vis)
    # Display validity is binary even when EE retains fractional source/AOI
    # masks. Preserve any nonzero support, including valid black RGB; retain
    # zero support as transparent. Analytical mask weights are not modified.
    alpha = (visible.mask().reduce(ee.Reducer.min()).unmask(0, sameFootprint=False)
             .gt(0).multiply(255).toByte().rename("alpha"))
    return visible.unmask(0, sameFootprint=False).addBands(alpha).toByte()


def check_display_export(ee, *, compute_pixels=None):
    """Tiny authenticated RGBA export exercising fractional masks, without data.

    One 2x2 GeoTIFF contains four RGBA groups with support 0, .25, .5 and 1.
    This tests the same graph and raster decoder as the real display exports
    before downloading large rasters. Constants are never published as layers.
    """
    grid = Grid("EPSG:32611", Affine(20, 0, 300000, 0, -20, 5500200), 2, 2)
    groups = []
    for fraction in (0, .25, .5, 1):
        image = (ee.Image.constant([0, 0, 0]).rename(["B4", "B3", "B2"])
                 .updateMask(ee.Image.constant(fraction)))
        groups.append(_rgba(ee, image, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 1}))
    if compute_pixels is None:
        compute_pixels = ee.data.computePixels
    values = _pixel_tile(ee.Image.cat(groups), grid, bands=16, dtype="uint8",
                         compute_pixels=compute_pixels, label="synthetic display self-check")
    expected = (0, 255, 255, 255)
    for group, alpha in enumerate(expected):
        if not (values[group*4:group*4+3] == 0).all() or not (values[group*4+3] == alpha).all():
            raise ValueError("Synthetic display self-check failed: black RGB/fractional validity was not preserved as binary alpha.")
    return {"status": "passed", "source_mask_values": [0, .25, .5, 1],
            "alpha_values": list(expected), "grid_pixels": 4,
            "note": "Constant images only; checks authenticated RGBA transport, not satellite observations."}


def export_layers(ee, analysis, config, directory, metrics):
    directory = Path(directory)
    layers_dir, tiffs_dir = directory / "layers", directory / "rasters"
    layers_dir.mkdir(parents=True)
    tiffs_dir.mkdir(parents=True, exist_ok=True)
    grid = grid_for_bbox(config.aoi_bbox, "EPSG:3857", config.display_pixel_size_m)
    rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3, "gamma": 1.3}
    nbr_vis = {"min": -1, "max": 1, "palette": NBR_COLORS}
    dnbr_vis = {"min": -0.3, "max": 0.8, "palette": DNBR_COLORS}
    rgb_legend = [{"color": "#ffffff", "label": "RGB reflectance, 0–0.30; gamma 1.3"}]
    nbr_legend = [{"color": NBR_COLORS[0], "label": "−1: low NBR"}, {"color": NBR_COLORS[-1], "label": "+1: high NBR"}]
    dnbr_legend = [{"color": DNBR_COLORS[i], "label": label} for i, label in enumerate(["≤ −0.30", "−0.025", "+0.25", "+0.525", "≥ +0.80"])]
    items = [
        ("pre_rgb", "Before • true colour", analysis.pre, rgb_vis, rgb_legend, "surface reflectance"),
        ("post_rgb", "After • true colour", analysis.post, rgb_vis, rgb_legend, "surface reflectance"),
        ("pre_nbr", "Before • NBR", analysis.pre_nbr, nbr_vis, nbr_legend, "NBR (unitless)"),
        ("post_nbr", "After • NBR", analysis.post_nbr, nbr_vis, nbr_legend, "NBR (unitless)"),
        ("dnbr", "Change • dNBR", analysis.dnbr, dnbr_vis, dnbr_legend, "dNBR (unitless)"),
        ("candidate", "Candidate burned area", analysis.candidate.selfMask(), {"min": 0, "max": 1, "palette": ["#e97439", "#e97439"]}, [{"color": "#e97439", "label": f"dNBR ≥ {config.threshold:g}"}], "candidate indicator"),
    ]
    layers = []
    for layer_id, label, image, vis, legend, units in items:
        tiff, png = tiffs_dir/f"{layer_id}_display.tif", layers_dir/f"{layer_id}.png"
        download_image(_rgba(ee, image, vis), tiff, grid, bands=4, dtype="uint8")
        bounds = tiff_to_png(tiff, png, grid, allow_empty=layer_id == "candidate" and metrics["candidate_area_ha"] == 0)
        layers.append({"id": layer_id, "label": label, "kind": "image", "url": f"layers/{png.name}",
                       "bounds": bounds, "legend": legend, "units": units, "sha256": sha256(png),
                       "display_crs": grid.crs, "display_pixel_size_map_m": config.display_pixel_size_m,
                       "width": grid.width, "height": grid.height, "resampling": "nearest (display only)"})
        # RGB(A) display TIFFs are intermediary encodings; retain the actual numerical TIFFs below.
        tiff.unlink()
    return layers


def export_analytical(ee, analysis, config, directory):
    """Download bounded single-band numerical data on the unchanged 20 m UTM grid."""
    from .event import ANALYSIS_CRS

    grid = grid_for_bbox(config.aoi_bbox, ANALYSIS_CRS, 20)
    directory = Path(directory)
    target = directory / "rasters"
    target.mkdir(exist_ok=True)
    items = {"pre_nbr": analysis.pre_nbr, "post_nbr": analysis.post_nbr,
             "dnbr": analysis.dnbr, "candidate": analysis.candidate,
             "pixel_area_m2": ee.Image.pixelArea().clip(analysis.aoi)}
    assets = []
    for name, image in items.items():
        path = target / f"{name}.tif"
        values, _ = download_image(image.toFloat().unmask(NODATA, sameFootprint=False), path, grid, bands=1, dtype="float32")
        valid = values != NODATA
        if not valid.any():
            raise ValueError(f"Analytical raster {name} contains no valid values.")
        if name in ("pre_nbr", "post_nbr", "dnbr"):
            limit = 2 if name == "dnbr" else 1
            if (np.abs(values[valid]) > limit + 1e-5).any():
                raise ValueError(f"Analytical {name} values exceed the expected range.")
        if name == "candidate" and not np.isin(values[valid], [0, 1]).all():
            raise ValueError("Candidate raster must contain only 0/1 within valid coverage.")
        # Set explicit nodata for ordinary GIS software; preserve returned grid and samples.
        with rasterio.open(path, "r+") as dataset:
            dataset.nodata = NODATA
        assets.append({"id": name, "url": f"rasters/{path.name}", "crs": grid.crs,
                       "scale_m": 20, "nodata": NODATA, "sha256": sha256(path)})
    return assets


def sanity_check_area(directory, metrics):
    """Independent client sum of the genuine 20 m exports, not field validation."""
    with rasterio.open(Path(directory)/"rasters/candidate.tif") as raster:
        candidate = raster.read(1, masked=True)
    with rasterio.open(Path(directory)/"rasters/pixel_area_m2.tif") as raster:
        pixel_area = raster.read(1, masked=True)
    selected = (candidate.filled(NODATA) == 1) & ~np.ma.getmaskarray(pixel_area)
    area = float(np.where(selected, pixel_area.filled(0), 0).sum(dtype=np.float64)) / 10000
    tolerance = max(1.0, metrics["candidate_area_ha"] * 0.01)
    difference = abs(area - metrics["candidate_area_ha"])
    if difference > tolerance:
        raise ValueError(f"Exported raster area differs from EE reducer by {difference:.2f} ha (allowed {tolerance:.2f}); inspect AOI boundary handling.")
    return {"status": "passed", "exported_candidate_area_ha": area,
            "absolute_difference_ha": difference, "tolerance_ha": tolerance,
            "note": "Client pixelArea sum; differences can arise at fractionally covered AOI edge pixels. This is implementation consistency, not independent fire validation."}


def write_manifest_atomic(directory, manifest):
    path = Path(directory) / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
