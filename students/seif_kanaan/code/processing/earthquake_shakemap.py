"""Bounded, offline analysis of a frozen USGS ShakeMap grid.

The product describes estimated shaking; it is never a damage classification.
Areas are WGS84 ellipsoidal areas of cells clipped to an explicit domain,
including water. Source grid nodes are assigned midpoint-bounded cells.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


NS = {"s": "http://earthquake.usgs.gov/eqcenter/shakemap"}
MAX_XML_BYTES = 40_000_000
AOI = (-135.0, 51.0, -129.0, 55.5)
MMI_BINS = (3.0, 4.0, 5.0, 6.0, 7.0)
MMI_COLORS = ("#d6e7ed", "#8dc4df", "#7bc8a4", "#f5dc62", "#ee9a4a", "#db5844")
MMI_LABELS = ("MMI < 3", "3 ≤ MMI < 4", "4 ≤ MMI < 5", "5 ≤ MMI < 6", "6 ≤ MMI < 7", "MMI ≥ 7")
SIGMA_BINS = (0.7, 0.9, 1.1)
SIGMA_COLORS = ("#e3dcef", "#b69fd0", "#8966b3", "#59367d")
SIGMA_LABELS = ("SD < 0.7", "0.7 ≤ SD < 0.9", "0.9 ≤ SD < 1.1", "SD ≥ 1.1 MMI")


@dataclass(frozen=True)
class ShakeGrid:
    lons: np.ndarray
    lats: np.ndarray  # descending: north to south
    fields: dict[str, np.ndarray]
    units: dict[str, str]
    attributes: dict[str, str]
    event: dict[str, str]


def read_grid(path: Path | str) -> ShakeGrid:
    """Read a regular north-up ShakeMap XML, rejecting invalid coordinates/data."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            payload = stream.read(MAX_XML_BYTES + 1)
    else:
        payload = path.read_bytes()
    if len(payload) > MAX_XML_BYTES:
        raise ValueError("ShakeMap XML exceeds the bounded input size")
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("DTD/entity declarations are not accepted")
    root = ET.fromstring(payload)
    spec = root.find("s:grid_specification", NS)
    event = root.find("s:event", NS)
    data = root.find("s:grid_data", NS)
    columns = root.findall("s:grid_field", NS)
    if spec is None or event is None or data is None or not columns:
        raise ValueError("Missing ShakeMap specification, event, fields or data")
    nx, ny = int(spec.attrib["nlon"]), int(spec.attrib["nlat"])
    if not (2 <= nx <= 10000 and 2 <= ny <= 10000 and nx * ny <= 2_000_000):
        raise ValueError("Invalid or excessive grid dimensions")
    indices = [int(field.attrib["index"]) for field in columns]
    if sorted(indices) != list(range(1, len(columns) + 1)):
        raise ValueError("Grid field indices must be unique and consecutive")
    ordered = sorted(columns, key=lambda field: int(field.attrib["index"]))
    names = [field.attrib["name"] for field in ordered]
    if len(set(names)) != len(names) or "LON" not in names or "LAT" not in names:
        raise ValueError("Grid fields must be unique and include LON/LAT")
    # fromstring can stop at malformed text, so validate the entire token count.
    tokens = (data.text or "").split()
    if len(tokens) != nx * ny * len(columns):
        raise ValueError("Grid dimensions do not match the numerical data")
    values = np.asarray(tokens, dtype=np.float64).reshape(ny, nx, len(columns))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite grid data")
    lon_grid, lat_grid = values[:, :, names.index("LON")], values[:, :, names.index("LAT")]
    lons, lats = lon_grid[0], lat_grid[:, 0]
    if not np.all(lon_grid == lons[None, :]) or not np.all(lat_grid == lats[:, None]):
        raise ValueError("Coordinates are not a rectilinear grid")
    if not ((np.diff(lons) > 0).all() and (np.diff(lats) < 0).all()):
        raise ValueError("Expected eastward columns and southward rows")
    if (np.abs(lons) > 180).any() or (np.abs(lats) > 85).any():
        raise ValueError("Coordinates exceed supported geographic bounds")
    # XML centres are rounded to four decimals, producing alternating spacings.
    if not np.allclose(np.diff(lons), np.diff(lons).mean(), atol=0.00011, rtol=0):
        raise ValueError("Irregular longitude spacing")
    if not np.allclose(np.diff(lats), np.diff(lats).mean(), atol=0.00011, rtol=0):
        raise ValueError("Irregular latitude spacing")
    for key, coordinate in (("lon_min", lons[0]), ("lon_max", lons[-1]),
                            ("lat_min", lats[-1]), ("lat_max", lats[0])):
        if abs(float(spec.attrib[key]) - coordinate) > 0.00011:
            raise ValueError(f"Grid specification mismatch: {key}")
    return ShakeGrid(lons.copy(), lats.copy(),
                     {name: values[:, :, index].copy() for index, name in enumerate(names)
                      if name not in ("LON", "LAT")},
                     {field.attrib["name"]: field.attrib.get("units", "") for field in ordered},
                     dict(root.attrib), dict(event.attrib))


def cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2 or not np.isfinite(centers).all():
        raise ValueError("At least two finite one-dimensional cell centres required")
    differences = np.diff(centers)
    if not ((differences > 0).all() or (differences < 0).all()):
        raise ValueError("Cell centres must be monotone")
    return np.r_[centers[0] - differences[0] / 2,
                 (centers[:-1] + centers[1:]) / 2,
                 centers[-1] + differences[-1] / 2]


def rectangle_area_km2(west, south, east, north):
    """Exact area of longitude/latitude rectangles on the WGS84 ellipsoid.

    Integrates a²(1-e²) cos(phi)/(1-e² sin²(phi))² over phi and lambda.
    Boundaries follow meridians/parallels, not geodesic corner-to-corner lines.
    Inputs may be numpy arrays for broadcasted cell areas.
    """
    a = 6378137.0
    flattening = 1 / 298.257223563
    eccentricity_squared = flattening * (2 - flattening)
    eccentricity = math.sqrt(eccentricity_squared)

    def primitive(latitude):
        sine = np.sin(np.deg2rad(latitude))
        return sine / (2 * (1 - eccentricity_squared * sine * sine)) + np.arctanh(eccentricity * sine) / (2 * eccentricity)

    if (np.any(np.asarray(east) < np.asarray(west)) or
            np.any(np.asarray(north) < np.asarray(south))):
        raise ValueError("Rectangle edges are reversed")
    return a * a * (1 - eccentricity_squared) * np.deg2rad(np.asarray(east) - np.asarray(west)) * (primitive(north) - primitive(south)) / 1_000_000


def clipped_cell_areas(grid: ShakeGrid, bbox=AOI) -> np.ndarray:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
        raise ValueError("Invalid domain bounding box")
    x, y = cell_edges(grid.lons), cell_edges(grid.lats)
    left, right = np.maximum(x[:-1], west), np.minimum(x[1:], east)
    bottom, top = np.maximum(y[1:], south), np.minimum(y[:-1], north)
    right, top = np.maximum(left, right), np.maximum(bottom, top)
    return rectangle_area_km2(left[None, :], bottom[:, None], right[None, :], top[:, None])


def analyze(grid: ShakeGrid, uncertainty: ShakeGrid, bbox=AOI) -> dict:
    """Area/threshold summaries; uncertainty shifts are sensitivity, not a CI."""
    if not (np.array_equal(grid.lons, uncertainty.lons) and np.array_equal(grid.lats, uncertainty.lats)):
        raise ValueError("Shaking and uncertainty grids are not aligned")
    if grid.event.get("event_id") != uncertainty.event.get("event_id"):
        raise ValueError("Shaking and uncertainty event IDs differ")
    for name, units in (("MMI", "intensity"), ("PGA", "%g")):
        if grid.units.get(name) != units:
            raise ValueError(f"Unexpected {name} units")
    if uncertainty.units.get("STDMMI") != "intensity":
        raise ValueError("Unexpected intensity uncertainty units")
    mmi, pga, sigma = grid.fields["MMI"], grid.fields["PGA"], uncertainty.fields["STDMMI"]
    if not ((mmi >= 1).all() and (mmi <= 10).all() and (pga >= 0).all() and (sigma >= 0).all()):
        raise ValueError("Invalid intensity, PGA or standard deviation range")
    areas = clipped_cell_areas(grid, bbox)
    inside = areas > 0
    domain_area = float(rectangle_area_km2(*bbox))
    modeled_area = float(areas.sum())
    if abs(modeled_area - domain_area) > 1e-7 * domain_area:
        raise ValueError("The source grid does not completely cover the pilot domain")

    def area(mask):
        return float(areas[mask].sum())

    classes = np.digitize(mmi, MMI_BINS, right=False)
    result = {
        "domain_bbox": list(bbox), "domain_area_km2": domain_area,
        "modeled_domain_area_km2": modeled_area, "modeled_fraction": min(1.0, modeled_area / domain_area),
        "native_cells_intersecting_domain": int(inside.sum()),
        "mmi_ge_5_area_km2": area(mmi >= 5), "mmi_ge_6_area_km2": area(mmi >= 6),
        "pga_ge_10_percent_g_area_km2": area(pga >= 10),
        "maximum_mmi_in_domain": float(mmi[inside].max()),
        "maximum_pga_percent_g_in_domain": float(pga[inside].max()),
        "mmi_stddev_area_weighted_mean": float((sigma * areas).sum() / modeled_area),
        "mmi_stddev_min_in_domain": float(sigma[inside].min()),
        "mmi_stddev_max_in_domain": float(sigma[inside].max()),
        "mmi_ge_6_minus_1sd_sensitivity_km2": area(mmi - sigma >= 6),
        "mmi_ge_6_plus_1sd_sensitivity_km2": area(mmi + sigma >= 6),
        "mmi_class_areas": [{"label": label, "area_km2": area(classes == i)}
                            for i, label in enumerate(MMI_LABELS)],
    }
    if not math.isclose(sum(row["area_km2"] for row in result["mmi_class_areas"]), modeled_area, rel_tol=1e-12):
        raise ValueError("Intensity classes do not close to total model domain area")
    return result


def mercator_sample(grid: ShakeGrid, field: str, bbox=AOI, width=900) -> np.ndarray:
    """Nearest native cell sampled onto pixel centres of a Web Mercator image.

    Display oversampling adds no information; numerical metrics use native cells.
    """
    west, south, east, north = bbox
    project = lambda lat: np.arcsinh(np.tan(np.deg2rad(lat)))
    y_south, y_north = project(south), project(north)
    height = round(width * (y_north - y_south) / np.deg2rad(east - west))
    x = west + (np.arange(width) + .5) / width * (east - west)
    y = y_north - (np.arange(height) + .5) / height * (y_north - y_south)
    latitude = np.rad2deg(np.arctan(np.sinh(y)))
    columns = np.searchsorted(cell_edges(grid.lons), x, side="right") - 1
    rows = np.searchsorted(-cell_edges(grid.lats), -latitude, side="right") - 1
    if ((columns < 0).any() or (columns >= grid.lons.size).any() or
            (rows < 0).any() or (rows >= grid.lats.size).any()):
        raise ValueError("Display domain exceeds the source grid")
    return grid.fields[field][np.ix_(rows, columns)]


def render_categories(grid, field, destination, bins, colors, bbox=AOI, width=900):
    samples = mercator_sample(grid, field, bbox, width)
    if len(colors) != len(bins) + 1:
        raise ValueError("Each categorical interval requires a colour")
    palette = np.array([[int(color[index:index + 2], 16) for index in (1, 3, 5)] + [220]
                        for color in colors], dtype=np.uint8)
    rgba = palette[np.digitize(samples, bins, right=False)]
    Image.fromarray(rgba).save(destination, optimize=True)
    return {"width": int(rgba.shape[1]), "height": int(rgba.shape[0]),
            "bounds": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            "crs": "EPSG:3857", "resampling": "nearest native cell"}
