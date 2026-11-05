"""Contracts and exact-grid primitives for the authenticated satellite pilots.

Separate from the immutable wildfire modules. No credentials or computed
satellite observations are stored in source files.
"""
from dataclasses import dataclass
import math

from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from .export import Grid


@dataclass
class SatelliteAnalysis:
    event: dict
    aoi: object
    bbox: tuple
    crs: str
    scale_m: int
    rasters: dict
    displays: list
    provenance: dict
    pre_valid_id: str
    post_valid_id: str
    shared_valid_id: str
    candidate_id: str | None
    audit: dict


def analysis_grid(bbox, crs, scale_m):
    """Bounded subwindow of [scale,0,0,0,-scale,10000000], including 30m phase."""
    if not 10 <= scale_m <= 60 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("Invalid analytical grid configuration")
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    origin_y = 10_000_000
    left = math.floor(west / scale_m) * scale_m
    right = math.ceil(east / scale_m) * scale_m
    top = origin_y + math.ceil((north - origin_y) / scale_m) * scale_m
    bottom = origin_y + math.floor((south - origin_y) / scale_m) * scale_m
    grid = Grid(crs, Affine(scale_m, 0, left, 0, -scale_m, top),
                int(round((right - left) / scale_m)), int(round((top - bottom) / scale_m)))
    grid.check_size(bands=1, bytes_per_sample=4)
    return grid


def grid_domain(ee, bbox, grid):
    """Binary centre support and a projected rectangle aligned to every full cell."""
    ll = ee.Image.pixelLonLat().reproject(crs=grid.crs, crsTransform=list(grid.transform)[:6])
    west, south, east, north = bbox
    domain = (ll.select("longitude").gte(west).And(ll.select("longitude").lt(east))
              .And(ll.select("latitude").gte(south)).And(ll.select("latitude").lt(north))
              .rename("domain").toByte())
    left, top = grid.transform.c, grid.transform.f
    region = ee.Geometry.Rectangle([left, top-grid.height*grid.transform.a,
                                    left+grid.width*grid.transform.a, top], proj=grid.crs, geodesic=False)
    return domain, region
