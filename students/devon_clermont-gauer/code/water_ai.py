"""
Project 11 (OSS4008A): AI water detection.
Runs the trained U-Net on Sentinel-2 imagery for a user-selected area and
returns a water mask, its map bounds, the water area and the resolution used.
Small boxes run at full 10 m detail (as trained); larger boxes run at a coarser
resolution so downloads stay a similar size (rougher: small water is missed).
"""
import base64, io, json, math
import numpy as np
import ee
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from ipyleaflet import ImageOverlay

BANDS = ['B2', 'B3', 'B4', 'B8', 'B11']        # same bands as training
START, END = '2024-06-01', '2024-09-30'        # same summer window as training
R = 6378137.0                                  # Earth radius used by web maps (Web Mercator)
TILE = 256                                     # model input size, same as training
CHUNK = 1024                                   # pixels per download request (keeps requests small)

_model, _mean, _std = None, None, None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path, stats_path):
    """Load the trained U-Net and the band statistics used to rescale inputs."""
    global _model, _mean, _std
    stats = json.load(open(stats_path))
    _mean = np.array(stats['mean'], np.float32)[:, None, None]
    _std = np.array(stats['std'], np.float32)[:, None, None]
    m = smp.Unet('resnet34', encoder_weights=None, in_channels=5, classes=1)
    m.load_state_dict(torch.load(model_path, map_location=_device))
    _model = m.to(_device).eval()


def _to_merc(lon, lat):
    """Longitude/latitude -> Web Mercator metres (the projection web maps use)."""
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _from_merc(x, y):
    """Web Mercator metres -> longitude/latitude."""
    return math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)


def _s2_image(region):
    """Cloud-filtered summer Sentinel-2 median composite, built exactly as for training."""
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region).filterDate(START, END)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(BANDS).median().toInt16().unmask(0))


def _fetch(image, x0, y_top, step, w, h):
    """Download image pixels for the grid, in chunks, as a (5, h, w) array.
    Earth Engine averages pixels on its side when the grid is coarser than 10 m."""
    out = np.zeros((5, h, w), np.float32)
    for r in range(0, h, CHUNK):
        for c in range(0, w, CHUNK):
            cw, ch = min(CHUNK, w - c), min(CHUNK, h - r)
            arr = ee.data.computePixels({
                'expression': image,
                'fileFormat': 'NUMPY_NDARRAY',
                'grid': {'dimensions': {'width': cw, 'height': ch},
                         'affineTransform': {'scaleX': step, 'shearX': 0, 'translateX': x0 + c * step,
                                             'shearY': 0, 'scaleY': -step, 'translateY': y_top - r * step},
                         'crsCode': 'EPSG:3857'}})
            out[:, r:r + ch, c:c + cw] = np.stack([arr[b] for b in BANDS]).astype(np.float32)
    return out


def _predict(img):
    """Run the U-Net over the image in 256x256 tiles; returns a True/False water mask."""
    _, h, w = img.shape
    H, W = math.ceil(h / TILE) * TILE, math.ceil(w / TILE) * TILE
    x = np.pad((img - _mean) / _std, ((0, 0), (0, H - h), (0, W - w)), mode='edge')
    prob = np.zeros((H, W), np.float32)
    with torch.no_grad():
        for r in range(0, H, TILE):
            for c in range(0, W, TILE):
                t = torch.from_numpy(x[None, :, r:r + TILE, c:c + TILE].copy()).to(_device)
                prob[r:r + TILE, c:c + TILE] = torch.sigmoid(_model(t))[0, 0].cpu().numpy()
    water = prob[:h, :w] > 0.5
    water[(img == 0).all(axis=0)] = False          # no imagery -> not water
    return water


def detect_water_ai(region, max_km2=5000, detail_km2=200, max_res=50):
    """AI water detection for a drawn area.
    Boxes up to detail_km2 run at full 10 m detail; larger boxes run coarser, up to max_res.
    Returns (water mask, map bounds ((south, west), (north, east)), water area in km², resolution in m)."""
    if _model is None:
        raise RuntimeError('Model not loaded: call load_model() first.')

    ring = region.bounds().coordinates().getInfo()[0]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    west, east, south, north = min(lons), max(lons), min(lats), max(lats)

    x0, y0 = _to_merc(west, south)
    x1, y1 = _to_merc(east, north)
    stretch = 1 / math.cos(math.radians((south + north) / 2))   # Web Mercator stretch at this latitude

    # Box size on the ground
    box_km2 = ((x1 - x0) / stretch) * ((y1 - y0) / stretch) / 1e6
    if box_km2 > max_km2:
        raise ValueError(f'Your box is {box_km2:.0f} km²; please draw one under {max_km2} km².')

    # Resolution: 10 m up to detail_km2, then coarser so the pixel count stays about the same
    res = min(max_res, max(10, 10 * math.sqrt(box_km2 / detail_km2)))
    step = res * stretch                                         # grid step in Web Mercator units
    w, h = int((x1 - x0) / step), int((y1 - y0) / step)

    img = _fetch(_s2_image(region), x0, y1, step, w, h)
    water = _predict(img)

    lon_w, lat_s = _from_merc(x0, y1 - h * step)
    lon_e, lat_n = _from_merc(x0 + w * step, y1)
    water_km2 = water.sum() * res * res / 1e6                    # each pixel is res x res metres
    return water, ((lat_s, lon_w), (lat_n, lon_e)), water_km2, round(res)


def water_overlay(water, bounds, color=(0, 255, 255), name='Water (AI)'):
    """Turn a water mask into a semi-transparent map layer."""
    rgba = np.zeros((*water.shape, 4), np.uint8)
    rgba[water] = (*color, 200)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    return ImageOverlay(url=url, bounds=bounds, name=name)
