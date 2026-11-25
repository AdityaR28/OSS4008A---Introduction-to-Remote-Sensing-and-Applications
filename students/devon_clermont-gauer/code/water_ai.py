"""
Project 11 (OSS4008A): AI water detection.
Runs the trained U-Net on Sentinel-2 imagery for a user-selected area and
returns a water mask, its map bounds, the water area and the resolution used.
Small boxes run at full 10 m detail (as trained); larger boxes run at a coarser
resolution so downloads stay a similar size (rougher: small water is missed).
Image chunks are downloaded in parallel. The model predicts overlapping tiles that are
blended together (no seams at tile edges), and holes inside water that are spectrally
water (NDWI > 0) are filled, so the middle of large lakes isn't left with gaps.
"""
import base64, io, json, math, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ee
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from scipy.ndimage import binary_fill_holes
from ipyleaflet import ImageOverlay

BANDS = ['B2', 'B3', 'B4', 'B8', 'B11']        # same bands as training
START, END = '2024-06-01', '2024-09-30'        # same summer window as training
R = 6378137.0                                  # Earth radius used by web maps (Web Mercator)
TILE = 256                                     # model input size, same as training
STRIDE = 128                                   # tiles overlap by half, so every pixel is seen with surrounding context
CHUNK = 512                                    # pixels per download request side
WORKERS = 8                                    # download requests at once (tested: 10-12 gave no gain;
                                               # 12 exceeded the Earth Engine client's 10-connection pool)

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
    """Cloud-filtered summer Sentinel-2 median composite, built exactly as for training.
    (A lighter composite of the 10 clearest images was tested: similar results, ~2x slower.)"""
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region).filterDate(START, END)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(BANDS).median().toInt16().unmask(0))


def _fetch(image, x0, y_top, step, w, h):
    """Download image pixels for the grid as a (5, h, w) array.
    Chunks are requested in parallel; Earth Engine averages pixels on its side
    when the grid is coarser than 10 m."""
    out = np.zeros((5, h, w), np.float32)
    jobs = [(r, c) for r in range(0, h, CHUNK) for c in range(0, w, CHUNK)]

    def get(job):
        r, c = job
        cw, ch = min(CHUNK, w - c), min(CHUNK, h - r)
        request = {'expression': image, 'fileFormat': 'NUMPY_NDARRAY',
                   'grid': {'dimensions': {'width': cw, 'height': ch},
                            'affineTransform': {'scaleX': step, 'shearX': 0, 'translateX': x0 + c * step,
                                                'shearY': 0, 'scaleY': -step, 'translateY': y_top - r * step},
                            'crsCode': 'EPSG:3857'}}
        for attempt in range(4):                 # retry if Earth Engine is busy
            try:
                arr = ee.data.computePixels(request)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        out[:, r:r + ch, c:c + cw] = np.stack([arr[b] for b in BANDS]).astype(np.float32)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(get, jobs))
    return out


def _window(size=TILE):
    """Weights for blending overlapping tiles: high in the tile centre, low at its edges
    (the model is least reliable at tile edges, where it has no surrounding context)."""
    ramp = np.minimum(np.arange(size) + 1, size - np.arange(size)).astype(np.float32)
    ramp = np.minimum(ramp / (size / 4), 1.0) * 0.9 + 0.1
    return np.outer(ramp, ramp)


def _tile_probs(x, predict_batch, batch=16):
    """Water probability for a (5, H, W) array using overlapping tiles, blended with _window.
    predict_batch: function taking (n, 5, TILE, TILE) and returning (n, TILE, TILE) probabilities."""
    _, H, W = x.shape
    weight = _window()
    prob = np.zeros((H, W), np.float32)
    total = np.zeros((H, W), np.float32)
    spots = [(r, c) for r in range(0, H - TILE + 1, STRIDE) for c in range(0, W - TILE + 1, STRIDE)]
    for i in range(0, len(spots), batch):
        group = spots[i:i + batch]
        p = predict_batch(np.stack([x[:, r:r + TILE, c:c + TILE] for r, c in group]))
        for (r, c), tile in zip(group, p):
            prob[r:r + TILE, c:c + TILE] += tile * weight
            total[r:r + TILE, c:c + TILE] += weight
    return prob / np.maximum(total, 1e-6)


def _predict(img):
    """Run the U-Net over the image with overlapping tiles; returns a True/False water mask.
    Afterwards, holes inside water that are spectrally water (NDWI > 0) are filled in;
    real islands (NDWI < 0) stay land."""
    _, h, w = img.shape
    # Pad so overlapping tiles cover the whole image, with a margin of half a tile on every side
    pad = STRIDE
    H = math.ceil((h + 2 * pad - TILE) / STRIDE) * STRIDE + TILE
    W = math.ceil((w + 2 * pad - TILE) / STRIDE) * STRIDE + TILE
    x = np.pad((img - _mean) / _std, ((0, 0), (pad, H - h - pad), (pad, W - w - pad)), mode='reflect')

    def predict_batch(tiles):
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(tiles)).to(_device)
            return torch.sigmoid(_model(t))[:, 0].cpu().numpy()

    prob = _tile_probs(x, predict_batch)[pad:pad + h, pad:pad + w]
    water = prob > 0.5

    # Fill holes inside water that look like water in the bands (NDWI = (green - NIR) / (green + NIR))
    ndwi = (img[1] - img[3]) / (img[1] + img[3] + 1e-6)
    holes = binary_fill_holes(water) & ~water
    water |= holes & (ndwi > 0)

    water[(img == 0).all(axis=0)] = False          # no imagery -> not water
    return water


def detect_water_ai(region, max_km2=20000, detail_km2=400, max_res=60):
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


def get_input_layer(region):
    """The Sentinel-2 image the AI analysed for this region, as a map layer.
    Returns (image, display settings, layer name) like the reference layers."""
    img = _s2_image(region).clip(region)
    return img, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, 'AI input (Sentinel-2)'
