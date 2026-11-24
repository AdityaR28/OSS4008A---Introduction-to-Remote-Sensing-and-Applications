"""
Project 11 (OSS4008A): water body classification.

Takes the AI's water mask (True = water) and labels every separate water body as
a lake, river or pond, using its size and shape:

  - River: long and narrow. Measured by "elongation" = (centre-line length)^2 / area.
           A round lake scores low (~1-8); a river scores high (>= 10).
  - Pond:  compact (not a river) and smaller than 2 ha (Biggs et al., 2005).
  - Lake:  compact and 2 ha or larger.

Classification only runs at 10-20 m resolution. At coarser resolution, narrow rivers
break into fragments that would be mislabelled as ponds, so the caller should show
"all water" instead.

Pure numpy + scikit-image: works the same in the Colab dashboard and the web app.
"""
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize

# Class codes used in the class map (0 = not water / too small to classify)
LAKE, RIVER, POND = 1, 2, 3
NAMES = {LAKE: 'lakes', RIVER: 'rivers', POND: 'ponds'}
COLORS = {LAKE: (30, 110, 255),      # blue
          RIVER: (0, 229, 255),      # cyan
          POND: (110, 255, 140)}     # green

# Rules (tune in 04_water_classification.ipynb)
POND_MAX_M2 = 20_000       # ponds are under 2 ha (20,000 m²)
RIVER_ELONGATION = 10      # elongation at or above this = river
MAX_RES_M = 20             # only classify at 10-20 m resolution
MIN_PIXELS = 3             # ignore specks smaller than this (likely noise)


def classify_water(water, res):
    """Label each water body in a water mask as lake, river or pond.

    water: 2-D True/False array from the AI (True = water)
    res:   pixel size in metres (10 m = full detail)

    Returns (class_map, summary), or None if res is too coarse to classify.
      class_map: same shape as water; 0 = none, 1 = lake, 2 = river, 3 = pond
      summary:   {'lakes': {'count', 'km2'}, 'rivers': ..., 'ponds': ..., 'partial': n}
                 'partial' = bodies touching the edge of the scanned area (only partly seen)
    """
    if res > MAX_RES_M:
        return None

    # 1. Split the mask into separate water bodies (touching pixels, incl. diagonals, = one body)
    bodies = label(water, connectivity=2)
    n = int(bodies.max())
    empty = {name: {'count': 0, 'km2': 0.0} for name in NAMES.values()}
    if n == 0:
        return np.zeros(water.shape, np.uint8), {**empty, 'partial': 0}

    # 2. Measure every body at once (bincount counts pixels per body number)
    area_px = np.bincount(bodies.ravel(), minlength=n + 1)          # size of each body in pixels
    centre_line = skeletonize(water)                                 # 1-pixel-wide centre lines
    length_px = np.bincount(bodies[centre_line], minlength=n + 1)    # centre-line length per body
    area_m2 = area_px * res * res
    elongation = length_px.astype(float) ** 2 / np.maximum(area_px, 1)

    # Bodies cut off by the edge of the scanned area (only partly visible)
    on_edge = np.zeros(n + 1, bool)
    for side in (bodies[0, :], bodies[-1, :], bodies[:, 0], bodies[:, -1]):
        on_edge[np.unique(side)] = True
    on_edge[0] = False

    # 3. Apply the rules to every body
    is_body = (np.arange(n + 1) > 0) & (area_px >= MIN_PIXELS)
    river = is_body & (elongation >= RIVER_ELONGATION)
    pond = is_body & ~river & (area_m2 < POND_MAX_M2)
    lake = is_body & ~river & ~pond

    cls = np.zeros(n + 1, np.uint8)
    cls[lake], cls[river], cls[pond] = LAKE, RIVER, POND
    class_map = cls[bodies]                                          # paint each body with its class

    # 4. Summary for the chat
    summary = {name: {'count': int((cls == k).sum()), 'km2': float(area_m2[cls == k].sum() / 1e6)}
               for k, name in NAMES.items()}
    summary['partial'] = int((on_edge & (cls > 0)).sum())
    return class_map, summary


def class_rgba(class_map, show=(LAKE, RIVER, POND), alpha=200):
    """Colour picture of the classes in `show` (others transparent), as an RGBA array."""
    rgba = np.zeros((*class_map.shape, 4), np.uint8)
    for k in show:
        rgba[class_map == k] = (*COLORS[k], alpha)
    return rgba


def summary_text(summary):
    """One-line description of a summary, e.g. '12 lakes (40.1 km²), 3 rivers (5.2 km²), 40 ponds (0.3 km²)'."""
    parts = [f"{s['count']} {name} ({s['km2']:.1f} km²)"
             for name, s in summary.items() if name != 'partial']
    text = ', '.join(parts)
    if summary['partial']:
        text += f"; {summary['partial']} extend beyond the scanned area, so their type may be less reliable"
    return text
