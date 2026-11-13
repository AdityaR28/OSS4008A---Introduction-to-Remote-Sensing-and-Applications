"""
Project 11 (OSS4008A): Water bodies in Canada.
Reusable layer functions and controls for the dashboard, built so the project
can plug into a combined UI.

- Layer functions return (image, display settings, layer name), so any map can
  show them with map.addLayer(*result).
- show_water() and build_water_controls() let a shared UI load this project and
  its filter buttons onto a map the shared UI owns.
"""
import ee
import ipywidgets as widgets

# Canada's boundary (LSIB global borders, filtered to Canada).
# union() merges its pieces so internal seams aren't drawn.
canada = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
           .filter(ee.Filter.eq('country_na', 'Canada')) \
           .union(maxError=100)

# JRC Global Surface Water: 1984-present water history at 30 m
_jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')


# ---------------------------------------------------------------------------
# Layer functions: each returns (image, display settings, layer name)
# ---------------------------------------------------------------------------

def get_overview_layer():
    """All water: pixels that were water more than 50% of the time since 1984."""
    img = (_jrc.select('occurrence')     # % of time each pixel was water
           .gt(50)                       # keep mostly-permanent water
           .selfMask()                   # hide non-water pixels
           .clipToCollection(canada))    # cut to Canada
    return img, {'palette': ['0066ff']}, 'Water (Canada)'


def get_permanent_layer():
    """Water present all 12 months of the year (JRC seasonality)."""
    img = _jrc.select('seasonality').eq(12).selfMask().clipToCollection(canada)
    return img, {'palette': ['0033cc']}, 'Permanent water'


def get_seasonal_layer():
    """Water present only part of the year (1-11 months): often ponds and wetlands."""
    s = _jrc.select('seasonality')
    img = s.gt(0).And(s.lt(12)).selfMask().clipToCollection(canada)
    return img, {'palette': ['33ff99']}, 'Seasonal water'


def get_rivers_layer():
    """Reference river network (HydroSHEDS), larger rivers only.
    Reference data, not this project's detection."""
    rivers = (ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
              .filterBounds(canada.geometry())
              .filter(ee.Filter.lte('RIV_ORD', 6)))   # 1 = largest rivers
    return rivers.style(color='00e5ff', width=1), {}, 'Rivers (reference)'


def detect_in_region(region, start='2024-06-01', end='2024-09-30'):
    """Water in a selected region.
    Placeholder: Dynamic World summer water class; to be replaced by the
    trained U-Net at Progress 3."""
    img = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
           .filterBounds(region)
           .filterDate(start, end)       # summer only, avoids ice
           .select('label')
           .mode()                       # most common class per pixel
           .eq(0)                        # class 0 = water
           .selfMask()
           .clip(region))
    return img, {'palette': ['00ffff']}, 'Water (region)'


# ---------------------------------------------------------------------------
# Integration with a combined UI
# ---------------------------------------------------------------------------

# Names of every layer this project adds, so filters only swap this project's layers
WATER_LAYER_NAMES = {'Water (Canada)', 'Permanent water', 'Seasonal water',
                     'Rivers (reference)', 'Water (region)'}

# Filter buttons: (button label, layer function)
FILTERS = [
    ('All water',       get_overview_layer),
    ('Permanent water', get_permanent_layer),
    ('Seasonal water',  get_seasonal_layer),
    ('Rivers',          get_rivers_layer),
]


def show_water(m):
    """Entry point for a shared UI: load this project's default view onto map m."""
    m.addLayer(*get_overview_layer())


def build_water_controls(get_map, log=print):
    """Filter buttons for the water project.
    get_map: function returning the map currently shown (owned by the shared UI)
    log:     function that receives status messages (e.g. the shared chat panel)
    Returns a row of buttons; each swaps this project's layer on the shared map."""
    def make_handler(label, fn):
        def handler(_):
            m = get_map()
            for layer in list(m.layers):                 # remove this project's previous layer
                if layer.name in WATER_LAYER_NAMES:
                    m.remove_layer(layer)
            log(f'Loading {label.lower()}...')
            try:
                m.addLayer(*fn())
                log(f'{label} has loaded.')
            except Exception as e:
                log(f'Could not load {label.lower()}: {e}')
        return handler

    buttons = []
    for label, fn in FILTERS:
        b = widgets.Button(description=label)
        b.on_click(make_handler(label, fn))
        buttons.append(b)
    return widgets.HBox(buttons)


# Project description for a shared dashboard
PROJECT_INFO = {
    'id': 'water',
    'title': 'Water bodies (lakes, rivers, ponds)',
    'student': 'Devon Clermont-Gauer',
    'entry': show_water,               # loads the default view
    'controls': build_water_controls,  # builds the filter buttons
    'layers': FILTERS,                 # (label, function) pairs
    'region_tool': detect_in_region,   # for "draw a box" analysis
}
