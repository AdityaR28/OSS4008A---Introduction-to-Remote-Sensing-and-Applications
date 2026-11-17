"""
Project 11 (OSS4008A): Water bodies in Canada.
Reference map layers and controls for the dashboard, built so the project
can plug into a combined UI.

- Reference layers return (image, display settings, layer name), so any map can
  show them with map.addLayer(*result).
- show_water() and build_water_controls() let a shared UI load this project and
  its filter buttons onto a map the shared UI owns.
- Water detection is done by the AI (U-Net) in water_ai.py; the layers here are
  reference data for country-scale views, where the AI is too slow to run.
"""
import ee
import ipywidgets as widgets

# Canada's boundary (LSIB global borders, filtered to Canada).
# union() merges its pieces so internal seams aren't drawn.
canada = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
           .filter(ee.Filter.eq('country_na', 'Canada')) \
           .union(maxError=100)


# ---------------------------------------------------------------------------
# Reference layers: each returns (image, display settings, layer name)
# ---------------------------------------------------------------------------

def get_overview_layer():
    """All water (reference): pixels that were water more than 50% of the time since 1984
    (JRC Global Surface Water). Not AI output."""
    img = (ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
           .select('occurrence')         # % of time each pixel was water
           .gt(50)                       # keep mostly-permanent water
           .selfMask()                   # hide non-water pixels
           .clipToCollection(canada))    # cut to Canada
    return img, {'palette': ['0066ff']}, 'All water (reference)'


def get_rivers_layer():
    """Rivers (reference): HydroSHEDS river network, larger rivers only.
    Derived from ~500 m elevation data, so lines are approximate at close zoom. Not AI output."""
    rivers = (ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
              .filterBounds(canada.geometry())
              .filter(ee.Filter.lte('RIV_ORD', 6)))   # 1 = largest rivers
    return rivers.style(color='00e5ff', width=1), {}, 'Rivers (reference)'


def detect_in_region(region, start='2024-06-01', end='2024-09-30'):
    """Dynamic World summer water class for a region.
    Used to build training labels; NOT used for detection in the tool
    (detection is done by the U-Net in water_ai.py)."""
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
WATER_LAYER_NAMES = {'All water (reference)', 'Rivers (reference)', 'Water (AI)'}

# Filter buttons: (button label, layer function)
FILTERS = [
    ('All water (reference)', get_overview_layer),
    ('Rivers (reference)',    get_rivers_layer),
]


def show_water(m):
    """Entry point for a shared UI: load this project's default reference view onto map m."""
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
    'entry': show_water,                          # loads the default reference view
    'controls': build_water_controls,             # builds the reference filter buttons
    'layers': FILTERS,                            # (label, function) pairs
    'region_tool': 'water_ai.detect_water_ai',    # AI detection lives in water_ai.py
}
