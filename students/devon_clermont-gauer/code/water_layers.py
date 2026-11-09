"""
Project 11 (OSS4008A): Water bodies in Canada.
Reusable layer functions for the dashboard. Each layer function returns
(image, display settings, layer name) so any UI can add it to a map.
"""
import ee

# Canada's boundary (LSIB global borders, filtered to Canada).
# union() merges its pieces so internal seams aren't drawn.
canada = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
           .filter(ee.Filter.eq('country_na', 'Canada')) \
           .union(maxError=100)

# JRC Global Surface Water: 1984-present water history at 30 m
_jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')


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
    """Water in a selected region (not used in the UI yet).
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
