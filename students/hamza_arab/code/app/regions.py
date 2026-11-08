"""Reference locations for the OSS4008A demonstration, not hospital ground truth."""

REGIONS = {
    "halifax": {"center": (44.6488, -63.5752), "zoom": 11},
    "ottawa": {"center": (45.4215, -75.6972), "zoom": 11},
}


def get_region(name: str) -> dict:
    """Return a copy of a known map region or raise a helpful error."""
    key = name.strip().lower()
    if key not in REGIONS:
        raise ValueError(f"Unknown region: {name!r}. Choose from {', '.join(REGIONS)}")
    return REGIONS[key].copy()
