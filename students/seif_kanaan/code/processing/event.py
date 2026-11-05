"""Reproducible pilot configuration; no imagery or statistics are fabricated here."""

from dataclasses import asdict, dataclass
from datetime import date
import math

S2_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_DATASET = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
ANALYSIS_CRS = "EPSG:32611"
ANALYSIS_TRANSFORM = [20, 0, 0, 0, -20, 10000000]


@dataclass(frozen=True)
class Window:
    start: str
    end_exclusive: str

    def __post_init__(self):
        start, end = date.fromisoformat(self.start), date.fromisoformat(self.end_exclusive)
        if not start < end or (end - start).days > 92:
            raise ValueError("Each acquisition window must be positive and no more than 92 days.")


@dataclass(frozen=True)
class Config:
    pre: Window = Window("2023-07-15", "2023-08-15")
    post: Window = Window("2023-09-15", "2023-10-16")
    aoi_bbox: tuple[float, float, float, float] = (-119.85, 49.78, -119.43, 50.12)
    threshold: float = 0.27
    cloud_score_threshold: float = 0.60
    scene_cloud_limit: float = 80
    min_valid_fraction: float = 0.40
    display_pixel_size_m: int = 60
    max_scenes_per_window: int = 256

    def __post_init__(self):
        west, south, east, north = self.aoi_bbox
        if not all(math.isfinite(v) for v in self.aoi_bbox):
            raise ValueError("AOI coordinates must be finite.")
        if not (-120 <= west < east <= -114 and 45 <= south < north <= 60):
            raise ValueError("This pilot is bounded to the Canadian part of UTM zone 11N.")
        if east - west > 0.7 or north - south > 0.7:
            raise ValueError("AOI too large for bounded synchronous exports; use a tiled batch workflow.")
        if self.pre.end_exclusive > self.post.start:
            raise ValueError("Pre/post acquisition windows must not overlap.")
        if not 0 < self.threshold <= 2:
            raise ValueError("Candidate dNBR threshold must be in (0, 2].")
        if not 0 <= self.cloud_score_threshold <= 1:
            raise ValueError("Cloud score threshold must be in [0, 1].")
        if not 0 < self.min_valid_fraction <= 1 or not 0 <= self.scene_cloud_limit <= 100:
            raise ValueError("Invalid coverage or scene-cloud threshold.")
        if self.display_pixel_size_m < 40 or self.max_scenes_per_window > 256:
            raise ValueError("Export pixels must be >=40 map metres and scene limit <=256.")
        if self.max_scenes_per_window < 1:
            raise ValueError("Scene limit must be positive.")

    def to_dict(self):
        return asdict(self)


EVENT = {
    "id": "mcdougall-creek-2023", "name": "McDougall Creek", "year": 2023,
    "location": "West Kelowna, British Columbia", "ignition_date": "2023-08-15",
    "center": [49.92, -119.65],
}
