"""Caller-owned Colab backend and widget factory for a future course host."""
from dataclasses import dataclass
from datetime import date
import math


@dataclass(frozen=True)
class DetectionRequest:
    bounds: tuple
    start: str
    end: str
    confidence: float = 0.55
    scene_cloud_max: float = 30

    def validate(self):
        from ship_map import validate_bounds
        validate_bounds(self.bounds)
        if date.fromisoformat(self.start) >= date.fromisoformat(self.end):
            raise ValueError('End date must follow start date; end is exclusive.')
        if not math.isfinite(self.confidence) or not 0.05 <= self.confidence <= 0.95:
            raise ValueError('Confidence must be between 0.05 and 0.95.')
        if not math.isfinite(self.scene_cloud_max) or not 0 <= self.scene_cloud_max <= 100:
            raise ValueError('Scene cloud maximum must be between 0 and 100.')
        return self


def run_detection(project, model, device, request):
    """Same Python call for standalone widgets and a host; no HTTP server/tunnel."""
    from pathlib import Path
    from datetime import datetime, timezone
    import time
    from ship_backend import checksum
    from ship_map import fetch_observation, detect_raster, geographic_results, save_detection
    request.validate()
    if model is None:
        raise ValueError('Restore the preserved checkpoint before running detection.')
    rgb, valid, transform, crs, metadata = fetch_observation(
        request.bounds, request.start, request.end, request.scene_cloud_max)
    started = time.perf_counter()
    boxes, scores = detect_raster(model, device, rgb, valid, request.confidence)
    metadata.update(threshold=request.confidence, detection_count=len(boxes),
                    scene_cloud_max=request.scene_cloud_max, inference_seconds=time.perf_counter()-started,
                    model_checkpoint_sha256=checksum(Path(project)/'results/models/vessel_detector.pt','sha256'),
                    run_utc=datetime.now(timezone.utc).isoformat(), canadian_accuracy_measured=False,
                    validation_status='experimental predictions; not validated ground truth')
    geojson = geographic_results(boxes, scores, transform, crs, metadata)
    stem = save_detection(project, rgb, boxes, scores, geojson, metadata)
    return {'geojson':geojson, 'metadata':metadata, 'output_stem':stem}


def create_project_view(project, model=None, device=None, metrics=None):
    """A course project button should call this factory to reload the entire map."""
    from ship_map import build_detection_app
    return build_detection_app(project, model, device, metrics or {})
