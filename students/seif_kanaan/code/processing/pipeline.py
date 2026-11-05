"""Authenticated computation followed by atomic validated publication of assets."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile

from . import __version__
from .auth import initialize
from .event import ANALYSIS_CRS, ANALYSIS_TRANSFORM, CLOUD_DATASET, EVENT, S2_DATASET, Config
from .export import EXPORT_TILE_SIZE, export_analytical, export_layers, sanity_check_area, write_manifest_atomic
from .metrics import MAX_REDUCTION_PIXELS, compute_metrics
from .optical import build_analysis
from .reference import compare_reference

LOGGER = logging.getLogger(__name__)


def build_manifest(config, analysis, metrics, layers, analytical_assets, sanity, reference):
    return {
        "schema_version": "1.0.0", "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "processing_version": __version__,
        "event": {**EVENT, "aoi_bbox": list(config.aoi_bbox)},
        "analysis": {
            "sensor": "Sentinel-2 MSI", "dataset": S2_DATASET, "cloud_dataset": CLOUD_DATASET,
            "crs": ANALYSIS_CRS, "crs_transform": ANALYSIS_TRANSFORM, "scale_m": 20,
            "threshold": config.threshold, "cloud_score_threshold": config.cloud_score_threshold,
            "cloud_score_band": "cs_cdf", "scene_cloud_limit_percent": config.scene_cloud_limit,
            "scl_kept_classes": [4, 5], "min_valid_fraction": config.min_valid_fraction,
            "composite": "Per-band median surface reflectance; NBR computed after compositing",
            "spatial_support": "10 m B8/RGB aggregated by area-weighted mean to native B12 20 m grid; all 10 m child pixels must pass clear/input-mask checks; common 20 m EPSG:32611 grid",
            "area_reducer": "ee.Image.pixelArea sum, 20 m EPSG:32611 fixed transform; no bestEffort",
            "area_reducer_max_pixels": MAX_REDUCTION_PIXELS,
            "raster_export": {"method": "ee.data.computePixels", "tile_size_pixels": EXPORT_TILE_SIZE,
                              "grid_assembly": "Exact raster windows, no local resampling", "workers": 1},
            "valid_fraction_denominator": "Full AOI including excluded water and uncertain land pixels",
            "display_crs": "EPSG:3857", "display_pixel_size_map_m": config.display_pixel_size_m,
            "display_note": "Nearest-neighbour display sampling only. Web Mercator map metres differ from ground metres; all statistics use the 20 m analysis grid.",
            **analysis.provenance,
        },
        "metrics": metrics, "layers": layers, "analytical_assets": analytical_assets,
        "reference": reference,
        "validation": {
            "status": "exploratory_reference_compared" if reference["status"] == "loaded" else "exploratory_unvalidated",
            "export_consistency": sanity,
            "notes": "Genuine Earth Engine data exported with numerical and coverage checks. No field validation or calibrated severity accuracy assessment. Inspect all layers before public release.",
        },
        "sources": [
            {"label": "Sentinel-2 harmonized surface reflectance · Copernicus/ESA", "url": "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED"},
            {"label": "Google Cloud Score+ · CC BY 4.0", "url": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED"},
            {"label": "BC Wildfire Service historical perimeters · Open Government Licence BC", "url": "https://catalogue.data.gov.bc.ca/dataset/bc-wildfire-fire-perimeters-historical"},
        ],
        "limitations": [
            "Analytical portfolio prototype; not an emergency warning system or evidence of property damage.",
            f"dNBR ≥ {config.threshold:g} flags candidate spectral change; the threshold is exploratory and not locally calibrated severity.",
            "Spectral change can reflect fire, phenology, moisture, harvest, smoke, shadows or other disturbance; composites span multiple dates.",
            "SCL vegetation/bare-land mask excludes water, dark/uncertain pixels, cloud, shadow and snow; no-data is not unburned.",
            "RGB and NBR before/after have individual coverage; dNBR and area use their valid intersection. Some haze or cloud may remain.",
            "Historical perimeter is a contextual hand-sketched source with unverified boundary precision; comparison is not pixel accuracy.",
            "The candidate estimate covers the analysis AOI and selected windows, not an official final fire size. SAR, WUI exposure and thermal layers are not implemented in this MVP.",
        ],
    }


def run_pipeline(project, output, *, config=None, authenticate=False, reference_path=None):
    config = config or Config()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory to preserve prior exports.")
    output.parent.mkdir(parents=True, exist_ok=True)
    ee = initialize(project, authenticate=authenticate)
    LOGGER.info("Discovering actual scenes and constructing composites")
    analysis = build_analysis(ee, config)
    return export_analysis(ee, output, analysis, config=config, reference_path=reference_path)


def export_analysis(ee, output, analysis, *, config=None, reference_path=None,
                    acquisition_selection=None, selection_report_bytes=None, progress_callback=None):
    """Export an existing graph once, after any independent coverage selection.

    Recovery callers provide the immutable selection report that determined the
    window. Its hash is checked before expensive export work, and the original
    pipeline's numerical checks and atomic publication remain unchanged.
    """
    config = config or Config()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory to preserve prior exports.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (acquisition_selection is None) != (selection_report_bytes is None):
        raise ValueError("Selection metadata and its report must be supplied together.")
    if acquisition_selection is not None:
        if not isinstance(selection_report_bytes, bytes):
            raise TypeError("The acquisition selection report must contain bytes.")
        if acquisition_selection.get("report_url") != "acquisition_selection_report.json":
            raise ValueError("Unexpected acquisition selection report path.")
        if hashlib.sha256(selection_report_bytes).hexdigest() != acquisition_selection.get("report_sha256"):
            raise ValueError("Acquisition selection report digest mismatch.")
    def progress(phase, **details):
        if progress_callback is not None:
            progress_callback(phase, details)

    progress("area_metrics")
    LOGGER.info("Computing real candidate areas and coverage at 20 m")
    metrics = compute_metrics(ee, analysis, config)
    LOGGER.info("Computed area metrics: %s", json.dumps(metrics, sort_keys=True))
    progress("area_metrics_computed", metrics=metrics)
    if acquisition_selection is not None:
        # Compare the production dNBR mask with the frozen coverage-only
        # decision before downloading rasters. Never silently export a graph
        # whose measured support differs from the selected observation support.
        rows = acquisition_selection.get("evaluated_windows", [])
        selected = acquisition_selection.get("selected_profile", {}).get("id")
        row = next((item for item in rows if item.get("id") == selected), {})
        for decision_key, metric_key in (
            ("shared_aoi_area_ha", "valid_area_ha"),
            ("pre_valid_area_ha", "pre_valid_area_ha"),
            ("post_valid_area_ha", "post_valid_area_ha"),
        ):
            expected, actual = row.get(decision_key), metrics.get(metric_key)
            if (isinstance(expected, bool) or not isinstance(expected, (int, float))
                    or not math.isfinite(expected) or expected < 0
                    or not math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-6)):
                raise ValueError(f"Selection/export coverage mismatch for {metric_key}: selected={expected!r}, export={actual!r} ha.")
    stage = Path(tempfile.mkdtemp(prefix=".wildfire-export-", dir=output.parent))
    try:
        progress("numerical_rasters", metrics=metrics)
        LOGGER.info("Exporting numerical 20 m rasters")
        assets = export_analytical(ee, analysis, config, stage)
        sanity = sanity_check_area(stage, metrics)
        progress("display_layers", metrics=metrics, numerical_consistency=sanity)
        LOGGER.info("Exporting six durable map layers")
        layers = export_layers(ee, analysis, config, stage, metrics)
        reference = {"status": "not_loaded", "url": None, "label": "BC Wildfire Service historical perimeter",
                     "source_url": "https://catalogue.data.gov.bc.ca/dataset/bc-wildfire-fire-perimeters-historical"}
        if reference_path:
            progress("reference_comparison", metrics=metrics, numerical_consistency=sanity)
            LOGGER.info("Computing descriptive reference overlap")
            reference = compare_reference(stage, reference_path)
        manifest = build_manifest(config, analysis, metrics, layers, assets, sanity, reference)
        if acquisition_selection is not None:
            (stage/"acquisition_selection_report.json").write_bytes(selection_report_bytes)
            manifest["acquisition_selection"] = acquisition_selection
            manifest["validation"]["review_status"] = "unapproved"
            manifest["limitations"].extend(acquisition_selection.get("temporal_limitations", []))
        schema_path = Path(__file__).resolve().parents[1]/"data/manifest.schema.json"
        if schema_path.exists():
            import jsonschema
            jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")), format_checker=jsonschema.FormatChecker()).validate(manifest)
        write_manifest_atomic(stage, manifest)
        # Output never exists until every scientific result and durable asset succeeded.
        stage.rename(output)
        LOGGER.info("Validated export complete: %s", output)
        return manifest
    except Exception:
        # Only our fresh temporary stage is removed. Existing exports are never changed.
        shutil.rmtree(stage)
        raise
