"""Bounded, authenticated coverage investigation; never a ready science export.

Uses the existing mask without changing its settings. Areas are evaluated on
the existing 20 m grid, simultaneously for the AOI and the reference polygon.
Only two reductions are submitted at once; image pixels never use getInfo().

Primary references:
https://developers.google.com/earth-engine/apidocs/ee-image-reduceregion
https://developers.google.com/earth-engine/apidocs/ee-image-reduceresolution
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED
https://sentiwiki.copernicus.eu/web/s2-processing
"""

from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import time

from .event import ANALYSIS_CRS, ANALYSIS_TRANSFORM, CLOUD_DATASET, S2_DATASET, Config

LOGGER = logging.getLogger(__name__)
MAX_PIXELS = 100_000_000
BATCH_SIZE = 2
STAGES = ("footprint", "raw_input", "scl_45", "qa_available", "cs_pass", "final")
AREA_BANDS = STAGES + tuple(f"scl_{n:02d}" for n in range(1, 12))


def _reduce_options(aoi):
    return dict(geometry=aoi, crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM,
                maxPixels=MAX_PIXELS, bestEffort=False, tileScale=4)


def load_reference(path):
    """Verify the bundled public reference before constructing server geometry."""
    path = Path(path)
    if path.stat().st_size > 5_000_000:
        raise ValueError("Reference exceeds the 5 MB bound.")
    payload = path.read_bytes()
    data = json.loads(payload)
    provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != provenance.get("sha256"):
        raise ValueError(
            "Reference digest does not match its provenance. Restore the original bundled "
            "reference files; do not replace the recorded digest with a new one. "
            f"Expected {provenance.get('sha256')}; received {actual_digest}."
        )
    if provenance.get("event_id") != "mcdougall-creek-2023":
        raise ValueError("Unexpected reference event.")
    if data.get("type") != "FeatureCollection" or not 1 <= len(data.get("features", [])) <= 100:
        raise ValueError("Expected a bounded reference FeatureCollection.")
    for feature in data["features"]:
        props = feature.get("properties", {})
        if props.get("FIRE_NUMBER") != "K52767" or props.get("FIRE_YEAR") != 2023:
            raise ValueError("Unexpected reference fire identifier.")
        if feature.get("geometry", {}).get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("Reference geometry must be a polygon.")
    return data, provenance


def _reference_geometry(ee, reference):
    """Send verified geometry only; retain all source properties in the local file.

    ArcGIS field names such as SHAPE.AREA/SHAPE.LEN are valid source metadata
    but rejected as Earth Engine feature property names. No property is needed
    for this region calculation. Coordinates and geometry defaults are unchanged.
    """
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(feature["geometry"]), {})
        for feature in reference["features"]
    ]).geometry()


def _failure_message(error):
    """Keep provider diagnostics useful without writing URLs or credentials."""
    message = re.sub(r"https?://[^\s<>\"']+", "[URL redacted]", str(error))
    message = re.sub(r"projects/[A-Za-z0-9_.:-]+", "projects/[redacted]", message)
    message = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", message)
    message = re.sub(r"(?i)(\b(?:access_token|refresh_token|token|api_key|signature)\b\s*[=:]\s*[\"']?)[^\s\"',;]+", r"\1[redacted]", message)
    return " ".join(message.split())[:1000] or type(error).__name__


def stage_masks(ee, image, config):
    """Independent stages, not sequential attrition: cs_pass does not require SCL.

    raw_input requires all four 10 m input bands at all child pixels and B12.
    final combines this input support, SCL4/5, and cs_cdf >= threshold. It is
    scene-level mask support before median compositing/NBR denominator checks.
    Missing Cloud Score+ has zero QA/final support, never an unmasked fallback.
    """
    image = ee.Image(image)
    native = image.select("B12").projection()
    raw10 = image.select(["B4", "B3", "B2", "B8"]).mask().reduce(ee.Reducer.min())

    def all_children(mask):
        return (mask.unmask(0, sameFootprint=False)
                .reduceResolution(reducer=ee.Reducer.min(), maxPixels=64)
                .reproject(crs=native))

    missing = (image.select("B4").multiply(0).rename("cs_cdf")
               .updateMask(ee.Image.constant(0)))
    qa = ee.Image(ee.Algorithms.If(
        image.get("cloud_score_image"),
        ee.Image(image.get("cloud_score_image")).select("cs_cdf"), missing,
    ))
    raw = all_children(raw10).And(image.select("B12").mask())
    scl = image.select("SCL")
    land = scl.eq(4).Or(scl.eq(5))
    qa_available = all_children(qa.mask())
    clear = all_children(qa.gte(config.cloud_score_threshold).And(raw10))
    footprint = ee.Image.constant(1).clip(image.geometry())
    masks = [footprint, raw, raw.And(land), raw.And(qa_available),
             raw.And(clear), raw.And(clear).And(land)]
    masks.extend(raw.And(scl.eq(n)) for n in range(1, 12))
    return (ee.Image.cat([mask.rename(name) for name, mask in zip(AREA_BANDS, masks, strict=True)])
            .unmask(0, sameFootprint=False)
            .reproject(crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM))


def _area_feature(ee, masks, aoi, reference_mask, properties):
    area = ee.Image.pixelArea().divide(10000)
    bands = []
    for name in AREA_BANDS:
        mask = masks.select(name)
        bands.extend([area.multiply(mask).rename(f"aoi_{name}_ha"),
                      area.multiply(mask).multiply(reference_mask).rename(f"reference_{name}_ha")])
    reduced = ee.Image.cat(bands).reduceRegion(reducer=ee.Reducer.sum(), **_reduce_options(aoi))
    return ee.Feature(None, reduced).set(properties)


def _joined_collection(ee, aoi, window, max_scenes):
    # Deliberately inventory all scene-cloud percentages. The production <=80
    # filter is compared in the report, not silently changed for the pipeline.
    raw = (ee.ImageCollection(S2_DATASET).filterBounds(aoi)
           .filterDate(window.start, window.end_exclusive).sort("system:time_start"))
    count = int(raw.size().getInfo())
    if count == 0:
        raise ValueError(f"No scenes in {window.start} to {window.end_exclusive}.")
    if count > max_scenes:
        raise ValueError(f"{count} scenes exceeds diagnostic bound {max_scenes}; narrow the dates.")
    cloud = (ee.ImageCollection(CLOUD_DATASET).filterBounds(aoi)
             .filterDate(window.start, window.end_exclusive))
    joined = ee.ImageCollection(ee.Join.saveFirst("cloud_score_image", outer=True).apply(
        primary=raw, secondary=cloud,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )).sort("system:time_start")
    return joined, count


def _check_deadline(deadline_at):
    if time.monotonic() > deadline_at:
        raise TimeoutError("Diagnostic time budget reached; partial report is preserved. Re-run with narrower dates.")


def _rows(ee, features, *, label, deadline_at, on_batch=None):
    rows = []
    for start in range(0, len(features), BATCH_SIZE):
        _check_deadline(deadline_at)
        batch = features[start:start + BATCH_SIZE]
        LOGGER.info("%s: rows %d–%d of %d", label, start+1, start+len(batch), len(features))
        result = ee.FeatureCollection(batch).getInfo()
        returned = result.get("features", [])
        if len(returned) != len(batch):
            raise RuntimeError("Unexpected diagnostic response row count.")
        validated = []
        for item in returned:
            props = item.get("properties", {})
            for key, value in props.items():
                if key.endswith("_ha") and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError(f"Invalid diagnostic area {key}={value!r}.")
            validated.append(props)
        rows.extend(validated)
        if on_batch:
            on_batch(validated)
    return rows


def _scene_properties(ee, image, period, config):
    image = ee.Image(image)
    cloud = image.get("CLOUDY_PIXEL_PERCENTAGE")
    return {
        "period": period, "row_kind": "scene", "image_id": image.id(),
        "date": image.date().format("YYYY-MM-dd"),
        "cloudy_pixel_percentage": cloud,
        "passes_scene_cloud_filter": ee.Number(cloud).lte(config.scene_cloud_limit),
        "has_cloud_score_match": ee.Algorithms.If(image.get("cloud_score_image"), True, False),
    }


def observation_summary(rows):
    """Counts mean selected granules and unique dates, never per-pixel counts."""
    scenes = [r for r in rows if r.get("row_kind") == "scene"]
    selected = [r for r in scenes if r.get("passes_scene_cloud_filter") and r.get("has_cloud_score_match")]
    return {
        "catalog_granules": len(scenes), "selected_matched_granules": len(selected),
        "catalog_unique_dates": len({r["date"] for r in scenes}),
        "selected_unique_dates": len({r["date"] for r in selected}),
        "missing_qa_granules": sum(not r.get("has_cloud_score_match") for r in scenes),
        "spatially_empty_raw_granules": sum(r.get("aoi_raw_input_ha", 0) <= 0 for r in scenes),
        "interpretation": "Granules on overlapping tiles can duplicate an acquisition; daily maxima count each date at most once per pixel.",
    }


def run_diagnostics(ee, output, reference_path, *, config=None, max_seconds=1800):
    """Compute a report only. OAuth credentials and project ID are not written."""
    config = config or Config()
    if not 60 <= max_seconds <= 3600:
        raise ValueError("Diagnostic runtime budget must be 60–3600 seconds.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "report_type": "coverage_diagnostics", "status": "in_progress", "phase": "reference_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(), "settings": config.to_dict(),
        "analysis_grid": {"crs": ANALYSIS_CRS, "transform": ANALYSIS_TRANSFORM, "scale_m": 20,
                          "max_pixels_per_reduction": MAX_PIXELS, "best_effort": False},
        "regions": {}, "reference_sha256": None,
        "periods": {}, "rows": [],
        "notes": [
            "Diagnostic only; no ready manifest, candidate area, new mask or new dates are produced.",
            "SCL and CS+ stages are independent tests on raw support, not successive losses. Final is their intersection.",
            "SCL class areas exclude raw-input failures. Class0 is already masked in the source SCL product.",
            "Daily SCL class areas are per-class unions and can overlap if same-date granules disagree; they are not a single categorical mosaic.",
            "The reference is hand-sketched context, not pixel ground truth. Missing observations are not unburned.",
            "Per-date union is a maximum over same-date granules. Unique-date count histograms avoid duplicate tile observations.",
            "Final diagnostic support is the production scene mask before median/NBR arithmetic. It may differ slightly from exported composite NBR coverage.",
        ],
    }
    deadline_at = time.monotonic() + max_seconds

    def save():
        (output/"coverage_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if report["rows"]:
            fields = sorted({key for row in report["rows"] for key in row})
            with (output/"coverage_by_scene_and_date.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(report["rows"])

    save()
    try:
        reference, provenance = load_reference(reference_path)
        report["reference_sha256"] = provenance["sha256"]
        report["phase"] = "region_totals"
        save()
        aoi = ee.Geometry.Rectangle(list(config.aoi_bbox), proj="EPSG:4326", geodesic=False)
        reference_geometry = _reference_geometry(ee, reference)
        reference_mask = (ee.Image.constant(1).clip(reference_geometry).unmask(0, sameFootprint=False)
                          .reproject(crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM))
        denominator = ee.Image.pixelArea().divide(10000)
        _check_deadline(deadline_at)
        report["regions"] = ee.Image.cat([
            denominator.rename("aoi_area_ha"),
            denominator.multiply(reference_mask).rename("reference_area_in_aoi_ha"),
        ]).reduceRegion(reducer=ee.Reducer.sum(), **_reduce_options(aoi)).getInfo()
        save()
        for period in ("pre", "post"):
            report["phase"] = f"{period}_scene_inventory"
            save()
            window = getattr(config, period)
            _check_deadline(deadline_at)
            joined, count = _joined_collection(ee, aoi, window, config.max_scenes_per_window)
            image_list = joined.toList(count)
            features = [_area_feature(ee, stage_masks(ee, image_list.get(i), config), aoi, reference_mask,
                                     _scene_properties(ee, image_list.get(i), period, config)) for i in range(count)]
            def checkpoint(batch):
                report["rows"].extend(batch)
                save()

            rows = _rows(ee, features, label=f"{period} granules", deadline_at=deadline_at, on_batch=checkpoint)
            info = observation_summary(rows)
            report["periods"][period] = info
            save()
            daily_masks = []
            daily_features = []
            report["phase"] = f"{period}_daily_unions"
            save()
            for date in sorted({row["date"] for row in rows}):
                day = joined.filterDate(date, ee.Date(date).advance(1, "day"))
                all_masks = day.map(lambda image: stage_masks(ee, image, config)).max()
                selected = day.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.scene_cloud_limit))
                # Excluding unmatched scenes is identical to zero final support.
                selected = selected.filter(ee.Filter.notNull(["cloud_score_image"]))
                has_selected = any(row["date"] == date and row.get("passes_scene_cloud_filter") and
                                   row.get("has_cloud_score_match") for row in rows)
                selected_masks = (selected.map(lambda image: stage_masks(ee, image, config)).max()
                                  if has_selected else ee.Image.constant([0]*len(AREA_BANDS)).rename(list(AREA_BANDS)))
                for selection, masks in (("all_scene_cloud_percentages", all_masks), ("production_selected", selected_masks)):
                    props = {"period": period, "row_kind": "daily_union", "date": date,
                             "selection": selection}
                    daily_features.append(_area_feature(ee, masks, aoi, reference_mask, props))
                daily_masks.append(selected_masks.select("final").unmask(0, sameFootprint=False))
            _rows(ee, daily_features, label=f"{period} daily unions", deadline_at=deadline_at, on_batch=checkpoint)
            count_image = ee.ImageCollection.fromImages(daily_masks).sum().rename("valid_dates").toInt()
            report["phase"] = f"{period}_unique_date_counts"
            save()
            _check_deadline(deadline_at)
            grouped = ee.Image.cat([denominator.rename("aoi_area_ha"),
                                   denominator.multiply(reference_mask).rename("reference_area_ha"), count_image]).reduceRegion(
                reducer=ee.Reducer.sum().repeat(2).group(groupField=2, groupName="valid_dates"),
                **_reduce_options(aoi)).getInfo()
            info["unique_date_count_area_histogram"] = [
                {"valid_dates": group["valid_dates"], "aoi_area_ha": group["sum"][0],
                 "reference_area_ha": group["sum"][1]} for group in grouped.get("groups", [])
            ]
            save()
        report["status"] = "complete_diagnostic_only"
        report["phase"] = "complete"
        save()
        return report
    except Exception as error:
        report["status"] = "incomplete_diagnostic_only"
        report["failure_type"] = type(error).__name__
        report["failure_message"] = _failure_message(error)
        save()
        raise
