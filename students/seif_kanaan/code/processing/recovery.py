"""Predeclared coverage-only acquisition recovery for this bounded pilot.

The October 21 diagnostic showed that the scene-cloud <=80 inventory discarded
usable dates. Its original post-window, even without that inventory filter,
cannot satisfy the existing 40% AOI guard. This experiment keeps the pixel mask,
20 m grid and dNBR threshold fixed, includes all scene-cloud percentages, and
tests at most three declared date windows. It never selects on candidate area
or agreement with the historical perimeter.

The reference coverage guard is a project triage criterion, not an accuracy
standard. A selected export remains exploratory and unapproved for publication.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import time

from .diagnostics import _failure_message, _reference_geometry, load_reference
from .event import ANALYSIS_CRS, ANALYSIS_TRANSFORM, CLOUD_DATASET, S2_DATASET, Config, Window
from .export import check_display_export
from .metrics import MAX_REDUCTION_PIXELS
from .optical import Analysis, _nbr, _prepare_scene
from .pipeline import export_analysis

LOGGER = logging.getLogger(__name__)
MIN_SHARED_AOI_FRACTION = 0.40
MIN_SHARED_REFERENCE_FRACTION = 0.40
COVERAGE_METHOD = "Explicit fractional NBR/reference support; nested mask minimum; weighted pixelArea sum on the fixed 20 m grid"
DIAGNOSTIC_REPORT_SHA256 = "209ebd01c1bab8693e48f6fd4aa452a6b20856049067d5c21a5a83c991dfbe5b"
DIAGNOSTIC_ZIP_SHA256 = "07b0afaededee763750598c4ab83c2854581b87fcf33ecaac9b9235fa4a806a4"


@dataclass(frozen=True)
class Profile:
    id: str
    window: Window
    temporal_context: str


PROFILES = (
    Profile("september_30_days", Window("2023-09-01", "2023-10-01"),
            "September observations may span ongoing fire activity and its aftermath; this is not a final-fire snapshot."),
    Profile("late_august_21_days", Window("2023-08-25", "2023-09-15"),
            "Late-August/early-September observations overlap the active fire period and can omit later burning."),
    Profile("late_august_to_mid_october_52_days", Window("2023-08-25", "2023-10-16"),
            "Last resort: the 52-day median spans ongoing fire, aftermath and substantial seasonal change; interpret spectral change cautiously."),
)


class WindowUnavailable(ValueError):
    """An expected empty scene/QA inventory; never a provider/bounds failure."""


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_info(value, deadline_at):
    if time.monotonic() > deadline_at:
        raise TimeoutError("Recovery selection budget reached; the checkpoint report is preserved.")
    return value.getInfo()


def _collection_with_budget(ee, aoi, window, config, deadline_at):
    """The production scene preparation, with a deadline before every query."""
    raw = (ee.ImageCollection(S2_DATASET).filterBounds(aoi)
           .filterDate(window.start, window.end_exclusive)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.scene_cloud_limit)))
    raw_count = int(_get_info(raw.size(), deadline_at))
    if raw_count == 0:
        raise WindowUnavailable(f"No Sentinel-2 scenes in {window.start}–{window.end_exclusive} exclusive.")
    if not 1 <= raw_count <= config.max_scenes_per_window:
        raise ValueError(f"Window {window.start} has {raw_count} scenes; required 1–{config.max_scenes_per_window}.")
    cloud = ee.ImageCollection(CLOUD_DATASET).filterBounds(aoi).filterDate(window.start, window.end_exclusive)
    joined = ee.ImageCollection(ee.Join.saveFirst("cloud_score_image").apply(
        primary=raw, secondary=cloud,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )).filter(ee.Filter.notNull(["cloud_score_image"])).sort("system:time_start")
    count = int(_get_info(joined.size(), deadline_at))
    if count == 0:
        raise WindowUnavailable(f"No usable Cloud Score+ matches in {window.start}–{window.end_exclusive} exclusive; no unmasked fallback.")
    if not 1 <= count <= raw_count:
        raise ValueError("Inconsistent Cloud Score+ scene count.")
    metadata = _get_info(ee.Dictionary({
        "indices": joined.aggregate_array("system:index"),
        "timestamps": joined.aggregate_array("system:time_start"),
    }), deadline_at)
    indices, timestamps = metadata["indices"], metadata["timestamps"]
    if len(indices) != count or len(timestamps) != count:
        raise RuntimeError("Scene provenance count mismatch.")
    images = [{
        "id": f"{S2_DATASET}/{index}", "cloud_score_id": f"{CLOUD_DATASET}/{index}",
        "acquired_at": datetime.fromtimestamp(ms/1000, timezone.utc).isoformat().replace("+00:00", "Z"),
    } for index, ms in zip(indices, timestamps, strict=True)]
    composite = (joined.map(lambda image: _prepare_scene(ee, image, config)).median()
                 .setDefaultProjection(ANALYSIS_CRS, ANALYSIS_TRANSFORM).clip(aoi))
    dates = sorted({image["acquired_at"][:10] for image in images})
    info = {
        "start": window.start, "end_exclusive": window.end_exclusive,
        "image_count": count, "scene_count_before_cloud_join": raw_count,
        "scenes_missing_cloud_score": raw_count-count, "acquisitions": dates,
        "source_images": images,
        "image_count_definition": "Selected spatially intersecting granules with Cloud Score+ matches; not independent per-pixel observations.",
        "unique_date_count": len(dates),
        "date_count_definition": "Catalog dates among selected granules. Valid dates can be fewer at a pixel; overlapping tiles do not add independent dates.",
    }
    return composite, info


def _reduce_options(aoi):
    return dict(geometry=aoi, crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM,
                maxPixels=MAX_REDUCTION_PIXELS, bestEffort=False, tileScale=4)


def _reference_support(ee, reference_geometry):
    """Reference support as fully defined fractional DATA on the analysis grid.

    A clipped constant still has value 1 at its partial boundary; the fraction
    lives in its mask. unmask(0) fills absent pixels but retains those fractional
    masks. Read the mask before filling missing support, so later operations
    cannot replace or promote an implicit fractional polygon mask.
    """
    return (ee.Image.constant(1)
            .reproject(crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM)
            .clip(reference_geometry).mask().unmask(0, sameFootprint=False))


def _region_graph(ee, aoi, reference_support):
    area = ee.Image.pixelArea().divide(10000)
    return ee.Image.cat([
        area.rename("aoi_area_ha"),
        area.updateMask(reference_support).rename("reference_area_in_aoi_ha"),
    ]).reduceRegion(reducer=ee.Reducer.sum(), **_reduce_options(aoi))


def _coverage_graph(ee, aoi, reference_support, pre_nbr, post_nbr):
    """Shared actual composite NBR masks; no dNBR or candidate expression."""
    area = ee.Image.pixelArea().divide(10000)
    # clip() can leave fractional edge masks. Boolean And promotes any positive
    # pair to 1 and can make shared area exceed either individual weighted area.
    # Minimum preserves their fractional intersection and the subset bounds.
    pre_support = pre_nbr.mask().unmask(0, sameFootprint=False)
    post_support = post_nbr.mask().unmask(0, sameFootprint=False)
    shared = pre_support.min(post_support)
    # Combine fractional support VALUES first, then mask fresh area exactly once.
    # updateMask on a reference-weighted area can replace its fractional mask
    # wherever nonzero, inflating the reference edge beyond its denominator.
    shared_reference = shared.min(reference_support)
    return ee.Image.cat([
        area.updateMask(shared).rename("shared_aoi_area_ha"),
        area.updateMask(shared_reference).rename("shared_reference_area_ha"),
        area.updateMask(pre_support).rename("pre_valid_area_ha"),
        area.updateMask(post_support).rename("post_valid_area_ha"),
    ]).reduceRegion(reducer=ee.Reducer.sum(), **_reduce_options(aoi))


def _mask_check_graph(ee):
    """Tiny, catalog-independent live check of the actual server mask rules."""
    aoi = ee.Geometry.Rectangle([300000, 5500000, 300200, 5500200],
                                proj=ANALYSIS_CRS, geodesic=False)
    reference = ee.Geometry.Rectangle([300013, 5500013, 300173, 5500173],
                                      proj=ANALYSIS_CRS, geodesic=False)
    support = _reference_support(ee, reference)
    clear = (ee.Image.constant(.4).rename("nbr")
             .reproject(crs=ANALYSIS_CRS, crsTransform=ANALYSIS_TRANSFORM).clip(aoi))
    pre, post = clear.updateMask(ee.Image.constant(.25)), clear.updateMask(ee.Image.constant(.75))
    production_mask = pre.subtract(post).mask()
    production_shared = (ee.Image.pixelArea().divide(10000).updateMask(production_mask)
                         .rename("valid_area_ha").reduceRegion(
                             reducer=ee.Reducer.sum(), **_reduce_options(aoi)))
    return ee.Dictionary({
        "regions": _region_graph(ee, aoi, support),
        "clear": _coverage_graph(ee, aoi, support, clear, clear),
        "fractional": _coverage_graph(ee, aoi, support, pre, post),
        "production_shared": production_shared,
    })


def _validate_mask_check(values):
    """Verify subset bounds and a completely observed fractional reference."""
    regions = values["regions"]
    clear = assess_coverage(values["clear"], regions)
    fractional = assess_coverage(values["fractional"], regions)
    if not (math.isclose(clear["shared_valid_fraction"], 1, abs_tol=1e-6)
            and math.isclose(clear["reference_valid_fraction"], 1, abs_tol=1e-6)):
        raise ValueError("Synthetic mask self-check failed: full support must retain the reference total.")
    # The provider quantizes pixel-intersection weights. This synthetic interval
    # allows that quantization; real coverage guards are unchanged and exact.
    if not .24 <= fractional["shared_valid_fraction"] <= .26:
        raise ValueError("Synthetic mask self-check failed: fractional support was not retained.")
    if not math.isclose(fractional["shared_aoi_area_ha"],
                        values["production_shared"]["valid_area_ha"], rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("Synthetic mask self-check failed: selection and production support differ.")
    return {**values, "status": "passed",
            "note": "Synthetic constant images only; checks server mask arithmetic, not satellite coverage or accuracy."}


def assess_coverage(values, regions):
    """Validate areas, then apply the two unchanged coverage-only gates."""
    values, regions = dict(values), dict(regions)
    for name in ("aoi_area_ha", "reference_area_in_aoi_ha"):
        number = regions.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"Invalid recovery denominator {name}.")
    aoi, reference = regions["aoi_area_ha"], regions["reference_area_in_aoi_ha"]
    if reference > aoi+1e-6:
        raise ValueError("Reference area exceeds the AOI.")
    for name in ("shared_aoi_area_ha", "shared_reference_area_ha", "pre_valid_area_ha", "post_valid_area_ha"):
        if values.get(name) is None:
            values[name] = 0.0  # Empty valid masks reduce to null, never positive coverage.
        number = values[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError(f"Invalid recovery area {name}.")
    shared, shared_reference = values["shared_aoi_area_ha"], values["shared_reference_area_ha"]
    violations = []
    if shared > aoi+1e-6:
        violations.append("shared_aoi_area_ha exceeds aoi_area_ha")
    if shared_reference > min(reference, shared)+1e-6:
        violations.append("shared_reference_area_ha exceeds reference or shared AOI area")
    for name in ("pre_valid_area_ha", "post_valid_area_ha"):
        if not shared-1e-6 <= values[name] <= aoi+1e-6:
            violations.append(f"{name} is outside shared AOI/AOI bounds")
    if violations:
        raise ValueError("Recovery areas violate the shared/individual region bounds: " + "; ".join(violations) + ".")
    aoi_fraction, reference_fraction = shared/aoi, shared_reference/reference
    return {**values, "shared_valid_fraction": aoi_fraction,
            "reference_valid_fraction": reference_fraction,
            "passes_coverage_guards": aoi_fraction >= MIN_SHARED_AOI_FRACTION
            and reference_fraction >= MIN_SHARED_REFERENCE_FRACTION}


def _selection_metadata(report, digest):
    return {
        "method": "First eligible predeclared window; coverage only",
        "selected_profile": report["selected_profile"],
        "selection_criteria": report["policy"],
        "evaluated_windows": report["evaluated_windows"],
        "report_url": "acquisition_selection_report.json", "report_sha256": digest,
        "diagnostic_report_sha256": DIAGNOSTIC_REPORT_SHA256,
        "diagnostic_zip_sha256": DIAGNOSTIC_ZIP_SHA256,
        "review_status": "unapproved",
        "temporal_limitations": [
            report["selected_profile"]["temporal_context"],
            "Date windows were changed after a documented coverage failure. Scene inventory includes all cloud percentages; pixel-level CS+ >=0.6 and SCL4/5 remain unchanged.",
            "Both 40% coverage guards are project triage criteria, not proof of accuracy. Candidate change is not official final fire area.",
        ],
    }


def _profile_dict(profile):
    return {"id": profile.id, "start": profile.window.start,
            "end_exclusive": profile.window.end_exclusive,
            "temporal_context": profile.temporal_context}


def _new_report():
    config = replace(Config(), scene_cloud_limit=100)
    return {
        "report_type": "bounded_coverage_recovery", "report_version": 1,
        "status": "in_progress", "phase": "reference_validation", "generated_at": _now(),
        "attempt": 1, "settings": config.to_dict(), "export_directory": None,
        "policy": {
            "minimum_shared_aoi_fraction": MIN_SHARED_AOI_FRACTION,
            "minimum_shared_reference_fraction": MIN_SHARED_REFERENCE_FRACTION,
            "decision_inputs": ["shared_valid_fraction", "reference_valid_fraction"],
            "rule": "First qualifying window in the fixed order. No candidate area, dNBR value or reference agreement is evaluated for selection.",
            "interpretation": "Project coverage triage; neither guard is an accuracy claim.",
        },
        "declared_profiles": [_profile_dict(profile) for profile in PROFILES],
        "analysis_grid": {"crs": ANALYSIS_CRS, "transform": ANALYSIS_TRANSFORM,
                          "scale_m": 20, "max_pixels": MAX_REDUCTION_PIXELS, "best_effort": False},
        "coverage_method": COVERAGE_METHOD,
        "evidence": {"report_sha256": DIAGNOSTIC_REPORT_SHA256, "zip_sha256": DIAGNOSTIC_ZIP_SHA256,
                     "original_post_all_scene_union_upper_bounds": {
                         "aoi_fraction": 0.338661, "reference_fraction": 0.192598},
                     "interpretation": "The original post window cannot satisfy 40% shared AOI coverage even at its scene-mask union upper bound; catalog-cloud filtering also discarded usable dates."},
        "regions": {}, "reference_sha256": None, "evaluated_windows": [], "selected_profile": None,
        "notes": [
            "The original Config defaults are preserved in the source. This separately declared experiment sets only scene_cloud_limit=100 and its enumerated post windows.",
            "The July 15–August 15 exclusive pre window is fixed and built once per attempt. Its composite is reused for each comparison.",
            "Only actual composite NBR validity is assessed before selection. Candidate and dNBR are constructed only after a window passes both guards.",
            "Fractional pre/post/reference support is represented as data, combined by minimum, and applied once to fresh pixelArea; weighted reductions also retain AOI boundary fractions.",
            "Granules and catalog acquisition dates are provenance counts, not per-pixel independent observation counts.",
            "The 30-minute budget covers selection, with a check before each server query. Durable tile export follows only if selection completes within budget.",
            "A resumed attempt preserves its previous evaluations but rebuilds graphs and rechecks coverage; cached JSON never authorizes an export.",
            "The historical perimeter is contextual and hand sketched. No-data is not unburned; no extrapolation to whole-fire area is permitted.",
        ],
    }


def run_recovery(ee, output, reference_path, *, max_seconds=1800, resume=False):
    """Evaluate at most three fixed windows and export the first eligible graph.

    ``output/recovery_report.json`` exists before any server request. Successful
    imports use the standalone ``output/export`` bundle; report-only outcomes
    deliberately have no ready manifest. Failure details exclude service URLs,
    OAuth secrets and project identifiers. Exceptions propagate after saving.
    """
    if not 60 <= max_seconds <= 1800:
        raise ValueError("Recovery selection budget must be 60–1800 seconds.")
    output = Path(output)
    report_path = output/"recovery_report.json"
    report = _new_report()
    if output.exists():
        if not resume or not report_path.is_file():
            raise FileExistsError("Recovery output exists; use a new directory or explicitly resume its checkpoint.")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("status") not in ("incomplete_recovery", "no_eligible_window"):
            raise ValueError("Only an incomplete or report-only recovery can be resumed.")
        for key in ("report_type", "report_version", "settings", "policy", "declared_profiles", "evidence", "coverage_method"):
            # Config's AOI is a tuple in memory and a JSON array on disk.
            if json.dumps(previous.get(key), sort_keys=True) != json.dumps(report[key], sort_keys=True):
                raise ValueError("Recovery checkpoint differs from this fixed experiment; start a new directory.")
        if (output/"export").exists():
            raise FileExistsError("An export already exists; recovery never overwrites it.")
        report["attempt"] = int(previous.get("attempt", 1))+1
        report["previous_evaluations"] = previous.get("evaluated_windows", [])
    else:
        output.mkdir(parents=True, exist_ok=False)
    deadline_at = time.monotonic()+max_seconds

    def save():
        report["updated_at"] = _now()
        temporary = output/".recovery_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(report_path)

    save()
    try:
        reference, provenance = load_reference(reference_path)
        report["reference_sha256"] = provenance["sha256"]
        report["phase"] = "synthetic_mask_self_check"
        save()
        report["mask_self_check"] = _get_info(_mask_check_graph(ee), deadline_at)
        save()
        report["mask_self_check"] = _validate_mask_check(report["mask_self_check"])
        save()
        report["phase"] = "synthetic_display_self_check"
        save()
        if time.monotonic() > deadline_at:
            raise TimeoutError("Recovery selection budget reached before display self-check; the checkpoint is preserved.")
        report["display_self_check"] = check_display_export(ee)
        save()
        config = replace(Config(), scene_cloud_limit=100)
        aoi = ee.Geometry.Rectangle(list(config.aoi_bbox), proj="EPSG:4326", geodesic=False)
        reference_geometry = _reference_geometry(ee, reference)
        reference_support = _reference_support(ee, reference_geometry)
        report["phase"] = "region_totals"
        save()
        report["regions"] = _get_info(_region_graph(ee, aoi, reference_support), deadline_at)
        # Validate denominators even before a candidate window can be considered.
        assess_coverage({}, report["regions"])
        report["phase"] = "pre_composite"
        save()
        pre, pre_info = _collection_with_budget(ee, aoi, config.pre, config, deadline_at)
        pre_nbr = _nbr(pre)
        report["pre"] = pre_info
        save()
        for profile in PROFILES:
            profile_config = replace(config, post=profile.window)
            report["phase"] = f"{profile.id}_composite"
            save()
            LOGGER.info("Coverage-only recovery: %s", profile.id)
            try:
                post, post_info = _collection_with_budget(ee, aoi, profile.window, profile_config, deadline_at)
            except WindowUnavailable as error:
                # An empty post inventory is a known ineligible profile, so the
                # declared search continues. Provider errors and bound failures
                # remain fatal; they are never mislabeled as zero coverage.
                report["evaluated_windows"].append({
                    **_profile_dict(profile), "coverage_status": "no_usable_inventory",
                    "shared_valid_fraction": 0.0, "reference_valid_fraction": 0.0,
                    "passes_coverage_guards": False, "skip_reason": str(error),
                })
                save()
                continue
            post_nbr = _nbr(post)
            report["phase"] = f"{profile.id}_shared_coverage"
            # Inventory survives a provider error; raw measurements survive a
            # local validation error. Neither state authorizes selection/export.
            row = {**_profile_dict(profile), "coverage_status": "pending",
                   "post": post_info}
            report["evaluated_windows"].append(row)
            save()
            raw_coverage = _get_info(
                _coverage_graph(ee, aoi, reference_support, pre_nbr, post_nbr), deadline_at)
            row["raw_coverage"] = {
                name: (str(value) if isinstance(value, float) and not math.isfinite(value) else value)
                for name, value in raw_coverage.items()
            }
            row["coverage_status"] = "measured_unvalidated"
            save()
            try:
                coverage = assess_coverage(raw_coverage, report["regions"])
            except ValueError:
                row["coverage_status"] = "invalid_measurement"
                raise
            row.update(coverage)
            row["coverage_status"] = "validated"
            save()
            if not coverage["passes_coverage_guards"]:
                continue
            if time.monotonic() > deadline_at:
                raise TimeoutError("Recovery selection budget reached before export; the checkpoint report is preserved.")
            # First qualifying window wins. Later profiles and candidate outcomes
            # cannot influence this already checkpointed selection.
            report["selected_profile"] = _profile_dict(profile)
            report["status"] = "selected_for_export"
            report["phase"] = "export"
            save()
            report_bytes = report_path.read_bytes()
            selection = _selection_metadata(report, hashlib.sha256(report_bytes).hexdigest())
            dnbr = pre_nbr.subtract(post_nbr).rename("dnbr")
            candidate = dnbr.gte(profile_config.threshold).rename("candidate").toByte()
            analysis = Analysis(aoi, pre, post, pre_nbr, post_nbr, dnbr, candidate,
                                {"pre": pre_info, "post": post_info})

            def export_progress(phase, details):
                report["export_progress"] = {"phase": phase, **details}
                save()

            export_analysis(ee, output/"export", analysis, config=profile_config,
                            reference_path=reference_path, acquisition_selection=selection,
                            selection_report_bytes=report_bytes, progress_callback=export_progress)
            report["status"] = "export_complete"
            report["phase"] = "complete"
            report["export_directory"] = "export"
            save()
            return report
        report["status"] = "no_eligible_window"
        report["phase"] = "complete"
        save()
        return report
    except Exception as error:
        report["status"] = "incomplete_recovery"
        report["failure_type"] = type(error).__name__
        report["failure_message"] = _failure_message(error)
        save()
        raise
