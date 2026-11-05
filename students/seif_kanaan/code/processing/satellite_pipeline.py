"""Authenticated satellite processing -> unreviewed rasters/PNGs -> portable ZIP.

No website import or publication. Reuses the reviewed wildfire pixel transport
without modifying its immutable analysis, notebooks or data.
"""
from datetime import datetime, timezone
import importlib
import json
import logging
import math
from pathlib import Path
import re
import time
import zipfile

import numpy as np
from PIL import Image
import rasterio

from .export import (NODATA, _rgba, check_display_export, download_image,
                     grid_for_bbox, safe_provider_error, sha256, tiff_to_png)
from .satellite_common import analysis_grid, grid_domain

LOGGER = logging.getLogger(__name__)
PILOTS = {key: "processing.satellite_" + key for key in ("flood", "earthquake", "tornado")}
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
SCHEMA = "canadian-disasters-satellite/1.0"


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def local_file(parent, value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("Unsafe bundle-relative path")
    path = (Path(parent) / value).resolve()
    if not path.is_relative_to(Path(parent).resolve()) or not path.is_file():
        raise ValueError("Missing or escaping bundle asset")
    return path


def _validate_analysis(analysis):
    if analysis.event.get("hazard") not in PILOTS or not re.fullmatch(r"[a-z0-9-]{3,100}", analysis.event.get("id", "")):
        raise ValueError("Invalid satellite pilot identity")
    if not analysis.event.get("sources") or not analysis.event.get("limitations"):
        raise ValueError("Satellite methods require sources and interpretation limits")
    if not 3 <= len(analysis.rasters) <= 48 or not 1 <= len(analysis.displays) <= 20:
        raise ValueError("Satellite export exceeds its bounded raster/layer contract")
    for key in analysis.rasters:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", key) or key in ("domain", "pixel_area_m2"):
            raise ValueError("Invalid or reserved analytical raster identifier")
    for key in (analysis.pre_valid_id, analysis.post_valid_id, analysis.shared_valid_id):
        if key not in analysis.rasters:
            raise ValueError("Missing required validity raster")
    if analysis.candidate_id is not None and analysis.candidate_id not in analysis.rasters:
        raise ValueError("Missing candidate raster")
    ids = [d["id"] for d in analysis.displays]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", key) for key in ids):
        raise ValueError("Invalid or repeated display identifier")
    # Prevent accidentally trying to JSON-encode an EE graph or credentials.
    json.dumps({"event": analysis.event, "provenance": analysis.provenance, "audit": analysis.audit}, allow_nan=False)


def _coverage_graph(ee, analysis, grid):
    transform = list(grid.transform)[:6]
    domain, region = grid_domain(ee, analysis.bbox, grid)
    area = ee.Image.pixelArea().reproject(crs=grid.crs, crsTransform=transform).toFloat().updateMask(domain).rename("pixel_area_m2")
    gates = analysis.provenance.get("coverage_gates")
    if gates is None:
        old = analysis.provenance.get("coverage_gate", {})
        gates = [{"population": old.get("raster", "domain"), "valid": analysis.shared_valid_id,
                  "min_fraction": old.get("min_shared_fraction", .8)}]
    if not isinstance(gates, list) or not 1 <= len(gates) <= 6:
        raise ValueError("Invalid coverage gates")
    ids = {analysis.pre_valid_id, analysis.post_valid_id, analysis.shared_valid_id}
    if "eligible" in analysis.rasters:
        ids.add("eligible")
    for gate in gates:
        if not isinstance(gate.get("min_fraction"), (float, int)) or isinstance(gate["min_fraction"], bool) or not 0 < gate["min_fraction"] <= 1:
            raise ValueError("Invalid coverage floor")
        ids.update([gate["population"], gate["valid"]])
    flags = {"domain": domain}
    for key in sorted(ids - {"domain"}):
        if key not in analysis.rasters:
            raise ValueError("Coverage gate refers to a missing raster")
        # Start from a global binary image so fractional source edge masks
        # cannot silently turn domain area into a fractional weighting scheme.
        flags[key] = domain.where(ee.Image(analysis.rasters[key]).unmask(0, sameFootprint=False).lt(.5), 0).rename(key).toByte()
    rasters = {**analysis.rasters, **flags, "pixel_area_m2": area}
    measures = {"domain_area_m2": domain,
                "pre_valid_area_m2": flags[analysis.pre_valid_id],
                "post_valid_area_m2": flags[analysis.post_valid_id],
                "shared_valid_area_m2": flags[analysis.shared_valid_id]}
    if "eligible" in flags:
        measures["eligible_area_m2"] = flags["eligible"]
    if analysis.candidate_id:
        measures["candidate_area_m2"] = domain.where(ee.Image(rasters[analysis.candidate_id]).unmask(0, sameFootprint=False).lt(.5), 0)
    for index, gate in enumerate(gates):
        measures[f"gate_{index}_population_m2"] = flags[gate["population"]]
        measures[f"gate_{index}_observed_m2"] = flags[gate["population"]].And(flags[gate["valid"]])
    stack = ee.Image.cat([area.updateMask(mask).rename(key) for key, mask in measures.items()])
    raw = stack.reduceRegion(reducer=ee.Reducer.sum(), geometry=region, crs=grid.crs,
                             crsTransform=transform, maxPixels=100_000_000,
                             bestEffort=False, tileScale=4).getInfo()
    metrics = {}
    for key in measures:
        value = raw.get(key)
        if value is None:
            value = 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Nonfinite or negative computed coverage")
        metrics[key] = float(value)
    metrics["area_convention"] = "Full WGS84 pixelArea values for cells whose centres fall inside the declared longitude/latitude box, evaluated on the exact anchored projected grid. Not fractionally clipped geographic boundary area."
    metrics["gate_results"] = []
    for index, gate in enumerate(gates):
        denominator = metrics[f"gate_{index}_population_m2"]
        numerator = metrics[f"gate_{index}_observed_m2"]
        fraction = numerator / denominator if denominator > 0 else None
        metrics["gate_results"].append({**gate, "population_area_m2": denominator, "observed_area_m2": numerator,
                                        "fraction": fraction, "passed": fraction is not None and fraction >= gate["min_fraction"]})
    metrics["shared_fraction_of_aoi"] = metrics["shared_valid_area_m2"] / metrics["domain_area_m2"] if metrics["domain_area_m2"] > 0 else None
    return rasters, flags, gates, metrics


def _assert_coverage(metrics):
    if metrics["domain_area_m2"] <= 0 or metrics["shared_valid_area_m2"] <= 0:
        raise ValueError("No shared satellite support in the analysis domain")
    if not all(row["population_area_m2"] > 0 and row["observed_area_m2"] / row["population_area_m2"] >= row["min_fraction"] for row in metrics["gate_results"]):
        raise ValueError("Predeclared coverage gates failed; inspect the saved coverage report. Dates/thresholds were not relaxed.")


def audit_arrays_common(arrays, manifest):
    """Independent calculations on the exported numerical TIFF samples."""
    domain = arrays["domain"] == 1
    if not np.isin(arrays["domain"], [0, 1]).all() or not domain.any():
        raise ValueError("Invalid or empty explicit domain raster")
    area = arrays["pixel_area_m2"]
    if not np.isfinite(area[domain]).all() or not (area[domain] > 0).all():
        raise ValueError("Pixel area is missing or nonpositive inside the domain")
    roles = manifest["validity"]
    flags = {}
    for key in {roles["pre"], roles["post"], roles["shared"], "domain", *manifest["coverage_flag_ids"]}:
        if not np.isin(arrays[key][domain], [0, 1]).all():
            raise ValueError(f"Nonbinary or missing validity flag: {key}")
        flags[key] = arrays[key] == 1
    shared = flags[roles["shared"]] & domain
    if (shared & ~(flags[roles["pre"]] & flags[roles["post"]])).any():
        raise ValueError("Shared support exceeds pre/post valid intersection")
    actual = {"domain_area_m2": float(area[domain].sum(dtype=np.float64)),
              "pre_valid_area_m2": float(area[domain & flags[roles["pre"]]].sum(dtype=np.float64)),
              "post_valid_area_m2": float(area[domain & flags[roles["post"]]].sum(dtype=np.float64)),
              "shared_valid_area_m2": float(area[shared].sum(dtype=np.float64))}
    if "eligible" in arrays:
        actual["eligible_area_m2"] = float(area[domain & (arrays["eligible"] == 1)].sum(dtype=np.float64))
    candidate_id = roles.get("candidate")
    if candidate_id:
        candidate = arrays[candidate_id]
        candidate_valid = domain & (candidate != NODATA)
        if not np.isin(candidate[candidate_valid], [0, 1]).all() or ((candidate == 1) & domain & ~shared).any():
            raise ValueError("Invalid candidate flags or candidate outside shared observations")
        actual["candidate_area_m2"] = float(area[domain & (candidate == 1)].sum(dtype=np.float64))
    for index, gate in enumerate(manifest["coverage_gates"]):
        population = domain & flags[gate["population"]]
        actual[f"gate_{index}_population_m2"] = float(area[population].sum(dtype=np.float64))
        actual[f"gate_{index}_observed_m2"] = float(area[population & flags[gate["valid"]]].sum(dtype=np.float64))
        denominator, numerator = actual[f"gate_{index}_population_m2"], actual[f"gate_{index}_observed_m2"]
        if denominator <= 0 or numerator / denominator < gate["min_fraction"]:
            raise ValueError("Exported numerical support fails its predeclared coverage gate")
        recorded = manifest["metrics"]["gate_results"][index]
        if (recorded["min_fraction"] != gate["min_fraction"] or
                abs(recorded["population_area_m2"] - denominator) > max(1.0, denominator*1e-6) or
                abs(recorded["observed_area_m2"] - numerator) > max(1.0, numerator*1e-6)):
            raise ValueError("Nested coverage report differs from exported numerical support")
    differences = {}
    for key, value in actual.items():
        expected = manifest["metrics"][key]
        difference = abs(value - expected)
        # Same centre rule and float32 source pixel areas, so no AOI boundary allowance.
        tolerance = max(1.0, expected * 1e-6)
        if difference > tolerance:
            raise ValueError(f"Export/EE area mismatch for {key}: {difference} m²")
        differences[key] = {"local_m2": value, "ee_m2": expected, "difference_m2": difference, "tolerance_m2": tolerance}
    _assert_coverage(manifest["metrics"])
    return {"status": "passed", "domain_pixels": int(domain.sum()), "shared_pixels": int(shared.sum()), "area_checks": differences,
            "scope": "Numerical implementation consistency; not independent hazard classification accuracy."}


def _numeric_asset(ee, image, key, target, grid, domain):
    path = target / f"{key}.tif"
    output = ee.Image(image).reproject(crs=grid.crs, crsTransform=list(grid.transform)[:6])
    if key != "domain":
        output = output.updateMask(domain)
    values, _ = download_image(output.toFloat().unmask(NODATA, sameFootprint=False), path, grid, bands=1, dtype="float32")
    with rasterio.open(path, "r+") as dataset:
        dataset.nodata = NODATA
    if values.shape != (1, grid.height, grid.width) or not np.isfinite(values).all():
        raise ValueError("Invalid numerical diagnostic/export samples")
    return values[0], {"id": key, "file": f"{target.name}/{path.name}", "sha256": sha256(path), "nodata": NODATA}


def _coverage_diagnostics(ee, rasters, flags, grid, directory, source_archive_sha256, checkpoint):
    """Retain observation evidence after failed gates, never a hazard product."""
    keys = set(flags) | {"domain", "pixel_area_m2"}
    keys |= {key for key in rasters if key.endswith(("_clear_count", "_observation_count", "_first_day", "_last_day"))}
    keys |= {key for key in ("original_pre_valid", "original_shared_valid", "worldcover_known_fraction", "worldcover_water_fraction") if key in rasters}
    target = directory / "coverage_diagnostics"
    target.mkdir()
    receipt = {"status": "incomplete_coverage_diagnostics_only", "publishing_approved": False,
               "review_status": "unapproved", "source_archive_sha256": source_archive_sha256,
               "grid": {"crs": grid.crs, "transform": list(grid.transform)[:6], "width": grid.width, "height": grid.height},
               "assets": [], "scope": "Observation support only; failed gates remain failed. No candidate or hazard-change rasters are exported here."}
    for key in sorted(keys):
        checkpoint("failed_coverage_diagnostics", raster=key)
        _, asset = _numeric_asset(ee, rasters[key], key, target, grid, flags["domain"])
        receipt["assets"].append(asset)
        write_json(directory / "coverage_diagnostics.json", receipt)
    return receipt


def export_pilot(ee, analysis, directory, *, source_archive_sha256, progress=None):
    try:
        return _export_pilot(ee, analysis, directory, source_archive_sha256=source_archive_sha256, progress=progress)
    except Exception as error:
        path = Path(directory) / "processing_report.json"
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            state.update(status="incomplete", failure_message=safe_provider_error(error), failed_at=now())
            write_json(path, state)
        raise


def _export_pilot(ee, analysis, directory, *, source_archive_sha256, progress=None):
    _validate_analysis(analysis)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    state = {"schema": SCHEMA, "event": analysis.event, "status": "processing", "phase": "source_inventory", "started_at": now(), "review_status": "unapproved"}
    def checkpoint(phase, **details):
        state.update(phase=phase, **details)
        write_json(directory / "processing_report.json", state)
        LOGGER.info("%s: %s", analysis.event["hazard"], phase)
        if progress:
            progress(phase, details)
    selection = {"event": analysis.event, "provenance": analysis.provenance, "audit_recipe": analysis.audit,
                 "source_archive_sha256": source_archive_sha256, "selection_frozen_at": now()}
    write_json(directory / "source_selection.json", selection)
    checkpoint("coverage_evaluation")
    grid = analysis_grid(analysis.bbox, analysis.crs, analysis.scale_m)
    rasters, flags, gates, metrics = _coverage_graph(ee, analysis, grid)
    try:
        _assert_coverage(metrics)
    except ValueError:
        # Candidate reducers are not a result when coverage fails. Do not expose
        # their unchecked area in a diagnostics-only receipt.
        diagnostic_metrics = {key: value for key, value in metrics.items() if key != "candidate_area_m2"}
        checkpoint("coverage_failed", metrics=diagnostic_metrics)
        try:
            diagnostics = _coverage_diagnostics(ee, rasters, flags, grid, directory, source_archive_sha256, checkpoint)
            checkpoint("coverage_failed", coverage_diagnostic_assets=len(diagnostics["assets"]))
        except Exception as diagnostic_error:
            checkpoint("coverage_failed", diagnostic_export_error=safe_provider_error(diagnostic_error))
        raise
    checkpoint("coverage_computed", metrics=metrics)
    manifest = {"schema": SCHEMA, "status": "export_complete_unreviewed", "review_status": "unapproved", "publishing_approved": False,
                "event": analysis.event, "generated_at": now(), "source_archive_sha256": source_archive_sha256,
                "source_selection": {"file": "source_selection.json", "sha256": sha256(directory / "source_selection.json")},
                "analysis_grid": {"crs": grid.crs, "scale_m": analysis.scale_m, "transform": list(grid.transform)[:6], "width": grid.width, "height": grid.height,
                                  "bbox": list(analysis.bbox), "anchor_transform": [analysis.scale_m, 0, 0, 0, -analysis.scale_m, 10_000_000]},
                "validity": {"pre": analysis.pre_valid_id, "post": analysis.post_valid_id, "shared": analysis.shared_valid_id, "candidate": analysis.candidate_id},
                "coverage_flag_ids": sorted(flags), "coverage_gates": gates, "metrics": metrics,
                "hazard_area_ha": None, "hazard_area_note": "No official event area or confirmed damage is inferred. Any candidate area is an exploratory satellite indicator only.",
                "audit_recipe": analysis.audit, "analytical_assets": [], "layers": []}
    target = directory / "rasters"; target.mkdir()
    arrays = {}
    for index, (key, image) in enumerate(rasters.items(), 1):
        checkpoint("numerical_rasters", raster=key, raster_index=index, raster_total=len(rasters))
        arrays[key], asset = _numeric_asset(ee, image, key, target, grid, flags["domain"])
        manifest["analytical_assets"].append(asset)
    checkpoint("independent_numerical_audit")
    common = audit_arrays_common(arrays, manifest)
    module = importlib.import_module(PILOTS[analysis.event["hazard"]])
    specific = module.audit_arrays(arrays, nodata=NODATA)
    display_grid = grid_for_bbox(analysis.bbox, "EPSG:3857", 60)
    (directory / "layers").mkdir()
    for spec in analysis.displays:
        checkpoint("display_layers", layer=spec["id"])
        tiff, png = directory / (spec["id"] + "_display.tif"), directory / "layers" / (spec["id"] + ".png")
        image = ee.Image(spec["image"]).updateMask(flags["domain"])
        download_image(_rgba(ee, image, spec["vis"]), tiff, display_grid, bands=4, dtype="uint8")
        bounds = tiff_to_png(tiff, png, display_grid, allow_empty=spec.get("allow_empty", False))
        tiff.unlink()
        manifest["layers"].append({key: spec[key] for key in ("id", "label", "legend", "units")})
        manifest["layers"][-1].update(file=f"layers/{png.name}", sha256=sha256(png), bounds=bounds,
                                      crs="EPSG:3857", pixel_size_map_m=60, width=display_grid.width, height=display_grid.height,
                                      resampling="nearest display sampling; statistics use numerical analysis grid")
    write_json(directory / "export_audit.json", {"status": "passed", "common": common, "hazard_specific": specific})
    manifest["export_audit"] = {"file": "export_audit.json", "sha256": sha256(directory / "export_audit.json")}
    # The ready-for-review manifest appears only after every numerical and PNG check.
    write_json(directory / "manifest.json", manifest)
    checkpoint("complete", status="export_complete_unreviewed", completed_at=now())
    return manifest


def run_batch(ee, output, *, hazards=("flood", "earthquake", "tornado"), source_archive_sha256="unspecified", max_seconds=10800):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh output directory to preserve prior results")
    if not hazards or len(set(hazards)) != len(hazards) or any(x not in PILOTS for x in hazards):
        raise ValueError("Select only the supported satellite pilots")
    if not re.fullmatch(r"[a-f0-9]{64}", source_archive_sha256):
        raise ValueError("A verified embedded source archive SHA-256 is required")
    output.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "processing", "started_at": now(), "source_archive_sha256": source_archive_sha256,
              "publishing_approved": False, "review_status": "unapproved", "events": [],
              "note": "All successful exports require independent return-ZIP audit and image review before website import or publication."}
    write_json(output / "run_report.json", report)
    started = time.monotonic()
    try:
        report["display_transport_self_check"] = check_display_export(ee)
    except Exception as error:
        report.update(status="diagnostics_only", failure_phase="display_transport_self_check", failure_message=safe_provider_error(error))
        write_json(output / "run_report.json", report)
        return report
    for hazard in hazards:
        row = {"hazard": hazard, "status": "processing", "phase": "build_satellite_graph", "started_at": now()}
        report["events"].append(row); write_json(output / "run_report.json", report)
        try:
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Batch time budget reached between requests; completed outputs are retained")
            module = importlib.import_module(PILOTS[hazard])
            analysis = module.build(ee)
            relative = "events/" + analysis.event["id"]
            row["directory"] = relative
            def progress(phase, details):
                row["phase"] = phase
                write_json(output / "run_report.json", report)
            manifest = export_pilot(ee, analysis, output / relative, source_archive_sha256=source_archive_sha256, progress=progress)
            row.update(status=manifest["status"], phase="complete", manifest_sha256=sha256(output / relative / "manifest.json"))
        except KeyboardInterrupt:
            row.update(status="interrupted", failure_message="Run interrupted; retained partial files are not completed analysis.")
            break
        except Exception as error:
            row.update(status="incomplete", failure_message=safe_provider_error(error))
            if isinstance(getattr(error, "satellite_diagnostics", None), dict):
                row["satellite_diagnostics"] = error.satellite_diagnostics
            LOGGER.error("%s incomplete: %s", hazard, row["failure_message"])
        row["completed_at"] = now(); write_json(output / "run_report.json", report)
    completed = sum(x["status"] == "export_complete_unreviewed" for x in report["events"])
    report.update(status="all_exports_complete_unreviewed" if completed == len(hazards) else "partial_results" if completed else "diagnostics_only", completed_at=now())
    write_json(output / "run_report.json", report)
    return report


def bundle_results(output, destination):
    """Only this fresh run's bounded results; never the Colab runtime or OAuth files."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(output):
        raise ValueError("Choose a fresh archive path outside its result directory")
    files = [p for p in sorted(output.rglob("*")) if p.is_file()]
    if not (output / "run_report.json").is_file() or len(files) > 1000 or any(p.is_symlink() for p in files):
        raise ValueError("Invalid or excessive result file set")
    if sum(p.stat().st_size for p in files) > MAX_BUNDLE_BYTES:
        raise ValueError("Result bundle exceeds 512 MiB; retain diagnostics and process one hazard per run")
    if any(p.suffix.lower() not in (".json", ".tif", ".png") or p.name.startswith(".") or any(s in p.name.lower() for s in ("credential", "token", "service-account")) for p in files):
        raise ValueError("Unexpected non-result file; refusing to archive runtime/source/credentials")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise ValueError("ZIP integrity check failed")
    return {"file": destination.name, "bytes": destination.stat().st_size, "sha256": sha256(destination)}
