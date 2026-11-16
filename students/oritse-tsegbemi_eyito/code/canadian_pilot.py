"""Freeze a Canadian scene before visually reviewing vessels in Colab."""
import hashlib
import json
from pathlib import Path

from PIL import Image

CASE = "saint_john_pilot"
PILOT_BOUNDS = {
    CASE: (-66.091, 45.251, -66.051, 45.279),
    "english_bay_pilot": (-123.230, 49.273, -123.179, 49.304),
}
START, END = "2025-05-01", "2025-10-01"
CONFIDENCE, IOU = 0.55, 0.50


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_scene(project, case=CASE):
    """Save the exact RGB grid, usable-pixel mask and metadata, without inference."""
    from ship_map import fetch_observation
    from ship_backend import write_json

    folder = Path(project) / "data" / "canadian_pilot"
    folder.mkdir(parents=True, exist_ok=True)
    bounds = PILOT_BOUNDS[case]
    manifest_path = folder / f"{case}_scene.json"
    image = folder / f"{case}.png"
    mask = folder / f"{case}_mask.png"
    label = folder / f"{case}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _digest(image) != manifest["image_sha256"] or _digest(mask) != manifest["mask_sha256"]:
            raise ValueError("Frozen image or mask changed.")
        return image, manifest
    if label.exists():
        raise ValueError("An existing annotation requires its original frozen scene.")

    rgb, valid, _, _, metadata = fetch_observation(bounds, START, END)
    Image.fromarray(rgb).save(image)
    Image.fromarray(valid.astype("uint8") * 255).save(mask)
    metadata.update(case=case, image_sha256=_digest(image), mask_sha256=_digest(mask),
                    date_range=[START, END], confidence=CONFIDENCE, match_iou=IOU)
    write_json(label, dict(version="5.6.0", imagePath=image.name, imageData=None,
                           imageWidth=rgb.shape[1], imageHeight=rgb.shape[0], shapes=[],
                           flags={"review_required": True}, annotation_status="UNANNOTATED",
                           image_sha256=metadata["image_sha256"]))
    write_json(manifest_path, metadata)
    return image, metadata


def _iou(a, b):
    intersect = (max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                 max(0, min(a[3], b[3]) - max(a[1], b[1])))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersect
    return intersect / union if union else 0.0


def score_boxes(boxes, scores, targets, iou_threshold=IOU):
    """Match each prediction and label at most once, by descending confidence."""
    if len(boxes) != len(scores):
        raise ValueError("Each prediction needs a confidence score.")
    remaining = set(range(len(targets)))
    matched = []
    for index in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
        overlap, target = max(((_iou(boxes[index], targets[j]), j) for j in remaining),
                              default=(0.0, None))
        if target is not None and overlap >= iou_threshold:
            remaining.remove(target)
            matched.append({"prediction": index, "label": target, "iou": overlap})
    tp, fp, fn = len(matched), len(boxes) - len(matched), len(remaining)
    return dict(true_positives=tp, false_positives=fp, false_negatives=fn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None, matches=matched)


def evaluate_frozen_scene(project, model, device, case=CASE):
    """Score the single-scene pilot against labels reviewed before prediction."""
    import numpy as np
    from canadian_validation import checked_annotation
    from ship_map import detect_raster
    from ship_backend import write_json

    project = Path(project)
    image, scene = freeze_scene(project, case)
    label_path = image.with_suffix(".json")
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if (not checked_annotation(label) or label.get("image_sha256") != scene["image_sha256"]
            or label.get("reviewed_before_predictions") is not True):
        raise ValueError("Review the full frozen image and confirm the labels before running inference.")
    label_sha = _digest(label_path)
    checkpoint = project / "results/models/vessel_detector.pt"
    rgb = np.array(Image.open(image).convert("RGB"))
    valid = np.array(Image.open(image.with_name(f"{case}_mask.png")).convert("L")) > 0
    if (rgb.shape[1], rgb.shape[0]) != (label["imageWidth"], label["imageHeight"]):
        raise ValueError("Review grid differs from the frozen image.")
    targets = [shape["points"][0] + shape["points"][1] for shape in label["shapes"]]
    boxes, scores = detect_raster(model, device, rgb, valid, scene["confidence"])
    boxes, scores = boxes.tolist(), scores.tolist()
    if _digest(label_path) != label_sha:
        raise ValueError("Labels changed while inference was running.")
    report = dict(scene=scene["image_id"], scope="Single-scene visual-reference pilot",
                  image_sha256=scene["image_sha256"], labels_sha256=label_sha,
                  checkpoint_sha256=_digest(checkpoint), score_threshold=scene["confidence"],
                  iou_threshold=scene["match_iou"], reviewer=label["reviewer"],
                  reviewed_utc=label["reviewed_utc"], review_notes=label.get("review_notes", ""),
                  reference_boxes=targets,
                  prediction_boxes=boxes, prediction_scores=scores,
                  **score_boxes(boxes, scores, targets, scene["match_iou"]))
    write_json(project / "results" / f"{case}_metrics.json", report)
    return report
