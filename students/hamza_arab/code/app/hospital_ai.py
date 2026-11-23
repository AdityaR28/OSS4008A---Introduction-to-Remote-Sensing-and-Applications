"""Hospital Infrastructure Canada: reproducible geospatial hospital-building detector.

This module never treats ODHF coordinates, OSM geometries, or unlabeled imagery as
training ground truth. It generates candidate image chips, requires human-reviewed
LabelMe annotations, enforces site-disjoint YOLO splits, trains/evaluates an actual
model, and exports pixel predictions in the existing map's GeoJSON schema.

The main Colab dashboard is NOT imported or modified by this module.
"""
from __future__ import annotations

import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform as transform_coordinates
from pyproj import Transformer
import yaml

# Published Geospatial Ontario imagery service. The relevant DRAPE 2024 imagery
# is described under the Open Government Licence - Ontario. Check coverage and
# service metadata for every downloaded site and preserve source attribution.
ONTARIO_IMAGERY_SERVICE = (
    "https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/"
    "AerialImagery/GEO_Imagery_Data_Service_2023to2027/ImageServer"
)
IMAGERY_LICENSE_URL = "https://data.ontario.ca/dataset/open-ontario-imagery"
COLUMN_NAMES = [
    "tile_id", "site_id", "name", "candidate_class", "latitude", "longitude",
    "image", "geotiff", "annotation", "imagery_source", "imagery_license_url",
    "source_extent_3857", "image_width", "image_height", "ground_width_m",
    "review_status", "reviewed_label", "reviewer", "source_checked",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _site_token(row: pd.Series) -> str:
    text = f"{row.get('id', '')}|{row.get('name', '')}|{float(row.latitude):.6f}|{float(row.longitude):.6f}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = math.pi / 180
    a = math.sin((lat2 - lat1) * r / 2) ** 2 + math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(a)))


def select_review_sites(
    odhf_csv: str | Path,
    center_lat: float = 45.4215,
    center_lon: float = -75.6972,
    radius_km: float = 30.0,
    hospitals: int = 12,
    controls: int = 12,
    min_site_spacing_m: float = 750,
) -> pd.DataFrame:
    """Sample known sites and *unverified* comparison facilities for imagery review.

    `controls` are comparison *candidates*, NOT verified nonhospital buildings.
    Every candidate requires independent visual annotation before training.
    """
    if not (1 <= radius_km <= 150 and 1 <= hospitals <= 120 and 0 <= controls <= 60):
        raise ValueError("Radius must be 1..150 km, hospitals 1..120, controls 0..60")
    df = pd.read_csv(odhf_csv, low_memory=False).fillna("")
    needed = {"name", "latitude", "longitude", "type"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Reference CSV missing columns: {sorted(needed - set(df.columns))}")
    df["latitude"] = pd.to_numeric(df.latitude, errors="coerce")
    df["longitude"] = pd.to_numeric(df.longitude, errors="coerce")
    df = df.loc[df.latitude.between(41, 84) & df.longitude.between(-142, -52)].copy()
    df["distance_km"] = [
        _great_circle_km(center_lat, center_lon, float(r.latitude), float(r.longitude))
        for _, r in df.iterrows()
    ]
    df = df.loc[df.distance_km <= radius_km].sort_values("distance_km").copy()
    df["candidate_class"] = np.where(
        df["type"].astype(str).str.fullmatch("hospital|hospitals", case=False, na=False),
        "known_hospital_site", "other_healthcare_site_UNVERIFIED",
    )
    selected: list[pd.Series] = []
    for category, cap in [("known_hospital_site", hospitals), ("other_healthcare_site_UNVERIFIED", controls)]:
        count = 0
        for _, row in df.loc[df.candidate_class.eq(category)].iterrows():
            if count >= cap:
                break
            if all(_great_circle_km(float(row.latitude), float(row.longitude),
                                    float(other.latitude), float(other.longitude)) * 1000 >= min_site_spacing_m
                   for other in selected):
                selected.append(row)
                count += 1
    if not selected:
        raise RuntimeError("No candidate healthcare sites found near the selected city")
    result = pd.DataFrame(selected).reset_index(drop=True)
    result["site_id"] = result.apply(_site_token, axis=1)
    return result


def prepare_review_chips_from_geotiff(
    odhf_csv: str | Path, source_geotiff: str | Path, project_dir: str | Path,
    imagery_license_url: str, imagery_source_name: str,
    license_confirmed: bool = False,
    center_lat: float = 45.4215, center_lon: float = -75.6972,
    radius_km: float = 30, hospitals: int = 12, controls: int = 12,
    image_size: int = 640, ground_width_m: float = 320,
) -> pd.DataFrame:
    """Offline imagery option: extract georeferenced chips from a licensed local GeoTIFF.

    Raster must have genuine georeferencing and <=1m native GSD at pilot area.
    This never generates automatic labels or training approval.
    """
    from rasterio.warp import reproject, Resampling
    source_geotiff, project_dir = Path(source_geotiff), Path(project_dir)
    if not license_confirmed or not imagery_license_url.startswith('https://') or not imagery_source_name.strip():
        raise ValueError('Source attribution, HTTPS license URL and explicit rights confirmation are required')
    if (project_dir / 'review_manifest.csv').exists():
        raise FileExistsError('Review manifest exists. Do not overwrite reviewed imagery or annotations')
    sites = select_review_sites(odhf_csv, center_lat, center_lon, radius_km, hospitals, controls)
    rows, errors = [], []
    with rasterio.open(source_geotiff) as source:
        if source.crs is None or source.count < 3 or source.dtypes[0] != 'uint8':
            raise ValueError('Source requires >=3 uint8 RGB bands and a real CRS')
        for _, site in sites.iterrows():
            lat, lon = float(site.latitude), float(site.longitude)
            # Do not silently resample 10m satellite imagery and claim roof detail.
            if source.crs.is_geographic:
                gsd_m = max(abs(source.res[0]) * 111320 * math.cos(math.radians(lat)),
                            abs(source.res[1]) * 111132)
            else:
                # Source CRS is projected, in meters for standard Ontario imagery.
                unit = source.crs.linear_units_factor[1] if source.crs.linear_units_factor else 1
                gsd_m = max(abs(source.res[0]), abs(source.res[1])) * unit
            if gsd_m > 1.0:
                raise ValueError(f'Imagery native pixel size {gsd_m:.2f}m is too coarse for hospital rooftops')
            tile=f'local_{site.site_id}_00'
            image_path=project_dir/'review'/'images'/f'{tile}.png'
            tiff_path=project_dir/'review'/'geotiffs'/f'{tile}.tif'
            ann_path=project_dir/'review'/'annotations'/f'{tile}.json'
            bound=_mercator_extent(lat,lon,ground_width_m)
            transform=from_bounds(*bound,width=image_size,height=image_size)
            dest=np.zeros((3,image_size,image_size),dtype=np.uint8)
            try:
                for band in range(1,4):
                    reproject(source=rasterio.band(source,band),destination=dest[band-1],
                              src_transform=source.transform,src_crs=source.crs,
                              dst_transform=transform,dst_crs='EPSG:3857',
                              resampling=Resampling.bilinear)
                if np.mean(np.any(dest>0,axis=0)) < .97 or dest.std()<3:
                    errors.append(f'{tile}: outside source coverage / blank pixels')
                    continue
                image_path.parent.mkdir(parents=True,exist_ok=True)
                tiff_path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(dest.transpose(1,2,0)).save(image_path)
                with rasterio.open(tiff_path,'w',driver='GTiff',width=image_size,height=image_size,
                                   count=3,dtype='uint8',crs='EPSG:3857',transform=transform,
                                   compress='deflate') as dst:dst.write(dest)
                rows.append({
                    'tile_id':tile,'site_id':site.site_id,'name':str(site['name']),
                    'candidate_class':site.candidate_class,'latitude':lat,'longitude':lon,
                    'image':str(image_path.relative_to(project_dir)),
                    'geotiff':str(tiff_path.relative_to(project_dir)),
                    'annotation':str(ann_path.relative_to(project_dir)),
                    'imagery_source':imagery_source_name,
                    'imagery_license_url':imagery_license_url,
                    'source_extent_3857':json.dumps(bound),'image_width':image_size,
                    'image_height':image_size,'ground_width_m':ground_width_m,
                    'review_status':'UNREVIEWED','reviewed_label':'UNKNOWN',
                    'reviewer':'','source_checked':'NO',
                })
            except (OSError,ValueError) as e:
                errors.append(f'{tile}: {str(e)[:240]}')
    project_dir.mkdir(parents=True,exist_ok=True)
    (project_dir/'imagery_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    if not rows:
        raise RuntimeError('No local GeoTIFF candidate chips overlap source. See imagery_errors.json')
    result=pd.DataFrame(rows,columns=COLUMN_NAMES)
    result.to_csv(project_dir/'review_manifest.csv',index=False)
    return result


def _mercator_extent(lat: float, lon: float, ground_width_m: float) -> tuple[float, float, float, float]:
    if not (-85 < lat < 85 and -180 <= lon <= 180):
        raise ValueError("Invalid latitude/longitude")
    if not 60 <= ground_width_m <= 1000:
        raise ValueError("Requested ground width must be 60..1000 meters")
    center_x, center_y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    half = (ground_width_m / max(math.cos(math.radians(lat)), .01)) / 2
    return center_x - half, center_y - half, center_x + half, center_y + half


def download_ontario_chip(
    lat: float, lon: float, image_path: str | Path, geotiff_path: str | Path,
    ground_width_m: float = 320, image_size: int = 640,
    session: Any | None = None, service_url: str = ONTARIO_IMAGERY_SERVICE,
) -> dict[str, Any]:
    """Request one image from official Ontario ArcGIS ImageServer and georeference.

    Uses REST exportImage f=json extent metadata. Fails rather than assuming an
    image's geolocation or silently accepting transparent/no-data pixels.
    """
    if not (256 <= image_size <= 1024):
        raise ValueError("Image size should be 256..1024 pixels")
    if not service_url.startswith("https://") or "/ImageServer" not in service_url:
        raise ValueError("Must use an explicit trusted HTTPS ArcGIS ImageServer URL")
    http = session if session is not None else requests.Session()
    bbox = _mercator_extent(float(lat), float(lon), float(ground_width_m))
    result = http.get(service_url.rstrip("/") + "/exportImage", params={
        "f": "json", "bbox": ",".join(f"{v:.3f}" for v in bbox),
        "bboxSR": 3857, "imageSR": 3857, "size": f"{image_size},{image_size}",
        "format": "png32", "adjustAspectRatio": "false",
        "interpolation": "RSP_BilinearInterpolation",
    }, timeout=80, headers={"User-Agent": "OSS4008-HospitalDetector-Educational/1.0"})
    result.raise_for_status()
    info = result.json()
    if "error" in info:
        raise RuntimeError(f"ArcGIS imagery service error: {info['error']}")
    extent = info.get("extent") or {}
    sr = extent.get("spatialReference") or {}
    if sr.get("wkid", sr.get("latestWkid")) not in (3857, 102100, 102113):
        raise ValueError("Image service did not return the requested EPSG:3857 extent")
    bounds = [extent.get(k) for k in ("xmin", "ymin", "xmax", "ymax")]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds):
        raise ValueError("ArcGIS response has no trustworthy geographic image extent")
    url = info.get("href", "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Image service did not return a safe HTTPS image link")
    # An ArcGIS image export may use a temporary subdomain on the same provider.
    if not (parsed.hostname.endswith("gov.on.ca") or parsed.hostname.endswith("arcgis.com")):
        raise ValueError("Image export link host is outside expected Ontario/ArcGIS domains")
    r = http.get(url, timeout=80, headers={"User-Agent": "OSS4008-HospitalDetector-Educational/1.0"})
    r.raise_for_status()
    if len(r.content) > 14_000_000:
        raise ValueError("Unexpectedly large image export")
    with Image.open(BytesIO(r.content)) as raw:
        if raw.size != (image_size, image_size):
            raise ValueError(f"ImageServer returned {raw.size}, expected {(image_size, image_size)}")
        if "A" in raw.getbands():
            alpha = np.asarray(raw.getchannel("A"))
            if np.mean(alpha > 0) < .97:
                raise ValueError("Imagery is mostly transparent: likely outside service coverage")
        img = raw.convert("RGB")
    rgb = np.asarray(img)
    if rgb.std() < 3:
        raise ValueError("Image export appears blank or nearly uniform")
    image_path, geotiff_path = Path(image_path), Path(geotiff_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    geotiff_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)
    transform = from_bounds(*bounds, width=image_size, height=image_size)
    with rasterio.open(geotiff_path, "w", driver="GTiff", width=image_size, height=image_size,
                       count=3, dtype="uint8", crs="EPSG:3857", transform=transform,
                       compress="deflate") as dest:
        dest.write(rgb.transpose(2, 0, 1))
    return {"image": str(image_path), "geotiff": str(geotiff_path),
            "extent": bounds, "size": image_size, "ground_width_m": ground_width_m,
            "service": service_url, "license_url": IMAGERY_LICENSE_URL}


def prepare_review_chips(
    odhf_csv: str | Path, project_dir: str | Path, center_lat: float = 45.4215,
    center_lon: float = -75.6972, radius_km: float = 30,
    hospitals: int = 12, controls: int = 12, image_size: int = 640,
    ground_width_m: float = 320, session: Any | None = None,
) -> pd.DataFrame:
    """Acquire chips & manifest. Does NOT create approved annotations."""
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    sites = select_review_sites(odhf_csv, center_lat, center_lon, radius_km, hospitals, controls)
    # RESUME SAFE: never overwrite existing human review or annotation files.
    manifest_path = project_dir / "review_manifest.csv"
    rows: list[dict[str, Any]] = []
    if manifest_path.is_file():
        existing = pd.read_csv(manifest_path, keep_default_na=False)
        if list(existing.columns) != COLUMN_NAMES:
            raise ValueError("Existing review manifest schema differs; refusing overwrite")
        rows = existing.to_dict('records')
    existing_by_site = {str(r['site_id']): j for j, r in enumerate(rows)}
    errors: list[str] = []
    for i, s in sites.iterrows():
        prior_index = existing_by_site.get(str(s.site_id))
        if prior_index is not None:
            prior = rows[prior_index]
            # Approved/rejected human decisions are immutable during download repair.
            if str(prior.get('review_status')) in ('APPROVED','REJECTED'):
                continue
            existing_image = project_dir / str(prior.get('image',''))
            existing_tiff = project_dir / str(prior.get('geotiff',''))
            if existing_image.is_file() and existing_tiff.is_file():
                continue
            # Only an unreviewed missing-file entry can be repaired in place.
        tile = f"ontario_{s.site_id}_00"
        image = project_dir / "review" / "images" / f"{tile}.png"
        tiff = project_dir / "review" / "geotiffs" / f"{tile}.tif"
        annotation = project_dir / "review" / "annotations" / f"{tile}.json"
        try:
            info = download_ontario_chip(float(s.latitude), float(s.longitude), image, tiff,
                                         ground_width_m, image_size, session=session)
        except (requests.RequestException, RuntimeError, ValueError, OSError) as error:
            errors.append(f"{tile}: {str(error)[:220]}")
            continue
        # Deliberately no blank LabelMe JSON auto-created. An empty annotation
        # could otherwise be confused with a human-verified negative example.
        candidate={
            "tile_id": tile, "site_id": s.site_id, "name": str(s["name"]),
            "candidate_class": s.candidate_class, "latitude": s.latitude, "longitude": s.longitude,
            "image": str(image.relative_to(project_dir)), "geotiff": str(tiff.relative_to(project_dir)),
            "annotation": str(annotation.relative_to(project_dir)),
            "imagery_source": info["service"], "imagery_license_url": info["license_url"],
            "source_extent_3857": json.dumps(info["extent"]),
            "image_width": image_size, "image_height": image_size, "ground_width_m": ground_width_m,
            "review_status": "UNREVIEWED", "reviewed_label": "UNKNOWN", "reviewer": "", "source_checked": "NO",
        }
        if prior_index is None:
            rows.append(candidate)
            existing_by_site[str(s.site_id)] = len(rows)-1
        else:
            rows[prior_index] = candidate  # previously UNREVIEWED, no human annotation lost
        # Save each completed chip. An interrupted 100-site acquisition must not
        # discard the 99 previously successful downloads or their metadata.
        checkpoint = pd.DataFrame(rows, columns=COLUMN_NAMES)
        checkpoint_tmp = manifest_path.with_suffix('.csv.checkpoint')
        checkpoint.to_csv(checkpoint_tmp, index=False)
        checkpoint_tmp.replace(manifest_path)
        if len(rows)%10 == 0:
            print(f"Saved {len(rows)} aerial chips in Drive (includes earlier images); current site {tile}")
    (project_dir / "imagery_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    if not rows:
        raise RuntimeError("No imagery downloaded; inspect imagery_errors.json for provider/coverage details")
    manifest = pd.DataFrame(rows, columns=COLUMN_NAMES)
    temp = manifest_path.with_suffix('.csv.pending')
    manifest.to_csv(temp, index=False)
    temp.replace(manifest_path)
    return manifest


def labelme_rectangle_to_yolo(points: Any, width: int, height: int) -> tuple[float, float, float, float]:
    if len(points) != 2 or width <= 0 or height <= 0:
        raise ValueError("LabelMe hospital rectangle must have two corners and a valid image size")
    (x1, y1), (x2, y2) = points
    x1, x2 = sorted((max(0, min(width, float(x1))), max(0, min(width, float(x2)))))
    y1, y2 = sorted((max(0, min(height, float(y1))), max(0, min(height, float(y2)))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Degenerate bounding box")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def _verified_annotation(annotation: Path, image: Path, verified_label: str) -> list[str]:
    if not annotation.is_file():
        raise ValueError(f"Missing independently reviewed LabelMe annotation: {annotation}")
    doc = json.loads(annotation.read_text(encoding="utf-8"))
    with Image.open(image) as img:
        width, height = img.size
    if int(doc.get("imageWidth", -1)) != width or int(doc.get("imageHeight", -1)) != height:
        raise ValueError(f"Annotation image dimensions mismatch for {image.name}")
    if doc.get("flags", {}).get("verified") is not True:
        raise ValueError(f"LabelMe file is missing explicit verified=true flag: {annotation}")
    boxes = []
    for shape in doc.get("shapes", []):
        if str(shape.get("label", "")).casefold() != "hospital":
            raise ValueError(f"Unsupported or ambiguous LabelMe shape label: {shape.get('label')}")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"Expected rectangle annotation: {annotation.name}")
        yolo = labelme_rectangle_to_yolo(shape.get("points", []), width, height)
        boxes.append("0 " + " ".join(f"{v:.7f}" for v in yolo))
    if verified_label == "HOSPITAL_PRESENT" and not boxes:
        raise ValueError("Positive, approved image must have at least one verified hospital box")
    if verified_label == "NO_HOSPITAL" and boxes:
        raise ValueError("Reviewed negative image cannot contain a hospital box")
    if verified_label not in ("HOSPITAL_PRESENT", "NO_HOSPITAL"):
        raise ValueError("reviewed_label must explicitly identify verified positive or negative")
    return boxes


def split_verified_dataset(
    project_dir: str | Path, seed: int = 4008, min_sites_per_class: int = 3,
) -> dict[str, Any]:
    """Build a YOLO dataset with no site leakage and independently reviewed labels.

    Requires at least three separately reviewed positive *sites* and three
    negative *sites*, one per class in each train/val/test split. Three sites
    per class is a mechanical minimum; meaningful accuracy requires many more.
    """
    project_dir = Path(project_dir)
    manifest = project_dir / "review_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing candidate review manifest: {manifest}")
    df = pd.read_csv(manifest, keep_default_na=False)
    missing = [k for k in COLUMN_NAMES if k not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    # Pending/auto-draft-only images are NOT training labels. Excluding them
    # avoids forcing the reviewer to clear 100 candidates before training on
    # a small, independently approved pilot subset. Never silently promote.
    if len(df) == 0 or not df.review_status.isin(["APPROVED", "REJECTED", "UNREVIEWED"]).all():
        raise ValueError("Invalid review status; only APPROVED images are eligible for training")
    if df.tile_id.duplicated().any():
        raise ValueError("Duplicate tile_id present")
    df = df.loc[df.review_status.eq("APPROVED")].copy()
    if df.empty or not df.source_checked.eq("YES").all() or df.reviewer.astype(str).str.strip().eq("").any():
        raise ValueError("At least one APPROVED image and named human reviewer/source check required")
    prepared = []
    for _, row in df.iterrows():
        img = (project_dir / row.image).resolve()
        ann = (project_dir / row.annotation).resolve()
        if not img.is_relative_to(project_dir.resolve()) or not ann.is_relative_to(project_dir.resolve()):
            raise ValueError("Manifest contains an unsafe path outside the project folder")
        if not img.is_file():
            raise FileNotFoundError(f"Missing reviewed image: {img}")
        boxes = _verified_annotation(ann, img, str(row.reviewed_label))
        prepared.append({"tile_id": row.tile_id, "site_id": row.site_id,
                         "category": row.reviewed_label, "image": img,
                         "boxes": boxes, "reviewer": row.reviewer})
    # Group by independent hospital/site; not by neighboring image crops.
    grouped: dict[str, list[dict]] = {}
    for item in prepared:
        grouped.setdefault(item["site_id"], []).append(item)
    classes = {"HOSPITAL_PRESENT": [], "NO_HOSPITAL": []}
    for sid, items in grouped.items():
        cats = {x["category"] for x in items}
        # When one campus has mixed positive/negative patches, keep the entire
        # campus together and count it as a positive group.
        primary = "HOSPITAL_PRESENT" if "HOSPITAL_PRESENT" in cats else "NO_HOSPITAL"
        classes[primary].append(sid)
    if any(len(v) < max(3, min_sites_per_class) for v in classes.values()):
        raise ValueError("Insufficient independent APPROVED positive/negative sites for site-disjoint train/val/test. "
                         f"Found: { {k: len(v) for k, v in classes.items()} }")
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for group_names in classes.values():
        group_names = sorted(group_names)
        rng.shuffle(group_names)
        # Always reserve an independent positive/negative group for val and test.
        test_id, val_id = group_names[0], group_names[1]
        assignment[test_id] = "test"
        assignment[val_id] = "val"
        for sid in group_names[2:]:
            assignment[sid] = "train"
    output = project_dir / "hospital_yolo"
    if output.exists():
        # Prevent silent reuse of stale images/labels when dataset changes.
        shutil.rmtree(output)
    counts = {"train": {"positive": 0, "negative": 0},
              "val": {"positive": 0, "negative": 0},
              "test": {"positive": 0, "negative": 0}}
    assignment_rows = []
    for item in prepared:
        split = assignment[item["site_id"]]
        dest_image = output / "images" / split / item["image"].name
        dest_label = output / "labels" / split / (item["image"].stem + ".txt")
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["image"], dest_image)
        dest_label.write_text("\n".join(item["boxes"]) + ("\n" if item["boxes"] else ""), encoding="utf-8")
        counts[split]["positive" if item["boxes"] else "negative"] += 1
        assignment_rows.append({"tile_id": item["tile_id"], "site_id": item["site_id"],
                                "split": split, "reviewed_label": item["category"],
                                "image_sha256": sha256(item["image"])})
    if any(value <= 0 for v in counts.values() for value in v.values()):
        raise ValueError("Need verified positive and negative image examples in each train/val/test split")
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val",
              "test": "images/test", "names": {0: "hospital"}}
    yaml_path = output / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pd.DataFrame(assignment_rows).to_csv(output / "site_split_manifest.csv", index=False)
    sources = sorted({str(v) for v in df.imagery_source if str(v).strip()})
    licenses = sorted({str(v) for v in df.imagery_license_url if str(v).strip()})
    if len(sources) == 0 or len(licenses) == 0:
        raise ValueError("Reviewed imagery must include recorded sources and licence URLs")
    provenance = {"status": "ANNOTATIONS_REVIEWED", "label_type": "LabelMe rectangles",
                  "image_sources": sources, "license_urls": licenses,
                  "source_manifest_sha256": sha256(manifest),
                  "split_manifest_sha256": sha256(output / "site_split_manifest.csv"),
                  "site_disjoint": True, "seed": seed, "counts": counts,
                  "notes": "Site disjoint does not guarantee cross-city generalization; inspect data distribution."}
    (output / "dataset_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"dataset_yaml": str(yaml_path), "counts": counts,
            "num_sites": len(grouped), "site_disjoint": True}


def train_and_evaluate(
    project_dir: str | Path, epochs: int = 40, image_size: int = 640,
    batch: int = 8, pretrained_model: str = "yolo11n.pt",
) -> dict[str, Any]:
    """Actually fine-tune YOLO on hospital labels, evaluate held-out sites.

    This is a real computational training step, not executed unless approved
    imagery/labels were provided, split_verified_dataset succeeded, and the
    Ultralytics package/model weights can be downloaded in a Colab runtime.
    """
    project_dir = Path(project_dir)
    data = project_dir / "hospital_yolo" / "data.yaml"
    provenance = project_dir / "hospital_yolo" / "dataset_provenance.json"
    if not data.is_file() or not provenance.is_file():
        raise RuntimeError("BLOCKED: Run split_verified_dataset() on real reviewed annotations first")
    info = json.loads(provenance.read_text(encoding="utf-8"))
    if info.get("status") != "ANNOTATIONS_REVIEWED" or info.get("site_disjoint") is not True:
        raise RuntimeError("Dataset has no verified site-disjoint training provenance")
    current_split = project_dir / "hospital_yolo" / "site_split_manifest.csv"
    if (not current_split.is_file() or
        info.get("split_manifest_sha256") != sha256(current_split) or
        info.get("source_manifest_sha256") != sha256(project_dir / "review_manifest.csv")):
        raise RuntimeError("The reviewed dataset changed after conversion; rebuild the site-disjoint dataset")
    if not 1 <= epochs <= 250 or not 256 <= image_size <= 1280 or not 1 <= batch <= 64:
        raise ValueError("Invalid training hyperparameters")
    from ultralytics import YOLO  # defer heavy import until real training
    import torch
    model = YOLO(pretrained_model)
    device = 0 if torch.cuda.is_available() else "cpu"
    fit = model.train(data=str(data), epochs=epochs, imgsz=image_size, batch=batch,
                      seed=4008, deterministic=True, patience=12,
                      device=device, workers=2, project=str(project_dir / "runs"),
                      name="hospital_detector", exist_ok=True)
    weights = Path(fit.save_dir) / "weights" / "best.pt"
    if not weights.is_file():
        raise RuntimeError("Training ended without a best.pt checkpoint")
    detector_dir = project_dir / "detector"
    detector_dir.mkdir(parents=True, exist_ok=True)
    model_path = detector_dir / "hospital_best.pt"
    shutil.copy2(weights, model_path)
    evaluator = YOLO(str(model_path))
    metrics = evaluator.val(data=str(data), split="test", imgsz=image_size, device=device,
                            project=str(project_dir / "runs"), name="heldout_test", exist_ok=True)
    values = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}
    report = {"status": "TRAINED_AND_EVALUATED", "model": str(model_path),
              "weights_sha256": sha256(model_path), "dataset_yaml": str(data),
              "image_size": image_size, "batch": batch, "epochs_requested": epochs,
              "device": str(device), "pretrained_model": pretrained_model,
              "test_metrics": values, "data_provenance": str(provenance),
              "source_manifest_sha256": info["source_manifest_sha256"],
              "split_manifest_sha256": info["split_manifest_sha256"],
              "limit": "Hospital rooftop appearance alone cannot establish building function; generalization requires independent sites/cities."}
    (detector_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _nms(boxes: list[dict[str, Any]], iou_limit: float = 0.5) -> list[dict[str, Any]]:
    """Greedy cross-tile NMS for axis-aligned boxes, highest model score first."""
    def iou(a, b):
        x1, y1 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
        x2, y2 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
        intersection = max(0, x2-x1) * max(0, y2-y1)
        aa = (a["box"][2]-a["box"][0]) * (a["box"][3]-a["box"][1])
        ab = (b["box"][2]-b["box"][0]) * (b["box"][3]-b["box"][1])
        return intersection / (aa + ab - intersection) if aa + ab > intersection else 0.0
    chosen = []
    for candidate in sorted(boxes, key=lambda x: -x["confidence"]):
        if all(iou(candidate, other) < iou_limit for other in chosen):
            chosen.append(candidate)
    return chosen


def _tile_starts(length: int, size: int, overlap: float):
    if length <= size:
        return [0]
    stride = max(1, int(size * (1 - overlap)))
    return sorted(set(list(range(0, length - size + 1, stride)) + [length - size]))


def export_geotiff_predictions(
    project_dir: str | Path, geotiff: str | Path, output_geojson: str | Path,
    confidence: float = .35, tile_size: int = 640, overlap: float = .2,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Export *true trained* hospital model boxes as WGS84 points for existing UI.

    `predictor` is injectable solely for offline coordinate-contract tests; real
    inference requires matching trained checkpoint and training_report.json.
    """
    project_dir, geotiff, output_geojson = Path(project_dir), Path(geotiff), Path(output_geojson)
    report_file = project_dir / "detector" / "training_report.json"
    if predictor is None:
        if not report_file.is_file():
            raise RuntimeError("BLOCKED: no evaluated hospital model report. Train and evaluate first.")
        report = json.loads(report_file.read_text(encoding="utf-8"))
        model_file = project_dir / "detector" / "hospital_best.pt"
        if (report.get("status") != "TRAINED_AND_EVALUATED" or not model_file.is_file() or
            report.get("weights_sha256") != sha256(model_file) or
            not (project_dir / "hospital_yolo" / "site_split_manifest.csv").is_file() or
            report.get("split_manifest_sha256") != sha256(project_dir / "hospital_yolo" / "site_split_manifest.csv") or
            report.get("source_manifest_sha256") != sha256(project_dir / "review_manifest.csv")):
            raise RuntimeError("Missing, stale or mismatched trained hospital detector checkpoint / dataset")
        from ultralytics import YOLO
        predictor = YOLO(str(model_file))
        if "hospital" not in [str(n).casefold() for n in predictor.names.values()]:
            raise RuntimeError("Model weights are not for the hospital class")
        model_id = report["weights_sha256"]
    else:
        # Injected predictors are not AI evidence and their exports cannot be
        # passed off as real predictions.
        model_id = "TEST_PREDICTOR_DO_NOT_PUBLISH"
    if not 0 < confidence < 1 or not 256 <= tile_size <= 1280 or not 0 <= overlap < .5:
        raise ValueError("Invalid prediction tiling/confidence configuration")
    with rasterio.open(geotiff) as src:
        if src.crs is None or src.count < 3 or src.dtypes[0] != "uint8":
            raise ValueError("Need georeferenced 3-band uint8 GeoTIFF for inference")
        if src.width > 30000 or src.height > 30000:
            raise ValueError("Input too large for in-memory inference; split into separate geotiffs")
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform = src.transform
        crs = src.crs
    boxes = []
    for y in _tile_starts(rgb.shape[0], tile_size, overlap):
        for x in _tile_starts(rgb.shape[1], tile_size, overlap):
            tile = rgb[y:y+tile_size, x:x+tile_size]
            results = predictor.predict(source=Image.fromarray(tile), conf=confidence, verbose=False)
            for result in results:
                for b in result.boxes:
                    if int(b.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                    x1, x2 = max(0, x+x1), min(rgb.shape[1], x+x2)
                    y1, y2 = max(0, y+y1), min(rgb.shape[0], y+y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append({"box": (x1, y1, x2, y2), "confidence": float(b.conf[0])})
    boxes = _nms(boxes)
    features = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        map_x, map_y = transform * ((x1 + x2) / 2, (y1 + y2) / 2)
        lon, lat = transform_coordinates(crs, "EPSG:4326", [map_x], [map_y])
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon[0], lat[0]]},
                         "properties": {"class": "hospital", "confidence": b["confidence"],
                                        "bbox_pixels": [x1, y1, x2, y2],
                                        "method": "DETECTOR_TEST_ONLY" if model_id.startswith("TEST_") else "trained_YOLO_hospital_building_detector",
                                        "weights_sha256": model_id, "source_geotiff": geotiff.name}})
    collection = {"type": "FeatureCollection", "features": features}
    if model_id.startswith("TEST_"):
        # Only return test data to caller, never save a fake model export.
        return collection
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    output_geojson.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    return collection


def make_candidate_contact_sheet(project_dir: str | Path, width: int = 900) -> Path:
    """Create a review sheet for human labeling; not an AI result."""
    root = Path(project_dir)
    df = pd.read_csv(root / "review_manifest.csv", keep_default_na=False)
    thumbs = []
    for _, r in df.iterrows():
        img_path = root / r.image
        with Image.open(img_path) as src:
            im = ImageOps.fit(src.convert("RGB"), (width // 3, width // 3))
            thumbs.append((im.copy(), f"{r.tile_id[:22]}\n{r.candidate_class}"))
    w = width
    h = (width // 3 + 54) * math.ceil(len(thumbs) / 3)
    sheet = Image.new("RGB", (w, max(h, 1)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(thumbs):
        x = (i % 3) * (width // 3)
        y = (i // 3) * (width // 3 + 54)
        sheet.paste(image, (x, y))
        draw.multiline_text((x + 5, y + width // 3 + 5), label, fill="black", spacing=3)
    target = root / "review_contact_sheet.jpg"
    sheet.save(target, quality=90)
    return target


def review_workspace_status(project_dir: str | Path) -> dict[str, Any]:
    """Read-only live inspection; never report imagery as downloaded without files."""
    project_dir = Path(project_dir)
    result: dict[str, Any] = {"manifest_found": False, "records": 0,
        "images_found": 0, "geotiffs_found": 0, "annotations_found": 0,
        "review_status": {}, "missing_files": [], "download_errors": []}
    errors = project_dir / "imagery_errors.json"
    if errors.is_file():
        try:
            result["download_errors"] = json.loads(errors.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            result["download_errors"] = ["Imagery error log could not be parsed"]
    manifest = project_dir / "review_manifest.csv"
    if not manifest.is_file():
        return result
    df = pd.read_csv(manifest, keep_default_na=False)
    result["manifest_found"] = True
    result["records"] = len(df)
    if "review_status" in df:
        result["review_status"] = {str(k): int(v) for k, v in df.review_status.value_counts().items()}
    for field, key in (("image", "images_found"), ("geotiff", "geotiffs_found"),
                       ("annotation", "annotations_found")):
        if field not in df:
            result["missing_files"].append(f"Missing column {field}")
            continue
        for loc, value in enumerate(df[field]):
            path = (project_dir / str(value)).resolve()
            if path.is_relative_to(project_dir.resolve()) and path.is_file():
                result[key] += 1
            else:
                # Unreviewed/rejected images SHOULD NOT have annotation JSON.
                # Only missing imagery or missing approved annotations are blockers.
                row_status = str(df.iloc[loc]["review_status"]) if "review_status" in df.columns else ""
                if field == "annotation" and row_status in ("UNREVIEWED", "REJECTED"):
                    continue
                if len(result["missing_files"]) < 12:
                    result["missing_files"].append(str(value))
    return result


def evaluated_model_ready(project_dir: str | Path) -> tuple[bool, str]:
    """Check an actual checkpoint against an evaluated, matching reviewed dataset."""
    root = Path(project_dir)
    report_path = root / "detector" / "training_report.json"
    model_path = root / "detector" / "hospital_best.pt"
    if not report_path.is_file() or not model_path.is_file():
        return False, "No evaluated hospital detector checkpoint exists"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        splits = root / "hospital_yolo" / "site_split_manifest.csv"
        original = root / "review_manifest.csv"
        if (report.get("status") != "TRAINED_AND_EVALUATED" or
            report.get("weights_sha256") != sha256(model_path) or
            not splits.is_file() or report.get("split_manifest_sha256") != sha256(splits) or
            not original.is_file() or report.get("source_manifest_sha256") != sha256(original)):
            return False, "Model report or reviewed dataset does not match checkpoint"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"Model provenance could not be verified: {exc}"
    return True, "Evaluated model checkpoint and dataset hashes verified"


def export_all_heldout_predictions(project_dir: str | Path, output_geojson: str | Path,
                                    confidence: float = .35, max_sites: int = 40) -> dict[str, Any]:
    """Run real model inference on all independent test sites and atomically save map GeoJSON.

    Never substitutes reference coordinates for detector predictions. Each feature
    inherits the SHA-256 fingerprint of the evaluated model weights.
    """
    project_dir = Path(project_dir)
    ready, reason = evaluated_model_ready(project_dir)
    if not ready:
        raise RuntimeError("Cannot export actual detections: " + reason)
    split_file = project_dir / "hospital_yolo" / "site_split_manifest.csv"
    splits = pd.read_csv(split_file, keep_default_na=False)
    if not {"tile_id", "site_id", "split"}.issubset(splits):
        raise ValueError("Invalid site split manifest")
    selected = splits.loc[splits.split.eq("test")]
    if selected.empty or selected.site_id.nunique() > max_sites:
        raise ValueError("No valid held-out test sites, or too many sites")
    review = pd.read_csv(project_dir / "review_manifest.csv", keep_default_na=False)
    if not {"tile_id", "geotiff", "site_id"}.issubset(review):
        raise ValueError("Review manifest lacks georeferenced imagery paths")
    combined = selected.merge(review[["tile_id", "geotiff", "site_id"]], on=["tile_id", "site_id"], validate="one_to_one")
    if len(combined) != len(selected):
        raise ValueError("Missing test imagery: refused partial map export")
    output_geojson = Path(output_geojson)
    features = []
    for _, row in combined.iterrows():
        tif = (project_dir / row.geotiff).resolve()
        if not tif.is_relative_to(project_dir.resolve()) or not tif.is_file():
            raise FileNotFoundError(f"Held-out source GeoTIFF unavailable: {row.geotiff}")
        single = export_geotiff_predictions(project_dir, tif, output_geojson.with_suffix('.temporary.geojson'), confidence=confidence)
        for f in single["features"]:
            f["properties"]["test_site_id"] = str(row.site_id)
            features.append(f)
    if len(features) > 3000:
        raise ValueError("Too many detections for the existing map uploader")
    geojson = {"type": "FeatureCollection", "features": features}
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_geojson.with_suffix('.incomplete')
    tmp.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    tmp.replace(output_geojson)
    interim = output_geojson.with_suffix('.temporary.geojson')
    if interim.is_file():
        interim.unlink()
    return {"status": "EXPORTED_EVALUATED_MODEL_OUTPUT", "test_sites": int(selected.site_id.nunique()),
            "features": len(features), "geojson": str(output_geojson)}


def infer_in_selected_circle(
    project_dir: str | Path, latitude: float, longitude: float, radius_m: float,
    confidence: float = .35, max_images: int = 16,
    session: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the actual trained detector inside a small drawn Ontario study circle.

    Backend for the already existing Colab map assistant. No new map widget or
    control is needed. A ≤250 m radius prevents attempting city-wide imagery
    acquisition in a free Colab environment. Returns only real model outputs;
    untrained reference records are never considered detections.
    """
    from tempfile import TemporaryDirectory
    project_dir=Path(project_dir)
    good, reason=evaluated_model_ready(project_dir)
    if not good:
        raise RuntimeError('Hospital inference unavailable: '+reason)
    latitude,longitude,radius_m=float(latitude),float(longitude),float(radius_m)
    if not (math.isfinite(latitude) and math.isfinite(longitude) and math.isfinite(radius_m)):
        raise ValueError('Invalid study circle coordinates')
    if not (41.7 <= latitude <= 57 and -95.2 <= longitude <= -73.8):
        raise ValueError('The current rights-verified imagery workflow covers Ontario pilot areas only')
    if not (20 <= radius_m <= 250):
        raise ValueError('Draw a study circle with radius 20–250 metres for AI inference')
    if not (0 < confidence < 1 and 1 <= max_images <= 16):
        raise ValueError('Invalid inference configuration')
    converter = Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True)
    inverse=Transformer.from_crs('EPSG:3857','EPSG:4326',always_xy=True)
    center_x,center_y=converter.transform(longitude,latitude)
    step_m=250.0
    xy_centres=[(0.0,0.0)] if radius_m <= 130 else [
        (dx*step_m,dy*step_m)
        for dx in (-1,0,1) for dy in (-1,0,1)
        if math.hypot(dx*step_m,dy*step_m) <= radius_m + 160
    ]
    if len(xy_centres)>max_images:
        raise RuntimeError('Study circle exceeds capped aerial imagery requests')
    features=[]
    errors=[]
    with TemporaryDirectory(prefix='oss4008_area_') as temp:
        temp_path=Path(temp)
        for idx,(dx,dy) in enumerate(xy_centres):
            # EPSG:3857 coordinates are stretched by 1/cos(latitude), so scale
            # physically measured offsets when moving the tile centres.
            lon,lat=inverse.transform(center_x+dx/max(math.cos(math.radians(latitude)),.01),
                                      center_y+dy/max(math.cos(math.radians(latitude)),.01))
            image=temp_path/f'chip_{idx:02d}.png'
            tif=temp_path/f'chip_{idx:02d}.tif'
            try:
                download_ontario_chip(lat,lon,image,tif,session=session)
                geo=export_geotiff_predictions(project_dir,tif,temp_path/f'partial_{idx:02d}.geojson',confidence=confidence)
                for feature in geo['features']:
                    f_lat=feature['geometry']['coordinates'][1]
                    f_lon=feature['geometry']['coordinates'][0]
                    if _great_circle_km(latitude,longitude,f_lat,f_lon)*1000 <= radius_m:
                        features.append(feature)
            except Exception as exc:
                errors.append(f'Tile {idx}: {type(exc).__name__}: {str(exc)[:170]}')
    if errors:
        raise RuntimeError(f'Incomplete imagery/inference for study circle ({len(errors)} of {len(xy_centres)} tiles): '+ '; '.join(errors[:2]))
    # Nearby predicted centres often duplicate across overlapping orthophoto
    # chips. Keep the strongest score within 12 metres; NOT ODHF data.
    unique=[]
    for feature in sorted(features,key=lambda f:-float(f['properties']['confidence'])):
        lon,lat=feature['geometry']['coordinates']
        if all(_great_circle_km(lat,lon,u['geometry']['coordinates'][1],u['geometry']['coordinates'][0])*1000>12
               for u in unique):
            unique.append(feature)
    if len(unique)>3000:
        raise RuntimeError('Excessive inference features; inspect detector quality')
    report=json.loads((project_dir/'detector'/'training_report.json').read_text(encoding='utf-8'))
    result={'type':'FeatureCollection','features':unique}
    output_dir=project_dir/'predictions'
    output_dir.mkdir(parents=True,exist_ok=True)
    out=output_dir/f'circle_{latitude:.4f}_{longitude:.4f}_{round(radius_m)}m.geojson'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return {'ok':True,'features':unique,'count':len(unique),
            'image_chips':len(xy_centres),'model_sha256':report['weights_sha256'],
            'saved_to':str(out), 'note':'Screening candidates, not verified hospital identity'}
