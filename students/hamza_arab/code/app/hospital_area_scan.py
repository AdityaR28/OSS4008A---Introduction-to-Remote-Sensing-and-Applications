"""Bounded, reference-guided hospital-roof candidate detection for Colab map.

Actual image-model YOLO detections are returned as georeferenced rectangular
POLYGONS. Scans only around official known hospital reference sites, NOT every
building in the selected city. Weak training does NOT verify hospital use.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer

SCAN_VERSION = 'reference_roof_boxes_v2_size_screened'
MAX_BATCH = 4


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for buf in iter(lambda: stream.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def great_circle_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    x = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return 2*6371008.8*math.asin(min(1, math.sqrt(x)))


def bbox_polygon(ds, box):
    """Pixel xyxy (Ultralytics) -> WGS84 GeoJSON Polygon (bounding rectangle)."""
    if ds.crs is None or len(box) != 4:
        raise ValueError('Raster CRS or pixel box missing')
    x1,y1,x2,y2=[float(t) for t in box]
    if not all(map(math.isfinite,(x1,y1,x2,y2))) or not (0<=x1<x2<=ds.width and 0<=y1<y2<=ds.height):
        raise ValueError('Invalid pixel bounding box')
    to_wgs=Transformer.from_crs(ds.crs,'EPSG:4326',always_xy=True)
    corners=[]
    for col,row in [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]:
        px,py=ds.transform*(col,row)
        lon,lat=to_wgs.transform(px,py)
        if not (-95.3<=lon<=-73.5 and 41.5<=lat<=57.1):
            raise ValueError('Bounding rectangle is outside Ontario pilot')
        corners.append([round(lon,8),round(lat,8)])
    return {'type':'Polygon','coordinates':[corners]}


# Conservative geometry screen for a roof *candidate*, not proof of hospital use.
# These thresholds target the many detached-house false positives visible in the pilot.
MIN_ROOF_WIDTH_M = 16.0
MIN_ROOF_BBOX_AREA_M2 = 750.0
MAX_ROOF_BBOX_ASPECT_RATIO = 6.0


def roof_box_metrics(geometry):
    """Measure a georeferenced bounding rectangle in metres (not building height)."""
    ring=geometry['coordinates'][0]
    if len(ring)<4:raise ValueError('Invalid rectangle polygon')
    # lon/lat coords; adjacent sides of a georeferenced image bbox.
    a,b,c=ring[0],ring[1],ring[2]
    width=great_circle_m(a[1],a[0],b[1],b[0])
    length=great_circle_m(b[1],b[0],c[1],c[0])
    area=width*length
    aspect=max(width,length)/max(.001,min(width,length))
    return {'width_m':round(width,1),'length_m':round(length,1),
            'bbox_area_m2':round(area,0),'bbox_aspect_ratio':round(aspect,2)}


def passes_roof_screen(metrics):
    return (min(metrics['width_m'],metrics['length_m'])>=MIN_ROOF_WIDTH_M
            and metrics['bbox_area_m2']>=MIN_ROOF_BBOX_AREA_M2
            and metrics['bbox_aspect_ratio']<=MAX_ROOF_BBOX_ASPECT_RATIO)


def polygon_center(feature):
    ring=feature['geometry']['coordinates'][0]
    return ((ring[0][1]+ring[2][1])/2,(ring[0][0]+ring[2][0])/2)


def _load_model_source(path, module_name):
    spec=importlib.util.spec_from_file_location(module_name,str(path))
    if spec is None or spec.loader is None:raise ImportError('Missing saved inference source')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def _valid_weak_checkpoint(root):
    exp=root/'weak_osm_experiment'
    report=exp/'weak_training_report.json'
    weights=exp/'weak_best.pt'
    split=exp/'dataset'/'weak_site_split_manifest.csv'
    if not all(f.is_file() for f in (report,weights,split)):
        raise RuntimeError('No experimental YOLO model saved; train it in Notebook 08 first')
    data=json.loads(report.read_text(encoding='utf-8'))
    if (data.get('status')!='TRAINED_ON_OSM_PSEUDO_LABELS_NOT_VERIFIED'
        or data.get('weights_sha256')!=sha256(weights)
        or data.get('split_sha256')!=sha256(split)):
        raise RuntimeError('Experimental model or split fingerprint mismatch; inference blocked')
    return weights, data['weights_sha256']


def _read_reuse_image(root, lat, lon):
    """Only reuse real previously downloaded georeferenced images for this site."""
    manifest=root/'review_manifest.csv'
    if not manifest.is_file():return None
    import pandas as pd
    try:
        df=pd.read_csv(manifest,keep_default_na=False,low_memory=False)
        if not {'latitude','longitude','geotiff'}.issubset(df.columns):return None
        choices=[]
        for _,r in df.iterrows():
            try:
                dist=great_circle_m(lat,lon,float(r['latitude']),float(r['longitude']))
                p=(root/str(r['geotiff'])).resolve()
                if dist<=12 and p.is_relative_to(root) and p.is_file():choices.append((dist,p))
            except (ValueError,TypeError):continue
        if choices:
            choices.sort(key=lambda e:e[0])
            return choices[0][1]
    except (OSError,ValueError):pass
    return None


def _read_valid_cache(path, *, weights_sha, site_lat, site_lon, confidence):
    if not path.is_file():return None
    try:
        x=json.loads(path.read_text(encoding='utf-8'))
        if (x.get('version')!=SCAN_VERSION or x.get('weights_sha256')!=weights_sha
            or x.get('site')!=[site_lat,site_lon] or x.get('confidence')!=confidence
            or not isinstance(x.get('features'),list) or len(x['features'])>80):return None
        for feat in x['features']:
            if feat.get('geometry',{}).get('type')!='Polygon' or feat.get('properties',{}).get('method')!='weak_yolo_geo_bbox':return None
        return x['features']
    except (OSError,ValueError,TypeError):return None


def _save_cache(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def scan_sites(root, sites, *, confidence=.45, model=None,
               downloader: Callable | None=None, max_distance_m=195):
    """Run actual weak YOLO on up to four known-reference hospital site chips.

    Sites are prevalidated by callback against Statistics Canada hospital points.
    This backend itself validates coordinates, limits, model hashes and cache.
    Per-site imagery failures are reported, rather than silently creating boxes.
    """
    root=Path(root).resolve()
    if not isinstance(sites,list) or not 1<=len(sites)<=MAX_BATCH:
        raise ValueError(f'Scan requires 1–{MAX_BATCH} reference hospital sites per batch')
    if not .05<=float(confidence)<=.95:
        raise ValueError('Invalid detection threshold')
    weights,weights_sha=_valid_weak_checkpoint(root)
    clean=[]
    for site in sites:
        lat,lon=float(site['lat']),float(site['lon'])
        if not (math.isfinite(lat) and math.isfinite(lon) and 41.7<=lat<=57 and -95.2<=lon<=-73.8):
            raise ValueError('Only Ontario hospital reference points supported')
        name=str(site.get('name','Reference hospital'))[:140]
        clean.append((round(lat,7),round(lon,7),name))
    cache_dir=root/'weak_osm_experiment'/'area_scan_cache'
    result=[];errors=[];cached=0;scanned=0;small_rejected=0
    if model is None:
        from ultralytics import YOLO
        model=YOLO(str(weights))
    if downloader is None:
        ai_file=root/'source'/'hospital_ai.py'
        if not ai_file.is_file():raise FileNotFoundError('Missing saved imagery downloader: run Notebook 08 setup')
        downloader=_load_model_source(ai_file,'oss4008_area_imagery_source').download_ontario_chip
    for lat,lon,name in clean:
        token=hashlib.sha256(f'{SCAN_VERSION}|{weights_sha}|{lat:.7f}|{lon:.7f}|{float(confidence):.3f}'.encode()).hexdigest()[:24]
        cache=cache_dir/(token+'.json')
        old=_read_valid_cache(cache,weights_sha=weights_sha,site_lat=lat,site_lon=lon,confidence=float(confidence))
        if old is not None:
            cached+=1;result+=old;continue
        tif=None
        try:
            tif=_read_reuse_image(root,lat,lon)
            if tif is None:
                png=cache_dir/'tiles'/(token+'.png');tif=cache_dir/'tiles'/(token+'.tif')
                if not tif.is_file():
                    png.parent.mkdir(parents=True,exist_ok=True)
                    downloader(lat,lon,png,tif)
            with rasterio.open(tif) as ds:
                if ds.crs is None or ds.count<3 or ds.width<200 or ds.height<200:
                    raise ValueError('Real image must be georeferenced RGB')
                image=Image.fromarray(ds.read((1,2,3)).transpose(1,2,0))
                rows=[]
                for predicted in model.predict(source=image,conf=float(confidence),verbose=False):
                    for box in predicted.boxes:
                        if int(box.cls[0])!=0:continue
                        xyxy=[float(k) for k in box.xyxy[0].tolist()]
                        try:
                            geo=bbox_polygon(ds,xyxy)
                            metrics=roof_box_metrics(geo)
                        except (ValueError,KeyError,IndexError):continue
                        if not passes_roof_screen(metrics):
                            small_rejected+=1
                            continue
                        ring=geo['coordinates'][0]
                        centerlat=(ring[0][1]+ring[2][1])/2
                        centerlon=(ring[0][0]+ring[2][0])/2
                        if great_circle_m(lat,lon,centerlat,centerlon)>max_distance_m:continue
                        conf=float(box.conf[0]);
                        if not math.isfinite(conf) or not 0<=conf<=1:continue
                        rows.append({'type':'Feature','geometry':geo,'properties':{
                            'confidence':round(conf,6),'method':'weak_yolo_geo_bbox',
                            'geometry_kind':'rectangle_approximation_not_exact_roof_outline',
                            'size_screen':'geometry_heuristic_NOT_hospital_classification',
                            **metrics,
                            'weights_sha256':weights_sha,'source_name':name,
                            'source':'Experimental YOLO applied to licensed Ontario overhead imagery near ODHF reference',
                            'warning':'UNVERIFIED candidate. Model trained on map-derived proxy labels; hospital use not established.'}})
                        if len(rows)>=80:break
                    if len(rows)>=80:break
            _save_cache(cache,{'version':SCAN_VERSION,'weights_sha256':weights_sha,'site':[lat,lon],
                         'confidence':float(confidence),'features':rows})
            result+=rows;scanned+=1
        except Exception as exc:
            errors.append({'site':name,'error':type(exc).__name__+': '+str(exc)[:160]})
    return {'ok':True,'mode':'reference_guided_area_scan','experimental':True,
            'features':result,'sites_requested':len(clean),'sites_processed':scanned+cached,
            'new_sites':scanned,'cached_sites':cached,'predictions':len(result),
            'small_roof_boxes_rejected_this_run':small_rejected,
            'errors':errors,'model_sha256':weights_sha,
            'scope':'ONLY image chips centered on known Statistics Canada hospital sites, NOT a full citywide search',
            'note':'Green polygons are YOLO bounding rectangles, NOT true roof footprint geometry or verified hospital identities.'}
