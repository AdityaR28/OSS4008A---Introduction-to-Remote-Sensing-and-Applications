"""Experimental Overture building footprints near known ODHF hospital points.

NO annotated building here is verified to be a hospital. This module proposes
weak *training hypotheses*, does not modify any human-reviewed labels, and
never claims validation or hospital identity from proximity alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import warnings
from typing import Callable

import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from pyproj import Transformer
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely import wkb
from shapely.ops import transform as geom_transform

ATTRIBUTION = '© OpenStreetMap contributors, Overture Maps Foundation'
LICENSE_URL = 'https://docs.overturemaps.org/attribution/'
EXCLUDE_CLASSES = {'house', 'apartments', 'residential', 'garage', 'garages',
                   'shed', 'carport', 'school', 'hotel', 'university', 'industrial',
                   'warehouse', 'retail', 'commercial', 'detached', 'bungalow'}


def fetch_overture_bbox(bbox):
    """Fetch only buildings in a small bounding box with official Overture client.

    Raise on missing STAC index rather than silently scanning the whole planet.
    """
    from overturemaps import record_batch_reader
    from overturemaps.core import StacIndexWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', StacIndexWarning)
        reader = record_batch_reader('building', bbox=bbox)
        if reader is None:
            return []
        rows = reader.read_all()
    if rows.num_rows > 6000:
        raise RuntimeError('Too many Overture buildings for one roof chip; refusing large extraction')
    return rows.to_pylist()


def _geometry(item):
    geom = item.get('geometry')
    if isinstance(geom, (bytes, bytearray, memoryview)):
        return wkb.loads(bytes(geom))
    if isinstance(geom, dict):
        return shape(geom)
    return None


def roof_candidates(buildings, *, latitude: float, longitude: float, geotiff):
    """Select mapped *building geometry*, not ODHF-site circles or grounds.

    Prefer a building explicitly classified as a hospital. Fallback is at most
    one physically nearby nonresidential building: very weak pseudo-label.
    """
    to_merc = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    reference = Point(*to_merc.transform(longitude, latitude))
    physical_scale = max(.3, math.cos(math.radians(latitude)))
    ranked = []
    with rasterio.open(geotiff) as ds:
        if ds.crs is None or ds.width < 100 or ds.height < 100:
            raise ValueError('Source GeoTIFF missing georeference or invalid size')
        to_raster = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
        pixel = ~ds.transform
        for item in buildings:
            category = str(item.get('class') or '').casefold().strip()
            if category in EXCLUDE_CLASSES:
                continue
            geom = _geometry(item)
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, MultiPolygon):
                geom = max(geom.geoms, key=lambda p: p.area)
            if not isinstance(geom, Polygon):
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not isinstance(geom, Polygon) or geom.is_empty:
                continue
            projected = geom_transform(to_merc.transform, geom)
            area_m2 = projected.area * physical_scale ** 2
            if not 45 <= area_m2 <= 60000:
                continue
            distance_m = projected.distance(reference) * physical_scale
            explicit = category == 'hospital'
            if distance_m > (120 if explicit else 45):
                continue
            raster_poly = geom_transform(to_raster.transform, geom)
            xys = [pixel * point for point in raster_poly.exterior.coords]
            vals = [v for p in xys for v in p]
            if not all(math.isfinite(v) for v in vals):
                continue
            x1 = max(0, min(p[0] for p in xys))
            x2 = min(ds.width, max(p[0] for p in xys))
            y1 = max(0, min(p[1] for p in xys))
            y2 = min(ds.height, max(p[1] for p in xys))
            if x2-x1 < 7 or y2-y1 < 7 or (x2-x1)*(y2-y1) > ds.width*ds.height*.75:
                continue
            evidence = 2 if explicit else 1
            ranked.append((0 if explicit else 1, distance_m, {
                'osm_id': 'overture/' + str(item.get('id') or ''),
                'bbox': [round(x1, 1),round(y1,1),round(x2,1),round(y2,1)],
                'evidence_level': evidence,
                'reason': 'overture_building_class_hospital' if explicit else 'overture_nearest_footprint_to_odhf_reference_only',
                'point_distance_m': round(distance_m,1),
                'source': 'Overture Maps buildings',
                'warning': 'UNVERIFIED use; nearest building is not proof of hospital identity',
                'overture_class': category,
            }))
    ranked.sort(key=lambda row:(row[0],row[1]))
    hospitals = [obj for level,_,obj in ranked if level == 0]
    return hospitals[:3] if hospitals else [ranked[0][2]] if ranked else []


def generate_overture_drafts(root, limit=50, target_sites=8, provider: Callable=fetch_overture_bbox):
    """Bounded fallback with per-image checkpoints, no simulated/approved labels.

    Stops after enough independent draft sites have been collected. Fail-fast
    on remote access error, so Colab doesn't spin through 50 dead requests.
    """
    root=Path(root).resolve()
    if not 1<=limit<=100 or not 1<=target_sites<=30:
        raise ValueError('Invalid candidate limits')
    manifest = root/'review_manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError('Saved review manifest is missing')
    df = pd.read_csv(manifest, keep_default_na=False)
    wanted = {'tile_id','site_id','image','geotiff','latitude','longitude','review_status','candidate_class'}
    if not wanted.issubset(df.columns):
        raise ValueError('Review manifest missing essential columns')
    df=df.loc[df.candidate_class.eq('known_hospital_site') & df.review_status.eq('UNREVIEWED')].head(limit)
    dest=root/'drafts'/'overture';dest.mkdir(parents=True,exist_ok=True)
    success_sites=set(); used=0; cached=0; errors=[]; proposals=0
    for row in df.itertuples(index=False):
        tile=str(row.tile_id); site=str(row.site_id)
        if not tile or '/' in tile or '..' in tile:
            continue
        image=(root/str(row.image)).resolve(); tif=(root/str(row.geotiff)).resolve()
        if not image.is_relative_to(root) or not tif.is_relative_to(root) or not image.is_file() or not tif.is_file():
            continue
        # Preserve any positive OpenStreetMap proposals; never modify them.
        osm=root/'drafts'/'osm'/f'{tile}.json'
        if osm.is_file():
            try:
                prev=json.loads(osm.read_text())
                if prev.get('approved') is False and any(int(c.get('evidence_level',0))>=1 for c in prev.get('candidates',[])):
                    success_sites.add(site)
                    continue
            except (ValueError,TypeError,OSError):
                pass
        out=dest/(tile+'.json')
        if out.is_file():
            try:
                doc=json.loads(out.read_text())
                if doc.get('tile_id')==tile and doc.get('approved') is False and doc.get('candidates'):
                    success_sites.add(site); cached += 1
                    if len(success_sites)>=target_sites:
                        break
                    continue
            except (ValueError,TypeError,OSError):pass
        if len(success_sites)>=target_sites:
            break
        try:
            with rasterio.open(tif) as ds:
                extent=transform_bounds(ds.crs,'EPSG:4326',*ds.bounds, densify_pts=10)
            print(f'Overture fallback {used+1}: checking saved site {tile}...',flush=True)
            elements=provider(extent)
            used += 1
            candidates=roof_candidates(elements,latitude=float(row.latitude),longitude=float(row.longitude),geotiff=tif)
            data={'tile_id':tile,'site_id':site,'approved':False,'verification':'NOT_GROUND_TRUTH',
                  'source':'Overture Maps buildings + Statistics Canada ODHF reference proximity',
                  'license_url':LICENSE_URL,'attribution':ATTRIBUTION,
                  'created_at_utc':datetime.now(timezone.utc).isoformat(),
                  'warning':'UNVERIFIED Overture roof hypothesis; NOT a confirmed hospital building',
                  'candidates':candidates}
            tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(out)
            if candidates:
                success_sites.add(site)
                proposals += len(candidates)
            print(f'Overture complete: {len(success_sites)} independent candidate sites, {len(candidates)} new roof boxes.',flush=True)
        except Exception as exc:
            msg=f'{type(exc).__name__}: {str(exc)[:210]}'
            errors.append(f'{tile}: {msg}')
            print('Overture unavailable; stopping rather than retrying every image: '+msg,flush=True)
            break
    return {'source':'Overture Maps','candidate_sites_with_proposals':len(success_sites),
            'requests':used,'cached':cached,'new_proposals':proposals,'errors':errors,
            'status':'DRAFTS_ONLY_NOT_VERIFIED'}
