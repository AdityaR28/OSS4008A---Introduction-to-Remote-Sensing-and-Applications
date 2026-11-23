"""OpenStreetMap-assisted *DRAFT* roof boxes for the OSS4008 hospital project.

OSM describes mapped buildings, not verified current hospital use. Every draft
remains outside the ground-truth LabelMe folder and is excluded from training.
No data here is a trained-model prediction or a validated hospital label.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from shapely.ops import transform as geometry_transform
from PIL import Image, ImageDraw, ImageFont

OVERPASS_ENDPOINT = 'https://overpass.private.coffee/api/interpreter'
# Public mirrors are not guaranteed. Use only after a connection/service failure,
# never to circumvent an explicit 429 quota/rate-limit response.
OVERPASS_FALLBACK_ENDPOINTS = ('https://overpass-api.de/api/interpreter',)
OSM_LICENSE_URL = 'https://www.openstreetmap.org/copyright'
OSM_CREDIT = '© OpenStreetMap contributors (ODbL)'
BLOCKED_BUILDING_TYPES = {'no','roof','garage','garages','carport','shed','house','detached',
                          'residential','apartments','semidetached_house','bungalow','terrace',
                          'hut','greenhouse','parking','industrial','warehouse'}


def query_for_centers(centers: list[tuple[float,float]], radius_m: int = 180) -> str:
    if not centers or len(centers)>8 or not 70<=radius_m<=300:
        raise ValueError('Query must include 1..8 centres and a 70..300 m radius')
    parts=[]
    for lat,lon in centers:
        if not 41<=float(lat)<=56 or not -96<=float(lon)<=-70:
            raise ValueError('Only Canadian Ontario-region coordinates are supported by the pilot')
        position=f'(around:{radius_m},{lat:.6f},{lon:.6f})'
        for tag in ('["building"]','["amenity"="hospital"]','["healthcare"="hospital"]'):
            parts.append('way'+tag+position+';')
        parts.extend(['relation["amenity"="hospital"]'+position+';',
                      'relation["healthcare"="hospital"]'+position+';'])
    return '[out:json][timeout:20];\n(\n'+ '\n'.join(parts)+'\n);\nout geom;'


def fetch_overpass(query: str, session: Any | None = None, endpoint: str = OVERPASS_ENDPOINT,
                   retries: int = 0, sleep=time.sleep,
                   fallback_endpoints: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Sequential fallback for timeouts/5xx; never evade OSM HTTP 429.

    Use the provider's current private.coffee hostname. Only try the public
    secondary instance when the primary cannot answer. Callers supply a session
    or fetcher in synthetic tests, so real live reliability is not presumed.
    """
    if not query or len(query) > 20000 or not 0 <= retries <= 2:
        raise ValueError('Invalid Overpass query/retry configuration')
    if fallback_endpoints is None:
        # Injected mock sessions default to one endpoint; explicit tests can
        # still supply fallback_endpoints to exercise sequential failover.
        fallback_endpoints = OVERPASS_FALLBACK_ENDPOINTS if session is None else ()
    servers = (endpoint,) + tuple(x for x in fallback_endpoints if x != endpoint)
    if not all(s.startswith('https://') and '/api/interpreter' in s for s in servers):
        raise ValueError('Invalid HTTPS Overpass server')
    http = session if session is not None else requests.Session()
    last_error = None
    headers = {'User-Agent':'OSS4008-HospitalInfrastructureCourse/1.0 (non-commercial education)',
               'Accept':'application/json'}
    for idx, server in enumerate(servers):
        for trial in range(retries + 1):
            try:
                response = http.post(server, data={'data': query},
                                     timeout=(5, 20), headers=headers)
                if response.status_code == 429:
                    # An explicit rate limit must never be bypassed by rotating
                    # to another public server in this teaching workflow.
                    raise RuntimeError('Overpass returned 429: rate limit; stop and retry later')
                if response.status_code in (502, 503, 504):
                    raise requests.HTTPError(f'Overpass returned HTTP {response.status_code}')
                response.raise_for_status()
                if hasattr(response, 'content') and len(response.content) > 25_000_000:
                    raise ValueError('Unexpectedly large OSM response; narrow the query')
                result = response.json()
                if not isinstance(result, dict) or not isinstance(result.get('elements'), list):
                    raise ValueError('Overpass returned no usable element list')
                if len(result['elements']) > 25000:
                    raise ValueError('OSM response too large; use smaller batches')
                return result
            except RuntimeError as exc:
                # Do not evade a provider rate limit.
                if '429' in str(exc):
                    raise
                last_error = exc
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
            if trial < retries:
                sleep(min(6, 2 * (trial + 1)))
        if idx + 1 < len(servers):
            print('OSM primary unavailable; trying separate public Overpass mirror...', flush=True)
    raise RuntimeError(f'OpenStreetMap fetch failed after {len(servers)} server(s): {last_error}')


def _way_polygons(element: dict) -> list[Polygon]:
    """Get WGS84 polygons, including relation *outer* member boundaries."""
    raw=[]
    if element.get('type')=='way':
        raw=[element.get('geometry') or []]
    elif element.get('type')=='relation':
        raw=[v.get('geometry') or [] for v in element.get('members',[])
             if v.get('type')=='way' and v.get('role')=='outer']
    result=[]
    for ring in raw:
        try:
            coords=[(float(p['lon']),float(p['lat'])) for p in ring]
            if len(coords)<4 or coords[0]!=coords[-1]:
                continue
            poly=Polygon(coords)
            if not poly.is_valid:
                poly=poly.buffer(0)
            if poly.geom_type=='Polygon' and not poly.is_empty and poly.area>0:
                result.append(poly)
        except (ValueError,TypeError,KeyError):
            pass
    return result


def extract_suggestions(elements: list[dict], lat: float, lon: float,
                        geotiff: str | Path, max_boxes: int = 7) -> list[dict]:
    """Find candidate *roof* footprints, NEVER hospital-site boundary shapes.

    Confidence labels mean evidence strength for HUMAN review, not model
    probabilities. Generic buildings are only considered if inside a mapped
    hospital campus or very near a known-site point; neither proves hospital use.
    """
    if not 0<=max_boxes<=25:
        raise ValueError('max_boxes must be 0..25')
    grounds=[]
    buildings=[]
    for element in elements:
        tags=element.get('tags') or {}
        is_hospital=tags.get('amenity')=='hospital' or tags.get('healthcare')=='hospital'
        btype=tags.get('building','')
        polys=_way_polygons(element)
        if is_hospital and (not btype or btype=='no'):
            grounds.extend(polys)
        if btype and btype not in BLOCKED_BUILDING_TYPES:
            for poly in polys:
                buildings.append((element,poly))
    # Clip in ORIGINAL georeferenced image space. A 100m road/campus polygon
    # cannot be treated as a building roof merely because it contains the site.
    candidates=[]
    with rasterio.open(geotiff) as ds:
        if ds.crs is None or ds.width<50 or ds.height<50:
            raise ValueError('Image lacks valid geographic reference')
        transformer=Transformer.from_crs('EPSG:4326',ds.crs,always_xy=True)
        px_inverse=~ds.transform
        ref_geom=Point(*transformer.transform(lon,lat))
        # Point-distance uses projected metres; Ontario imagery stores EPSG:3857.
        for element,poly in buildings:
            tags=element.get('tags') or {}
            try:
                proj=geometry_transform(transformer.transform,poly)
                if proj.is_empty or proj.area<35 or proj.area>60000:
                    continue
                geo_centre=proj.centroid
                # Web Mercator overstates ground distance by ~1/cos(lat).
                meters=geo_centre.distance(ref_geom)*max(.25,math.cos(math.radians(lat)))
                matched=tags.get('amenity')=='hospital' or tags.get('healthcare')=='hospital'
                typed=tags.get('building')=='hospital'
                campus=any(proj.intersection(geometry_transform(transformer.transform,g)).area/
                           max(proj.area,1)>0.55 for g in grounds)
                if matched:
                    why='explicit_hospital_use_tag'
                    strength=3
                elif typed:
                    why='building_hospital_tag_not_confirmed_current_use'
                    strength=2
                elif campus:
                    why='building_inside_mapped_hospital_campus'
                    strength=1
                elif meters <= 95 and not grounds:
                    why='near_odhf_reference_only'
                    strength=0
                else:
                    continue
                # Pixel affine inverse: x=column; y=row.
                coords=[px_inverse * p for p in proj.exterior.coords]
                x1=max(0.0,min(p[0] for p in coords));x2=min(float(ds.width),max(p[0] for p in coords))
                y1=max(0.0,min(p[1] for p in coords));y2=min(float(ds.height),max(p[1] for p in coords))
                if x2-x1 < 6 or y2-y1 < 6 or (x2-x1)*(y2-y1)>ds.width*ds.height*.75:
                    continue
                raw=[p[0] for p in coords]+[p[1] for p in coords]
                if not all(math.isfinite(v) for v in raw):continue
                candidates.append({
                    'osm_id':f"{element['type']}/{element['id']}",'bbox':[round(x1,1),round(y1,1),round(x2,1),round(y2,1)],
                    'reason':why,'evidence_level':strength,'point_distance_m':round(meters,1),
                    'osm_tags':{k:str(v)[:120] for k,v in tags.items() if k in ('building','amenity','healthcare','name')},
                })
            except (ValueError,TypeError,OverflowError):
                continue
    # Strong OSM tags first; then closest campus/point buildings. Limit weak
    # near-reference guesses to at most three per tile, never silently mark as hospital.
    candidates.sort(key=lambda x:(-x['evidence_level'],x['point_distance_m']))
    accepted=[]; weak_count=0
    def iou(a,b):
        l=max(a[0],b[0]);t=max(a[1],b[1]);r=min(a[2],b[2]);bt=min(a[3],b[3]);inter=max(0,r-l)*max(0,bt-t)
        return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    for item in candidates:
        if item['evidence_level']==0 and weak_count>=3:continue
        if any(iou(item['bbox'],other['bbox'])>.70 for other in accepted):continue
        accepted.append(item)
        if item['evidence_level']==0:weak_count+=1
        if len(accepted)>=max_boxes:break
    return accepted


def load_draft_boxes(root: str | Path,tile_id: str) -> list[list[float]]:
    """Read local proposed boxes; these never indicate approved hospital use."""
    root=Path(root).resolve()
    fname=root/'drafts'/'osm'/f'{tile_id}.json'
    if not fname.is_file():return []
    doc=json.loads(fname.read_text(encoding='utf-8'))
    if doc.get('tile_id')!=tile_id or doc.get('approved') is not False:
        raise ValueError('OSM draft identity/status mismatch')
    return [r['bbox'] for r in doc.get('candidates',[]) if len(r.get('bbox',[]))==4]


def generate_osm_drafts(root: str | Path, limit: int = 50, batch_size: int = 4,
                        session: Any | None = None, pause_s: float = 1.5,
                        sleeper=time.sleep, fetcher=fetch_overpass,
                        overwrite_drafts: bool = False) -> dict[str, Any]:
    """Batch Overpass lookups, cache, create draft boxes (NEVER approved labels).

    Returns actual coverage; '100' is a target, not a promise. Skips already
    approved/rejected images; never touches review_manifest.csv or verified JSON.
    """
    root=Path(root).resolve()
    if not 1<=limit<=200 or not 1<=batch_size<=8:
        raise ValueError('Unsupported draft target/batch size')
    manifest_path=root/'review_manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError('Download pilot aerial images first: review_manifest.csv missing')
    df=pd.read_csv(manifest_path,keep_default_na=False)
    if not {'tile_id','image','geotiff','latitude','longitude','candidate_class','review_status'}.issubset(df.columns):
        raise ValueError('Manifest missing required imagery/label fields')
    df=df.loc[df.candidate_class.eq('known_hospital_site') & df.review_status.eq('UNREVIEWED')].head(limit)
    rows=[]; errors=[]; pending=[]; skipped=0
    outdir=root/'drafts'/'osm';outdir.mkdir(parents=True,exist_ok=True)
    for _,r in df.iterrows():
        name=str(r.tile_id)
        if '/' in name or '..' in name or not name:
            errors.append(f'{name}: invalid tile id');continue
        if not (root/str(r.image)).is_file() or not (root/str(r.geotiff)).is_file():
            errors.append(f'{name}: missing image or GeoTIFF');continue
        path=outdir/f'{name}.json'
        if path.is_file() and not overwrite_drafts:
            try:
                prev=json.loads(path.read_text(encoding='utf-8'))
                if prev.get('approved') is False:
                    rows.append({'tile_id':name,'draft_boxes':len(prev.get('candidates',[])),'cached':True})
                    skipped+=1;continue
            except (OSError,ValueError):pass
        try:
            coords=(float(r.latitude),float(r.longitude))
            if not (41<=coords[0]<=56 and -96<=coords[1]<=-70):
                raise ValueError('Outside supported Ontario pilot region')
            pending.append((r,coords))
        except ValueError as exc:
            errors.append(f'{name}: {exc}')
    # Show progress BEFORE contacting the remote server; interruption retains
    # all existing images and completed per-image draft JSON files in Drive.
    total_batches = math.ceil(len(pending) / batch_size)
    print(f'OSM matching: {len(df)} candidate hospital images (max {limit}), '
          f'{skipped} cached, {len(pending)} to query in {total_batches} small batches.', flush=True)
    progress_file = root / 'drafts' / 'osm_progress.json'
    def write_progress(stage, batch_number, last_message=''):
        progress = {'stage': stage, 'batch': batch_number, 'total_batches': total_batches,
                    'images_checked_or_cached': len(rows),
                    'images_with_drafts': sum(bool(item['draft_boxes']) for item in rows),
                    'draft_boxes': sum(item['draft_boxes'] for item in rows),
                    'errors_so_far': len(errors), 'last_message': str(last_message)[:250],
                    'note': 'Unverified OSM draft boxes, never auto-approved training labels'}
        tmp=progress_file.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        tmp.replace(progress_file)
    write_progress('STARTED', 0)
    consecutive_failures=0
    for start in range(0,len(pending),batch_size):
        batch=pending[start:start+batch_size]
        batch_number=start//batch_size+1
        print(f'OSM batch {batch_number}/{total_batches}: looking up {len(batch)} saved images...', flush=True)
        write_progress('REQUESTING', batch_number)
        query=query_for_centers([c for _,c in batch])
        try:
            response=fetcher(query,session=session,sleep=sleeper)
            elements=response['elements']
        except Exception as exc:
            consecutive_failures+=1
            message=f'OSM request {batch_number}: {type(exc).__name__}: {str(exc)[:160]}'
            errors.append(message)
            print(f'OSM batch {batch_number}/{total_batches} failed: {type(exc).__name__}: {str(exc)[:120]}', flush=True)
            write_progress('BATCH_FAILED', batch_number, message)
            # Do not hammer public OSM servers after repeated timeouts/429s.
            if consecutive_failures>=2:
                errors.append('Stopped after consecutive Overpass errors; existing draft files are preserved.')
                print('OSM: stopping after two failed batches. Previously saved drafts are intact.', flush=True)
                write_progress('STOPPED_ON_SERVER_ERRORS', batch_number)
                break
            continue
        consecutive_failures=0
        for r,(lat,lon) in batch:
            try:
                suggested=extract_suggestions(elements,lat,lon,root/str(r.geotiff))
                data={'tile_id':str(r.tile_id),'approved':False,'verification':'REQUIRES_HUMAN_REVIEW',
                      'source':'OpenStreetMap', 'license_url':OSM_LICENSE_URL,'attribution':OSM_CREDIT,
                      'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_site':str(r.get('name','')),
                      'candidate_class':str(r.candidate_class),'candidates':suggested}
                target=outdir/f'{r.tile_id}.json';temp=target.with_suffix('.json.tmp')
                temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                temp.replace(target)
                rows.append({'tile_id':str(r.tile_id),'draft_boxes':len(suggested),'cached':False})
            except Exception as exc:
                errors.append(f'{r.tile_id}: {type(exc).__name__}: {str(exc)[:130]}')
        print(f'OSM batch {batch_number}/{total_batches} complete: '
              f'{len(rows)} images checked (including cached); '
              f'{sum(bool(item["draft_boxes"]) for item in rows)} images have draft boxes.', flush=True)
        write_progress('BATCH_COMPLETE', batch_number)
        if start+batch_size<len(pending) and pause_s>0:
            sleeper(pause_s)
    summary={'requested_max':limit,'candidate_images':len(df),'queried_or_cached':len(rows),
             'images_with_drafts':sum(bool(r['draft_boxes']) for r in rows),
             'proposed_boxes':sum(r['draft_boxes'] for r in rows),'already_cached':skipped,
             'approved_automatically':0,'trained_models':0,
             'errors':errors[:50],'attribution':OSM_CREDIT,
             'message':'OSM polygons are draft boxes ONLY. Independently verify hospital function and alignment before training.'}
    report=root/'drafts'/'osm_inventory.json'
    report.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(rows,columns=['tile_id','draft_boxes','cached']).to_csv(root/'drafts'/'osm_inventory.csv',index=False)
    write_progress('FINISHED', total_batches)
    print(f'OSM matching finished: {summary["images_with_drafts"]} images with drafts, '
          f'{summary["proposed_boxes"]} draft boxes; automatically approved: 0.', flush=True)
    return summary


def make_draft_contact_sheet(root: str | Path, max_items: int = 36, thumb: int = 230) -> Path:
    """Prepare a compact visual review grid with tentative proposed boxes."""
    root=Path(root)
    df=pd.read_csv(root/'review_manifest.csv',keep_default_na=False)
    tiles=[]
    for _,r in df.iterrows():
        draft=root/'drafts'/'osm'/f'{r.tile_id}.json'
        image=root/str(r.image)
        if draft.is_file() and image.is_file():
            doc=json.loads(draft.read_text());
            if doc.get('candidates'):tiles.append((r,image,doc))
            if len(tiles)>=max_items:break
    if not tiles:
        raise ValueError('No OSM draft boxes available to preview')
    cols=4;rows=math.ceil(len(tiles)/cols)
    size=(cols*(thumb+12),rows*(thumb+48))
    result=Image.new('RGB',size,'white'); draw=ImageDraw.Draw(result)
    for index,(r,image,doc) in enumerate(tiles):
        with Image.open(image) as im:
            w,h=im.size;resized=im.convert('RGB').resize((thumb,thumb))
        x=index%cols*(thumb+12);y=index//cols*(thumb+48)
        result.paste(resized,(x,y))
        for candidate in doc['candidates']:
            x1,y1,x2,y2=candidate['bbox'];scx=thumb/w;scy=thumb/h
            color='#2764ef' if candidate['evidence_level']>=2 else '#efa629'
            draw.rectangle((x+x1*scx,y+y1*scy,x+x2*scx,y+y2*scy),outline=color,width=3)
        draw.text((x+2,y+thumb+3),str(r['name'])[:30],fill='#17213b')
        draw.text((x+2,y+thumb+21),'DRAFT boxes: '+str(len(doc['candidates'])),fill='#9b5311')
    out=root/'drafts'/'osm_contact_sheet.jpg';out.parent.mkdir(parents=True,exist_ok=True)
    result.save(out,quality=87)
    return out
