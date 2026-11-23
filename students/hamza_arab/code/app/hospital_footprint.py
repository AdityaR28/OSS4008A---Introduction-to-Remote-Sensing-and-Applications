"""Optional, ON-DEMAND map-derived footprint near an ODHF reference.

This is NOT an AI detection or confirmation of hospital building use.
It runs only when a user explicitly asks for the map outline, not per city.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import wkb
from shapely.ops import transform as geometry_transform
from pyproj import Transformer

VERSION='mapped_large_buildings_osm_fallback_v4'
# A large complex may contain multiple separate wings/roof polygons. All are
# source-derived *hypotheses*, not confirmations of hospital use.
MAX_FOOTPRINTS=40
MAX_SITE_RADIUS_M=350
LINK_GAP_M=28
MIN_WING_AREA_M2=130
EXCLUDE_CLASSES={'house','residential','apartments','garage','garages','shed','carport',
                 'school','hotel','retail','warehouse','industrial','university',
                 'detached','semidetached_house','terrace','bungalow','cabin','hut',
                 'dormitory','static_caravan','farm','farm_auxiliary','greenhouse'}


def _individual_polygons(geometry):
    if isinstance(geometry,(bytes,bytearray,memoryview)):
        geometry=wkb.loads(bytes(geometry))
    elif isinstance(geometry,dict):
        from shapely.geometry import shape
        geometry=shape(geometry)
    if not geometry or geometry.is_empty:
        return []
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [geometry] if isinstance(geometry,Polygon) else []


def select_footprints(rows, latitude, longitude):
    """Return connected, mapped roof hypotheses around a hospital reference.

    A hospital may span many structures. Select one or more plausible anchors,
    then physically adjoining/nearby wings; do not blindly mark every building
    in a 270-m circle as a hospital. No Overture feature certifies current use.
    """
    lat,lon=float(latitude),float(longitude)
    if not (41<=lat<=84 and -142<=lon<=-52):
        raise ValueError('Canadian reference coordinates required')
    project=Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True).transform
    reference=Point(*project(lon,lat)); factor=max(.2,math.cos(math.radians(lat)))
    pool=[];seen=set()
    for row in rows:
        try:
            category=str(row.get('class') or '').casefold().strip()
            if category in EXCLUDE_CLASSES:continue
            height=row.get('height')
            try:height=float(height) if height is not None else None
            except (TypeError,ValueError):height=None
            if height is not None and not (0<height<500):height=None
            for part in _individual_polygons(row.get('geometry')):
                if not part.is_valid:part=part.buffer(0)
                if not isinstance(part,Polygon) or part.is_empty:continue
                geom=geometry_transform(project,part)
                area=geom.area*factor*factor
                distance=geom.distance(reference)*factor
                if not (MIN_WING_AREA_M2<=area<=120000 and distance<=MAX_SITE_RADIUS_M):continue
                coords=[[round(float(x),7),round(float(y),7)] for x,y in part.exterior.coords]
                if not (4<=len(coords)<=1500) or any(not (-142<=x<=-52 and 41<=y<=84) for x,y in coords):continue
                shape_key=tuple(round(t,5) for t in part.bounds)
                if shape_key in seen:continue
                seen.add(shape_key)
                pool.append({'geometry':geom,'ring':coords,'area':area,'distance':distance,
                             'category':category,'height':height,'id':str(row.get('id') or '')})
        except (ValueError,TypeError,KeyError,OverflowError):continue
    if not pool:return []
    # Explicit hospital-use classification is most useful. Otherwise anchor at
    # a substantial building nearest the reference pin, never a random house.
    explicit=[i for i,p in enumerate(pool) if p['category']=='hospital' and p['distance']<=220]
    anchors=explicit[:]
    if not anchors:
        near=[i for i,p in enumerate(pool) if p['distance']<=100 and p['area']>=650]
        if not near:return []
        anchors=[min(near,key=lambda i:(pool[i]['distance'], -min(pool[i]['area'],15000)))]
    selected=set(anchors)
    # Expand across mapped nearby wings. Link gaps reflect separation between
    # adjacent building polygons, but long chains across neighborhoods are avoided.
    changed=True
    while changed and len(selected)<MAX_FOOTPRINTS:
        changed=False
        for i,candidate in enumerate(pool):
            if i in selected or candidate['distance']>MAX_SITE_RADIUS_M:continue
            if candidate['category']=='hospital' and candidate['distance']<=220:
                selected.add(i);changed=True;continue
            gap=min(candidate['geometry'].distance(pool[j]['geometry'])*factor for j in selected)
            if gap<=LINK_GAP_M and (candidate['distance']<=185 or gap<=12):
                selected.add(i);changed=True
            if len(selected)>=MAX_FOOTPRINTS:break
    ordered=sorted(selected,key=lambda i:(0 if pool[i]['category']=='hospital' else 1,
                                          pool[i]['distance'],-pool[i]['area']))[:MAX_FOOTPRINTS]
    features=[]
    for i in ordered:
        c=pool[i]
        evidence=('overture_class_hospital' if c['category']=='hospital'
                  else 'nearby_connected_footprint_unverified')
        features.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[c['ring']]},
            'properties':{'method':'mapped_building_near_odhf_unverified',
                          'evidence':evidence,'source':'Overture Maps buildings + Statistics Canada ODHF point',
                          'area_m2':round(c['area']),'distance_from_reference_m':round(c['distance']),
                          'height_m_if_mapped':c['height'],'overture_id':c['id'],
                          'warning':'Mapped roof/wing candidate: NOT AI detection, NOT verified hospital use'}})
    return features


def select_nearby_roofs(rows, latitude, longitude, radius_m=180, *, max_footprints=MAX_FOOTPRINTS, minimum_area_m2=900):
    """Trace real mapped large-building polygons inside a circle around a point.

    These are OBJECT geometries from Overture, not inferred image segmentation,
    and not hospital identifications. The reference point can be on a street or
    parking lot; it must never be used as roof ground truth.
    """
    lat, lon, radius = float(latitude), float(longitude), float(radius_m)
    if not (41 <= lat <= 84 and -142 <= lon <= -52 and 20 <= radius <= 350):
        raise ValueError('Canadian centre and a 20–350 metre radius required')
    if not (200 <= float(minimum_area_m2) <= 5000):
        raise ValueError('Minimum roof area must be 200–5000 m²')
    if not 1 <= int(max_footprints) <= 80:
        raise ValueError('Invalid building-footprint output cap')
    # Local azimuthal equidistant metre projection; Web Mercator gives
    # misleading metre distances and areas at Canadian latitudes.
    crs = f'+proj=aeqd +lat_0={lat:.8f} +lon_0={lon:.8f} +datum=WGS84 +units=m +no_defs'
    project = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform
    centre = Point(0, 0)
    candidates = []
    seen = set()
    for row in rows:
        try:
            category = str(row.get('class') or '').lower().strip()
            if category in EXCLUDE_CLASSES:
                continue
            for part in _individual_polygons(row.get('geometry')):
                if not part.is_valid:
                    part = part.buffer(0)
                if not isinstance(part, Polygon) or part.is_empty:
                    continue
                projected = geometry_transform(project, part)
                area = float(projected.area)
                distance = float(projected.distance(centre))
                # Exclude house-scale generic footprints, but preserve small
                # explicitly hospital-tagged annexes. No height claims.
                minimum = 90 if category == 'hospital' else float(minimum_area_m2)
                if not (minimum <= area <= 160000 and distance <= radius):
                    continue
                if len(part.exterior.coords) > 1500 or any(len(h.coords) > 1500 for h in part.interiors):
                    continue
                boundary = [[round(float(x), 7), round(float(y), 7)] for x, y in part.exterior.coords]
                holes = [[[round(float(x), 7), round(float(y), 7)] for x, y in h.coords] for h in part.interiors]
                if not all(-142 <= x <= -52 and 41 <= y <= 84 for ring in [boundary, *holes] for x, y in ring):
                    continue
                key = (row.get('id'), tuple(round(v, 7) for v in part.bounds))
                if key in seen:
                    continue
                seen.add(key)
                height = row.get('height')
                try:
                    height = float(height) if height is not None else None
                except (ValueError, TypeError):
                    height = None
                if height is not None and not (0 < height < 500 and math.isfinite(height)):
                    height = None
                feature = {'type':'Feature',
                    'geometry':{'type':'Polygon','coordinates':[boundary, *holes]},
                    'properties':{
                        'method':'large_building_within_search_circle_unverified',
                        'evidence':('osm_hospital_tag' if row.get('source')=='OpenStreetMap' and category=='hospital' else 'overture_class_hospital' if category=='hospital' else 'large_mapped_building_near_reference_only'),
                        'source':str(row.get('source') or 'Overture Maps')+' mapped building geometry',
                        'area_m2':round(area), 'distance_from_search_centre_m':round(distance),
                        'radius_m':radius, 'height_m_if_mapped':height,
                        'mapped_building_id':str(row.get('id') or ''),
                        'warning':'Mapped large building, NOT verified hospital or AI detection'
                    }}
                candidates.append((0 if category == 'hospital' else 1, round(distance, 2), -area, feature))
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
    # A clearly hospital-tagged building ranks first. Otherwise favour nearby
    # large roofs rather than tiny buildings beside an offset street pin.
    candidates.sort(key=lambda entry:(entry[0], entry[1]*0.55+200/(1+math.sqrt(-entry[2]/1000)), entry[2]))
    return [item[3] for item in candidates[:int(max_footprints)]]


def overpass_query(latitude, longitude, radius_m):
    """Bounded source-geometry request; building ways/relations, not hospital labels."""
    lat,lon,r=float(latitude),float(longitude),float(radius_m)
    if not (41<=lat<=84 and -142<=lon<=-52 and 20<=r<=350):
        raise ValueError('Invalid Canada-only mapped building request')
    # A source building may overlap the circle while its nodes lie just outside.
    return (f'[out:json][timeout:18];('
            f'way["building"](around:{math.ceil(r+25)},{lat:.7f},{lon:.7f});'
            f'relation["building"](around:{math.ceil(r+25)},{lat:.7f},{lon:.7f});'
            f'way["building:part"](around:{math.ceil(r+25)},{lat:.7f},{lon:.7f});'
            f'relation["building:part"](around:{math.ceil(r+25)},{lat:.7f},{lon:.7f});'
            ');out body geom;')


def osm_rows_from_overpass(payload):
    """Convert real OSM way/multipolygon geometry to the same candidate format."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    if not isinstance(payload,dict) or not isinstance(payload.get('elements'),list):
        raise ValueError('Invalid OpenStreetMap building response')
    if len(payload['elements'])>3000:
        raise ValueError('OSM response too large for one location')
    rows=[]
    for item in payload['elements']:
        try:
            tags=item.get('tags') or {}
            if not (tags.get('building') or tags.get('building:part')):continue
            kind=str(tags.get('building') or tags.get('building:part') or '').lower()
            if kind in EXCLUDE_CLASSES or kind in {'no','roof','greenhouse','barn','stable','farm_auxiliary'}:
                continue
            classified=('hospital' if kind=='hospital' or
                        tags.get('amenity')=='hospital' or tags.get('healthcare')=='hospital' else kind)
            if item.get('type')=='way':
                paths=[item.get('geometry') or []]
                lines=[]
                for path in paths:
                    coords=[(float(point['lon']),float(point['lat'])) for point in path]
                    if len(coords)>=4 and coords[0]==coords[-1]:lines.append(Polygon(coords))
                shapes=lines
            elif item.get('type')=='relation':
                # Multipolygon relations can split an exterior across many ways.
                # Polygonize complete outer rings without inventing missing edges.
                outer=[];inner=[]
                for member in item.get('members') or []:
                    coords=[(float(z['lon']),float(z['lat'])) for z in member.get('geometry') or []]
                    if len(coords)<2:continue
                    if member.get('role')=='outer':outer.append(LineString(coords))
                    elif member.get('role')=='inner':inner.append(LineString(coords))
                shapes=list(polygonize(outer))
                cut=list(polygonize(inner))
                if cut:
                    cutter=unary_union(cut)
                    shapes=[g.difference(cutter) for g in shapes]
            else:continue
            for k,geom in enumerate(shapes):
                if geom.is_empty:continue
                if not geom.is_valid:geom=geom.buffer(0)
                for poly in _individual_polygons(geom):
                    rows.append({'geometry':poly,'class':classified,'id':f'osm/{item["type"]}/{item["id"]}/{k}',
                                 'source':'OpenStreetMap','height':tags.get('height')})
        except (ValueError,TypeError,KeyError,OverflowError):
            continue
    return rows


def fetch_osm_buildings(lat,lon,radius_m, *,post=None):
    """Retrieve *mapped* building outlines from Overpass; never infer roofs."""
    import requests
    request=post or requests.post
    query=overpass_query(lat,lon,radius_m)
    endpoints=('https://overpass-api.de/api/interpreter',
               'https://overpass.kumi.systems/api/interpreter')
    failure=None
    for url in endpoints:
        try:
            response=request(url,data={'data':query},timeout=20,
                             headers={'User-Agent':'OSS4008HospitalResearch/1.1 (educational, bounded queries)',
                                      'Accept':'application/json'})
            response.raise_for_status()
            if len(response.content)>6_000_000:raise ValueError('Overpass response too large')
            return osm_rows_from_overpass(response.json())
        except (requests.RequestException,ValueError,TypeError,KeyError) as exc:
            failure=exc
    raise RuntimeError('OpenStreetMap/Overpass unavailable: '+str(failure)[:100])


def _merge_source_features(collections,lat,lon,cap=MAX_FOOTPRINTS):
    """Merge source geometries without duplicating near-identical roof polygons."""
    from shapely.geometry import shape
    crs=f'+proj=aeqd +lat_0={lat:.7f} +lon_0={lon:.7f} +datum=WGS84 +units=m +no_defs'
    project=Transformer.from_crs('EPSG:4326',crs,always_xy=True).transform
    merged=[];regions=[]
    for features in collections:
        for f in features:
            try:
                poly=geometry_transform(project,shape(f['geometry']))
                if poly.is_empty or poly.area<=0:continue
                if any(poly.intersection(other).area/max(.01,min(poly.area,other.area))>=.8 for other in regions):
                    continue
                merged.append(f);regions.append(poly)
                if len(merged)>=cap:return merged
            except (ValueError,TypeError,KeyError):continue
    return merged


def mapped_footprint(root, lat, lon, *, provider=None, osm_provider=None,
                     radius_m=180, mode='campus', minimum_area_m2=900):
    """Search near a pin/circle. OSM first, then Overture if OSM is insufficient.

    Always return source-derived outlines. Never draw a made-up polygon from a pin.
    Network failures and data gaps are displayed explicitly to the user.
    """
    root=Path(root).resolve();lat=float(lat);lon=float(lon);radius_m=float(radius_m)
    minimum_area_m2=float(minimum_area_m2)
    if not (41<=lat<=84 and -142<=lon<=-52 and 20<=radius_m<=350 and
            200<=minimum_area_m2<=5000):
        raise ValueError('Canadian centre, 20–350 m radius and 200–5000 m² minimum required')
    if mode not in ('nearby','campus'):
        raise ValueError('Unsupported mapped building search mode')
    token=hashlib.sha256(f'{VERSION}:{mode}:{lat:.7f}:{lon:.7f}:{radius_m:.1f}:{minimum_area_m2}'.encode()).hexdigest()[:24]
    path=root/'reference_footprints'/(token+'.json')
    if path.exists():
        try:
            prev=json.loads(path.read_text(encoding='utf-8'))
            if (prev.get('version')==VERSION and prev.get('radius_m')==radius_m and
                prev.get('minimum_area_m2')==minimum_area_m2 and prev.get('mode')==mode and
                isinstance(prev.get('features'),list) and prev['features']):
                return {**prev,'cached':True}
        except (OSError,TypeError,ValueError):pass
    errors=[];used=[];groups=[]
    if mode=='nearby':
        try:
            rows=(osm_provider or fetch_osm_buildings)(lat,lon,radius_m)
            osm=select_nearby_roofs(rows,lat,lon,radius_m,minimum_area_m2=minimum_area_m2)
            groups.append(osm);used.append('OpenStreetMap')
        except Exception as exc:
            errors.append('OpenStreetMap: '+type(exc).__name__+': '+str(exc)[:130])
    # A second source is needed if OSM has too few roof footprints, or in campus mode.
    if mode=='campus' or not groups or len(groups[0])<3:
        try:
            if provider is None:
                from overturemaps import record_batch_reader
                provider=lambda bbox:record_batch_reader('building',bbox=bbox).read_all().to_pylist()
            dlat=(radius_m+25)/110574
            dlon=(radius_m+25)/(111320*max(math.cos(math.radians(lat)),0.2))
            data=provider((lon-dlon,lat-dlat,lon+dlon,lat+dlat))
            if len(data)>8000:raise RuntimeError('Too many mapped building records')
            overture=(select_footprints(data,lat,lon) if mode=='campus' else
                     select_nearby_roofs(data,lat,lon,radius_m,minimum_area_m2=minimum_area_m2))
            groups.append(overture);used.append('Overture Maps')
        except Exception as exc:
            errors.append('Overture Maps: '+type(exc).__name__+': '+str(exc)[:130])
    features=_merge_source_features(groups,lat,lon)
    reply={'ok':True,'version':VERSION,'mode':mode,'radius_m':radius_m,
           'minimum_area_m2':minimum_area_m2,'features':features,'cached':False,
           'sources_attempted':used,'provider_errors':errors,
           'source':' + '.join(used) or 'No source succeeded','max_footprints':MAX_FOOTPRINTS,
           'warning':'These are actual mapped building polygons near a point, NOT verified hospital buildings or AI detections.'}
    # Never cache an empty trace: service outages and temporary empty map responses
    # should not permanently poison the next attempt.
    if features or (mode=='campus' and not errors):
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(reply),encoding='utf-8');tmp.replace(path)
    return reply
