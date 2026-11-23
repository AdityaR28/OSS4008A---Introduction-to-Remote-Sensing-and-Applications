"""Prepare genuine Canadian healthcare reference locations for the Colab map.

Preferred: Statistics Canada ODHF v2 (2026) official GeoJSON archive.
Backup: Government of Canada ODHF v1 (2020) ArcGIS REST service when v2
cannot be downloaded (e.g., HTTP 403). Backup has older type classifications
and no trustworthy emergency-services indicator. The source is always labelled.
No data is fabricated. Reference facilities are not machine-learning detections.
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
import zipfile
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import shape

DATA_FILE = Path('/content/odhf_facilities.csv') if Path('/content').exists() else Path('odhf_facilities.csv')
META_FILE = DATA_FILE.with_suffix('.source.json')
V2_URL = 'https://www150.statcan.gc.ca/n1/pub/13-26-0001/2020001/zip/odhf_v2_geojson.zip'
# Official Canadian government public map-service layer for ODHF v1, exposed as JSON.
V1_REST_URLS = (
    'https://www.earthquakescanada.nrcan.gc.ca/server2_serveur2/rest/services/StatCan/OpenDatabaseHealthFacilities/MapServer/0/query',
    'https://maps-cartes.services.geo.ca/server2_serveur2/rest/services/StatCan/OpenDatabaseHealthFacilities/MapServer/0/query',
)
HEADERS = {'User-Agent':'Mozilla/5.0 (compatible; OSS4008-Research-Project/1.0)',
           'Accept':'application/json,application/zip,*/*'}
COLUMNS = ['id','name','latitude','longitude','type','address','province','emergency_services','ownership','source_version']


def _normalize(rows, version):
    if not rows:
        raise ValueError('The source returned zero valid healthcare facilities.')
    df = pd.DataFrame.from_records(rows)
    for col in COLUMNS:
        if col not in df:
            df[col] = ''
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.loc[df.latitude.between(41,84) & df.longitude.between(-142,-52)].copy()
    df = df.drop_duplicates(subset=['name','latitude','longitude','type'])
    df['source_version'] = version
    if df.empty:
        raise ValueError('No source coordinates passed the Canadian geographic validation.')
    return df[COLUMNS].reset_index(drop=True)


def _from_v2(session):
    response = session.get(V2_URL, timeout=45, headers=HEADERS)
    response.raise_for_status()
    if len(response.content)>300_000_000 or not zipfile.is_zipfile(BytesIO(response.content)):
        raise ValueError('Invalid or oversized ODHF v2 archive')
    transformer = Transformer.from_crs('EPSG:3347','EPSG:4326',always_xy=True)
    records=[]
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        files=[name for name in archive.namelist()
               if name.lower().endswith(('.geojson','.json')) and not name.startswith('__MACOSX/')]
        if not files:
            raise ValueError('The official v2 ZIP does not contain a GeoJSON file')
        for filename in files:
            content=json.loads(archive.read(filename))
            for feature in content.get('features',[]):
                p=feature.get('properties') or {}
                geom=feature.get('geometry')
                if not geom:
                    continue
                try:
                    g=shape(geom)
                    if g.is_empty:
                        continue
                    pt=g.representative_point()
                    lon,lat=float(pt.x),float(pt.y)
                    if abs(lon)>180 or abs(lat)>90:
                        lon,lat=transformer.transform(lon,lat)
                    if not (41<=lat<=84 and -142<=lon<=-52):
                        continue
                except (TypeError,ValueError):
                    continue
                records.append({
                    'id':str(p.get('id') or ''),'name':str(p.get('name') or 'Unnamed facility'),
                    'latitude':lat,'longitude':lon,'type':str(p.get('type') or 'Unspecified'),
                    'address':str(p.get('address') or ''),
                    'province':str(p.get('prov_terr') or ''),
                    'emergency_services':str(p.get('emergency_services') or ''),
                    'ownership':str(p.get('ownership') or ''),
                })
    return _normalize(records, 'ODHF v2 (2026)')


def _from_v1_service(session, service_url):
    # Object-ID pagination avoids ArcGIS maxRecordCount truncation.
    params={'where':'1=1','returnIdsOnly':'true','f':'json'}
    response=session.get(service_url,params=params,timeout=35,headers=HEADERS)
    response.raise_for_status()
    index=response.json()
    if index.get('error'):
        raise RuntimeError(f"ArcGIS error: {index['error']}")
    ids=index.get('objectIds') or []
    if not ids:
        raise ValueError('ODHF v1 service returned no Object IDs')
    if len(ids)>30000:
        raise ValueError('Unexpectedly large ODHF v1 response')
    rows=[]
    for start in range(0,len(ids),300):
        chunk=ids[start:start+300]
        r=session.get(service_url,params={
            'objectIds':','.join(map(str,chunk)),
            'outFields':'OBJECTID,name_nom,facility_type,latitude,longitude,address,province_en',
            'returnGeometry':'false', 'f':'json'},timeout=35,headers=HEADERS)
        r.raise_for_status()
        result=r.json()
        if result.get('error'):
            raise RuntimeError(f"ArcGIS query error: {result['error']}")
        for feat in result.get('features',[]):
            p=feat.get('attributes') or {}
            rows.append({
                'id':str(p.get('OBJECTID') or ''),
                'name':str(p.get('name_nom') or 'Unnamed facility'),
                'latitude':p.get('latitude'),'longitude':p.get('longitude'),
                'type':str(p.get('facility_type') or 'Unspecified'),
                'address':str(p.get('address') or ''),
                'province':str(p.get('province_en') or ''),
                'emergency_services':'','ownership':''})
    # Do not accept partially paged data as if it were the national source.
    if len(rows) < max(1,int(len(ids)*0.95)):
        raise RuntimeError(f'Incomplete ODHF v1 retrieval: {len(rows)} of {len(ids)} IDs')
    return _normalize(rows, 'ODHF v1 (2020 fallback)')


def build_dataset(force=False, allow_v1_fallback=True, session=None):
    """Download official reference records, use verified cache, or explain failure.

    Returns (DataFrame, metadata dict). Never invents records.
    """
    if not force and DATA_FILE.is_file():
        df=pd.read_csv(DATA_FILE, low_memory=False).fillna('')
        if {'latitude','longitude','name','type'}.issubset(df.columns) and len(df):
            meta=json.loads(META_FILE.read_text()) if META_FILE.exists() else {
                'source_version':str(df.get('source_version',pd.Series(['Existing reference cache'])).iloc[0]),
                'source_url':'Existing local CSV', 'note':'Cached reference records; source age may be unknown.'}
            print(f'Using {len(df):,} cached reference facilities: {meta["source_version"]}')
            return df,meta
    client=session or requests.Session()
    errors=[]
    try:
        df=_from_v2(client)
        meta={'source_version':'ODHF v2 (2026)','source_url':V2_URL,
              'note':'Official Statistics Canada 2026 reference records; not AI predictions.'}
    except (requests.RequestException,ValueError,RuntimeError,zipfile.BadZipFile,KeyError) as e:
        errors.append('2026 source: '+str(e)[:200])
        if not allow_v1_fallback:
            raise RuntimeError('ODHF v2 unavailable; no fake records were created. '+'; '.join(errors)) from e
        df=None
        meta=None
        for endpoint in V1_REST_URLS:
            try:
                df=_from_v1_service(client,endpoint)
                meta={'source_version':'ODHF v1 (2020 fallback)','source_url':endpoint,
                      'note':'Older official Canadian reference data. No reliable emergency-services or fee fields.'}
                break
            except (requests.RequestException,ValueError,RuntimeError,KeyError) as backup_err:
                errors.append('backup: '+str(backup_err)[:200])
        if df is None:
            raise RuntimeError('No official hospital reference source could be downloaded. '
                               'Internet/network restrictions may be blocking both sources. '
                               +'; '.join(errors))
    DATA_FILE.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(DATA_FILE,index=False)
    META_FILE.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'Saved {len(df):,} official reference facilities to {DATA_FILE}.')
    print('Source:',meta['source_version'])
    if errors:
        print('Note: ODHF v2 download failed; using the older official reference layer.')
    return df,meta

if __name__=='__main__':
    build_dataset()
