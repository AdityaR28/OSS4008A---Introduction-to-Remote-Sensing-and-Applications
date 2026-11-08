"""Fetch and normalize Statistics Canada's ODHF v2 GeoJSON for mapping.

Uses public data published September 24, 2026. Intended as a location reference only.
"""
from pathlib import Path
import json
from io import BytesIO
import zipfile
import requests
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

SOURCE='https://www150.statcan.gc.ca/n1/pub/13-26-0001/2020001/zip/odhf_v2_geojson.zip'
TARGET=Path('/content/odhf_hospitals.csv') if Path('/content').exists() else Path('odhf_hospitals.csv')

def is_hospital(properties):
    # v2 field `type` documented by Statistics Canada.
    kind=str(properties.get('type') or '').strip().casefold()
    return kind in ('hospital', 'hospitals')

def build_dataset():
    response=requests.get(SOURCE,timeout=100,headers={'User-Agent':'OSS4008-Canada-Hospitals-CourseProject/1.0'})
    response.raise_for_status()
    if len(response.content)>300_000_000:
        raise ValueError('The dataset archive unexpectedly exceeded 300 MB.')
    records=[]
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        candidates=[n for n in archive.namelist() if n.lower().endswith(('.geojson','.json')) and not n.startswith('__MACOSX/')]
        if not candidates:
            raise ValueError('No GeoJSON file found in the ODHF v2 archive.')
        print('GeoJSON files:', candidates)
        for name in candidates:
            # GeoJSON content may contain point or polygon geometries.
            features=json.loads(archive.read(name)).get('features') or []
            for feature in features:
                properties=feature.get('properties') or {}
                if not is_hospital(properties):
                    continue
                geometry=feature.get('geometry')
                if not geometry:
                    continue
                geom=shape(geometry)
                if geom.is_empty:
                    continue
                point=geom.representative_point()
                x,y=point.x,point.y
                if abs(x)>180 or abs(y)>90:
                    # National published projection: NAD83 / Statistics Canada Lambert (EPSG:3347).
                    transformer=Transformer.from_crs('EPSG:3347','EPSG:4326',always_xy=True)
                    x,y=transformer.transform(x,y)
                if not (41<=y<=84 and -142<=x<=-52):
                    continue
                records.append({'name':str(properties.get('name') or 'Unnamed hospital'),
                    'latitude':y,'longitude':x,'type':str(properties.get('type') or ''),
                    'address':str(properties.get('address') or ''),
                    'province':str(properties.get('prov_terr') or ''),
                    'id':str(properties.get('id') or ''),
                    'geometry_type':str(geometry.get('type') or '')})
    if not records:
        raise ValueError('The archive contained no recognizable hospital features. Inspect the published field schema; no fake reference locations were generated.')
    df=pd.DataFrame(records).drop_duplicates(subset=['id','name','latitude','longitude'])
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(TARGET,index=False)
    print('Saved',len(df),'hospital reference records to',TARGET)
    print(df[['name','province','geometry_type']].head(8).to_string(index=False))
    return df

if __name__=='__main__':
    build_dataset()
