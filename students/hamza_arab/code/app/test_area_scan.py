from pathlib import Path
import hashlib, json, sys, tempfile
from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
import importlib.util
import ast, nbformat

BASE=Path(__file__).parent
sys.path.insert(0,str(BASE))
import hospital_area_scan as scan

class FakeDetector:
    def __init__(self):self.calls=0
    def predict(self,*,source,conf,verbose):
        self.calls+=1
        b=SimpleNamespace(cls=[0],xyxy=np.array([[180.,190.,400.,400.]]),conf=[.81])
        return [SimpleNamespace(boxes=[b])]

def create_project(root):
    exp=root/'weak_osm_experiment'
    (exp/'dataset').mkdir(parents=True)
    (exp/'weak_best.pt').write_bytes(b'experimental weights dummy for geospatial test only')
    (exp/'dataset'/'weak_site_split_manifest.csv').write_text('site_id,split\nt1,train\n')
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (exp/'weak_training_report.json').write_text(json.dumps({
         'status':'TRAINED_ON_OSM_PSEUDO_LABELS_NOT_VERIFIED',
         'weights_sha256':h(exp/'weak_best.pt'),'split_sha256':h(exp/'dataset'/'weak_site_split_manifest.csv')}))

def provider(lat,lon,png,tif):
    from pyproj import Transformer
    lon_m,lat_m=Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True).transform(lon,lat)
    from math import cos,radians
    half=160/max(cos(radians(lat)),.01)
    t=from_bounds(lon_m-half,lat_m-half,lon_m+half,lat_m+half,width=640,height=640)
    pix=np.full((3,640,640),180,dtype='uint8')
    with rasterio.open(tif,'w',driver='GTiff',width=640,height=640,count=3,crs='EPSG:3857',transform=t,dtype='uint8') as ds:ds.write(pix)
    Image.fromarray(pix.transpose((1,2,0))).save(png)


def test_georeferenced_rectangle_and_cache(tmp_path):
    create_project(tmp_path)
    model=FakeDetector()
    site={'lat':45.42,'lon':-75.69,'name':'Test reference (not verified hospital)'}
    x=scan.scan_sites(tmp_path,[site],model=model,downloader=provider)
    assert x['ok'] and x['new_sites']==1 and x['sites_processed']==1 and len(x['features'])==1
    f=x['features'][0]
    assert f['geometry']['type']=='Polygon'
    poly=f['geometry']['coordinates'][0]
    assert len(poly)==5 and poly[0]==poly[-1]
    lat,lon=scan.polygon_center(f)
    assert scan.great_circle_m(lat,lon,45.42,-75.69)<90
    assert f['properties']['method']=='weak_yolo_geo_bbox'
    assert 'UNVERIFIED' in f['properties']['warning']
    y=scan.scan_sites(tmp_path,[site],model=model,downloader=lambda *args: (_ for _ in ()).throw(AssertionError('redownloaded')))
    assert y['cached_sites']==1 and y['new_sites']==0 and model.calls==1


def test_batch_size_and_untrained_guard(tmp_path):
    create_project(tmp_path)
    model=FakeDetector()
    with pytest.raises(ValueError,match='1–4'):
        scan.scan_sites(tmp_path,[{'lat':45.42,'lon':-75.69}]*5,model=model,downloader=provider)
    (tmp_path/'weak_osm_experiment'/'weak_best.pt').write_bytes(b'wrong weights')
    with pytest.raises(RuntimeError,match='fingerprint'):
        scan.scan_sites(tmp_path,[{'lat':45.42,'lon':-75.69}],model=model,downloader=provider)


def test_missing_imagery_is_explicit_failure(tmp_path):
    create_project(tmp_path)
    res=scan.scan_sites(tmp_path,[{'lat':45.42,'lon':-75.69,'name':'Test'}],model=FakeDetector(),
                        downloader=lambda *a: (_ for _ in ()).throw(RuntimeError('Imagery service unavailable')))
    assert res['ok'] and len(res['features'])==0 and len(res['errors'])==1
    assert 'Imagery service unavailable' in res['errors'][0]['error']


def test_app_contains_city_scan_and_polygons():
    spec=importlib.util.spec_from_file_location('hospital_dashboard_area_scan',str(BASE/'app.py'))
    app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
    import pandas as pd
    df=pd.DataFrame([{'name':'Ottawa hospital demo','latitude':45.4215,'longitude':-75.6972,'type':'Hospital',
                      'province':'ON','address':'','emergency_services':'','ownership':''}])
    ui=app.build_dashboard_html(df)
    assert 'referenceSiteTargets' in ui
    assert 'L.polygon(p.ring' in ui
    assert "mode:'reference_sites'" in ui
    assert 'experimentalReady=false' in ui
    assert 'SCREEN_MARGIN_PX' not in ui or '65' not in ui # CSS is separate from iframe sizing


def test_inline_notebook_sources_and_registration():
    nb=nbformat.read(BASE.parent/'notebooks'/'01_final_hospital_map.ipynb',as_version=4)
    source=nb.cells[-1].source
    tree=ast.parse(source)
    vals={t.id:n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
          for t in n.targets if isinstance(t,ast.Name) and t.id in {'APP_SOURCE','AREA_SCAN_SOURCE','WEAK_ROOF_SOURCE'}}
    assert vals['APP_SOURCE']==(BASE/'app.py').read_text()
    assert vals['AREA_SCAN_SOURCE']==(BASE/'hospital_area_scan.py').read_text()
    assert vals['WEAK_ROOF_SOURCE']==(BASE/'weak_osm_training.py').read_text()
    assert "request.get('mode')=='reference_sites'" in source
    assert 'SCREEN_MARGIN_PX=65' in source
    assert 'colab_output.register_callback' in source
