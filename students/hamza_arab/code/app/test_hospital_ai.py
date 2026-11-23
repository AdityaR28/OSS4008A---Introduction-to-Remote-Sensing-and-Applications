"""Offline contract tests. ALL synthetic imagery is TEST-ONLY, never evidence of detection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import rasterio
import pytest

import hospital_ai as ai


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self.payload, self.content = payload, content
    def raise_for_status(self):
        return None
    def json(self):
        return self.payload


class FakeImagerySession:
    def __init__(self):
        a=np.zeros((640,640,4),dtype=np.uint8)
        a[:,:,0]=np.arange(640,dtype=np.uint8)[None,:]
        a[:,:,1]=np.arange(640,dtype=np.uint8)[:,None]
        a[:,:,2]=100
        a[:,:,3]=255
        b=BytesIO();Image.fromarray(a,'RGBA').save(b,format='PNG')
        self.image=b.getvalue()
        self.calls=0
    def get(self,url,params=None,**kwargs):
        self.calls+=1
        if url.endswith('/exportImage'):
            bb=list(map(float,params['bbox'].split(',')))
            return FakeResponse({'href':'https://ws.geoservices.lrc.gov.on.ca/tmp/imagery.png',
                                 'extent':dict(zip(['xmin','ymin','xmax','ymax'],bb),
                                               spatialReference={'wkid':3857})})
        return FakeResponse(content=self.image)


def test_real_georeference_contract_with_mock_image(tmp_path):
    session=FakeImagerySession()
    result=ai.download_ontario_chip(45.42,-75.7,tmp_path/'test.png',tmp_path/'test.tif',session=session)
    assert session.calls==2
    assert Path(result['image']).is_file() and Path(result['geotiff']).is_file()
    with rasterio.open(result['geotiff']) as src:
        assert str(src.crs)=='EPSG:3857'
        assert src.width==src.height==640
        x,y=src.transform*(320,320)
        lon,lat=ai.transform_coordinates(src.crs,'EPSG:4326',[x],[y])
        assert abs(lat[0]-45.42)<.0001 and abs(lon[0]+75.7)<.0001


def _write_manifest(root, n=4, approved=True):
    image_dir=root/'review'/'images';ann_dir=root/'review'/'annotations'
    image_dir.mkdir(parents=True);ann_dir.mkdir(parents=True)
    rows=[]
    for category in ('HOSPITAL_PRESENT','NO_HOSPITAL'):
        for j in range(n):
            stem=f'test_{category}_{j}'
            img=image_dir/(stem+'.png');Image.new('RGB',(120,80),(150,90+j,40)).save(img)
            ann=ann_dir/(stem+'.json')
            shapes=([{'label':'hospital','shape_type':'rectangle','points':[[20,15],[70,60]]}]
                    if category=='HOSPITAL_PRESENT' else [])
            ann.write_text(json.dumps({'imageWidth':120,'imageHeight':80,
                                       'flags':{'verified':True},'shapes':shapes}))
            rows.append({'tile_id':stem,'site_id':f'{category}_{j}',
                         'image':str(img.relative_to(root)), 'annotation':str(ann.relative_to(root)),
                         'review_status':'APPROVED' if approved else 'UNREVIEWED',
                         'reviewed_label':category,'reviewer':'test_reviewer','source_checked':'YES',
                         'geotiff':'test.tif','name':'Synthetic test','candidate_class':'synthetic',
                         'latitude':45.4,'longitude':-75.7,'imagery_source':'MOCK_NOT_REAL',
                         'imagery_license_url':'TEST_ONLY','source_extent_3857':'[]',
                         'image_width':120,'image_height':80,'ground_width_m':300})
    pd.DataFrame(rows,columns=ai.COLUMN_NAMES).to_csv(root/'review_manifest.csv',index=False)


def test_labels_site_disjoint_and_block_missing_review(tmp_path):
    _write_manifest(tmp_path, approved=False)
    with pytest.raises(ValueError,match='APPROVED'):
        ai.split_verified_dataset(tmp_path)
    df=pd.read_csv(tmp_path/'review_manifest.csv');df['review_status']='APPROVED'
    df.to_csv(tmp_path/'review_manifest.csv',index=False)
    r=ai.split_verified_dataset(tmp_path)
    assert r['site_disjoint'] and r['num_sites']==8
    for split in ('train','val','test'):
        assert r['counts'][split]['positive']>=1
        assert r['counts'][split]['negative']>=1
    split_table=pd.read_csv(tmp_path/'hospital_yolo'/'site_split_manifest.csv')
    assert split_table.groupby('site_id').split.nunique().max()==1
    assert (tmp_path/'hospital_yolo'/'data.yaml').is_file()
    with pytest.raises(RuntimeError,match='BLOCKED'):
        ai.export_geotiff_predictions(tmp_path,tmp_path/'anything.tif',tmp_path/'f.geojson')


def test_labelme_validation_rejects_falsified_negative(tmp_path):
    _write_manifest(tmp_path,n=3)
    df=pd.read_csv(tmp_path/'review_manifest.csv');row=df.iloc[0]
    df.loc[0,'reviewed_label']='NO_HOSPITAL';df.to_csv(tmp_path/'review_manifest.csv',index=False)
    with pytest.raises(ValueError,match='cannot contain a hospital box'):
        ai.split_verified_dataset(tmp_path)


def test_no_covert_mock_geojson_saved(tmp_path):
    source=tmp_path/'chip.tif'
    arr=np.zeros((3,512,512),dtype=np.uint8)
    arr[0]=150;arr[1]=80;arr[2]=50
    with rasterio.open(source,'w',driver='GTiff',width=512,height=512,count=3,dtype='uint8',
                       crs='EPSG:4326',transform=rasterio.transform.from_bounds(-75.8,45.4,-75.7,45.5,512,512)) as dst:
        dst.write(arr)
    class FakeBox:
        xyxy=[SimpleNamespace(tolist=lambda:[10,20,110,120])]
        conf=[0.91]
        cls=[0]
    class FakePredictor:
        def predict(self,**kwargs):return [SimpleNamespace(boxes=[FakeBox()])]
    out=tmp_path/'predictions.geojson'
    geo=ai.export_geotiff_predictions(tmp_path,source,out,predictor=FakePredictor())
    assert not out.exists(), 'Mock predictions must NEVER be saved as real AI output'
    assert len(geo['features'])==1
    x,y=geo['features'][0]['geometry']['coordinates']
    assert -75.8<x<-75.7 and 45.4<y<45.5
    assert geo['features'][0]['properties']['method']=='DETECTOR_TEST_ONLY'


def test_selection_from_known_points_is_not_a_label(tmp_path):
    points=[]
    for i in range(4):
        points.append({'id':str(i),'name':f'Hospital {i}','type':'Hospital',
                       'latitude':45.4+i*.02,'longitude':-75.7})
        points.append({'id':str(i+10),'name':f'Clinic {i}','type':'Clinic',
                       'latitude':45.4+i*.02,'longitude':-75.69})
    csv_path=tmp_path/'reference.csv';pd.DataFrame(points).to_csv(csv_path,index=False)
    sites=ai.select_review_sites(csv_path,center_lat=45.44,center_lon=-75.7,radius_km=20,hospitals=4,controls=4)
    assert len(sites)==8
    assert set(sites['candidate_class'])=={'known_hospital_site','other_healthcare_site_UNVERIFIED'}
    assert 'reviewed_label' not in sites.columns


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))


def test_local_geotiff_alternate_with_valid_georeferencing(tmp_path):
    from rasterio.transform import from_bounds
    from pyproj import Transformer
    import numpy as np
    x,y=Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True).transform(-75.7,45.4)
    src=tmp_path/'licensed_test_ortho.tif'
    side=4096
    # Submeter synthetic raster for *pipeline-only* testing, not training evidence.
    yy,xx=np.indices((side,side))
    im=np.stack([np.full((side,side),180,dtype=np.uint8),
                 (xx%240+10).astype(np.uint8),(yy%240+10).astype(np.uint8)])
    with rasterio.open(src,'w',driver='GTiff',width=side,height=side,count=3,dtype='uint8',
                       crs='EPSG:3857',transform=from_bounds(x-1500,y-1500,x+1500,y+1500,side,side),
                       compress='deflate') as out:
        out.write(im)
    del im,xx,yy
    points=pd.DataFrame([
        {'id':'h1','name':'Test Hospital','type':'Hospital','latitude':45.4,'longitude':-75.7},
        {'id':'c1','name':'Test Clinic','type':'Clinic','latitude':45.4,'longitude':-75.69},
    ])
    ref=tmp_path/'facilities.csv';points.to_csv(ref,index=False)
    project=tmp_path/'project'
    with pytest.raises(ValueError,match='rights confirmation'):
        ai.prepare_review_chips_from_geotiff(ref,src,project,
          imagery_license_url='https://data.ontario.ca/dataset/open-ontario-imagery',
          imagery_source_name='TEST-ONLY', license_confirmed=False)
    df=ai.prepare_review_chips_from_geotiff(ref,src,project,
       imagery_license_url='https://data.ontario.ca/dataset/open-ontario-imagery',
       imagery_source_name='TEST-ONLY',license_confirmed=True,
       center_lat=45.4,center_lon=-75.7,radius_km=3,hospitals=1,controls=1)
    assert len(df)==2
    assert all(df.review_status=='UNREVIEWED')
    assert all((project/Path(x)).is_file() for x in df.geotiff)


def test_train_and_evaluate_contract_uses_site_split_and_real_report_gate(tmp_path, monkeypatch):
    """MOCK trainer tests plumbing only; no real weights/accuracy are produced."""
    import sys
    import types
    _write_manifest(tmp_path,n=3)
    ai.split_verified_dataset(tmp_path)
    class TestYOLO:
        def __init__(self,path):
            self.path=path
            self.names={0:'hospital'}
        def train(self, **kwargs):
            run=tmp_path/'runs'/'hospital_detector'
            (run/'weights').mkdir(parents=True)
            (run/'weights'/'best.pt').write_bytes(b'MOCK MODEL - NEVER DISTRIBUTE')
            return SimpleNamespace(save_dir=str(run))
        def val(self,**kwargs):
            assert kwargs.get('split')=='test'
            return SimpleNamespace(box=SimpleNamespace(mp=.5,mr=.4,map50=.3,map=.2))
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=TestYOLO))
    report=ai.train_and_evaluate(tmp_path,epochs=1,batch=1)
    assert report['status']=='TRAINED_AND_EVALUATED'
    assert report['test_metrics']=={'precision':.5,'recall':.4,'mAP50':.3,'mAP50_95':.2}
    assert ai.sha256(tmp_path/'detector'/'hospital_best.pt')==report['weights_sha256']
    # These values are a **mock** fixture and must never be reported as model accuracy.
