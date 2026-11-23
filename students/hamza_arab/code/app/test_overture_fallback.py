import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely import wkb
import rasterio
from rasterio.transform import from_bounds
import overture_weak_labels as ov
import weak_osm_training as weak


def fixture(tmp_path, n=8):
    lat, lon = 45.42, -75.70
    to_merc=Transformer.from_crs(4326,3857,always_xy=True)
    x,y=to_merc.transform(lon,lat)
    image=tmp_path/'review/images/A.png';image.parent.mkdir(parents=True)
    Image.new('RGB',(256,256),'white').save(image)
    tif=tmp_path/'review/geotiffs/A.tif';tif.parent.mkdir(parents=True)
    with rasterio.open(tif,'w',driver='GTiff',height=256,width=256,count=3,dtype='uint8',crs='EPSG:3857',transform=from_bounds(x-130,y-130,x+130,y+130,256,256)) as ds:
        ds.write(np.full((3,256,256),190,dtype='uint8'))
    data=[]
    for i in range(n):
        data.append({'tile_id':f'tile{i}','site_id':f'site{i}','image':'review/images/A.png','geotiff':'review/geotiffs/A.tif','latitude':lat,'longitude':lon,'candidate_class':'known_hospital_site','review_status':'UNREVIEWED','imagery_source':'Ontario','imagery_license_url':'https://data.ontario.ca/dataset/open-ontario-imagery'})
    pd.DataFrame(data).to_csv(tmp_path/'review_manifest.csv',index=False)
    inv=Transformer.from_crs(3857,4326,always_xy=True)
    xy=[(x-25,y-25),(x+25,y-25),(x+25,y+25),(x-25,y+25),(x-25,y-25)]
    geom=Polygon([inv.transform(*p) for p in xy])
    return lat,lon,{'id':'a-building','class':'hospital','geometry':wkb.dumps(geom)}, {'id':'other-building','class':None,'geometry':wkb.dumps(geom)}


def test_overture_footprint_pixel_boxes_and_provenance(tmp_path):
    lat,lon,structured,generic=fixture(tmp_path)
    boxes=ov.roof_candidates([structured],latitude=lat,longitude=lon,geotiff=tmp_path/'review/geotiffs/A.tif')
    assert len(boxes)==1 and boxes[0]['evidence_level']==2
    assert boxes[0]['reason']=='overture_building_class_hospital'
    assert 0<boxes[0]['bbox'][0]<boxes[0]['bbox'][2]<256
    assert ov.roof_candidates([generic],latitude=lat,longitude=lon,geotiff=tmp_path/'review/geotiffs/A.tif')[0]['evidence_level']==1
    assert not ov.roof_candidates([{**generic,'class':'residential'}],latitude=lat,longitude=lon,geotiff=tmp_path/'review/geotiffs/A.tif')


def test_overture_fallback_yields_8_site_disjoint_weak_dataset_no_approvals(tmp_path):
    lat,lon,structured,_=fixture(tmp_path)
    original=(tmp_path/'review_manifest.csv').read_bytes()
    requests=[]
    def provider(bbox):
        requests.append(bbox)
        assert bbox[0]<lon<bbox[2] and bbox[1]<lat<bbox[3]
        return [structured]
    output=ov.generate_overture_drafts(tmp_path,limit=50,target_sites=8,provider=provider)
    assert output['candidate_sites_with_proposals']==8 and output['requests']==8
    assert (tmp_path/'review_manifest.csv').read_bytes()==original
    status=weak.make_weak_dataset(tmp_path,min_evidence=1,minimum_sites=8)
    assert status['status']=='WEAK_PSEUDO_DATASET_READY_NOT_GROUND_TRUTH'
    assert status['train_sites']==4 and status['val_sites']==2 and status['test_sites']==2
    assert not list((tmp_path/'review').rglob('*.json'))
    assert all((tmp_path/'drafts/overture'/f'tile{i}.json').exists() for i in range(8))
    # Cached entries are sufficient; no more requests when the user restarts.
    another=ov.generate_overture_drafts(tmp_path,limit=50,target_sites=8,
                                        provider=lambda _:(_ for _ in ()).throw(AssertionError('Must be cached')))
    assert another['requests']==0 and another['candidate_sites_with_proposals']==8


def test_overture_no_remote_result_fails_fast_without_inventing_matches(tmp_path):
    lat,lon,structured,_=fixture(tmp_path)
    def unavailable(_):raise ConnectionError('STAC unavailable')
    out=ov.generate_overture_drafts(tmp_path,limit=50,target_sites=8,provider=unavailable)
    assert out['candidate_sites_with_proposals']==0 and out['requests']==0 and len(out['errors'])==1
    assert not list((tmp_path/'drafts/overture').glob('*.json'))
    status=weak.make_weak_dataset(tmp_path,min_evidence=1,minimum_sites=8)
    assert status['status']=='INSUFFICIENT_PSEUDO_LABEL_SITES' and status['training_started'] is False


def test_preserves_osm_drafts_and_human_labels(tmp_path):
    lat,lon,structured,_=fixture(tmp_path)
    human=tmp_path/'review/annotations/any.json';human.parent.mkdir(parents=True)
    human.write_text('{"flags":{"verified":true}}')
    draft=tmp_path/'drafts/osm/tile0.json';draft.parent.mkdir(parents=True)
    draft.write_text(json.dumps({'tile_id':'tile0','approved':False,'candidates':[{'evidence_level':2,'bbox':[20,20,80,80]}]}))
    before=draft.read_bytes()
    out=ov.generate_overture_drafts(tmp_path,limit=8,target_sites=8,provider=lambda _: [structured])
    assert out['candidate_sites_with_proposals']==8
    assert draft.read_bytes()==before and human.read_text()=='{"flags":{"verified":true}}'
