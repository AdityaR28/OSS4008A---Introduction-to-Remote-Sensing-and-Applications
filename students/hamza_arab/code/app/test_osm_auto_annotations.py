"""Synthetic geospatial contract tests only; no real OSM labels or predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from pyproj import Transformer
from PIL import Image
import osm_auto_annotations as osm
import hospital_review as review


def _create_test_chip(tmp_path,site='A',status='UNREVIEWED'):
    root=tmp_path
    for name in ('review/images','review/geotiffs','review/annotations'):
        (root/name).mkdir(parents=True,exist_ok=True)
    lat,lon=45.42,-75.7
    t=Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True)
    x,y=t.transform(lon,lat)
    extent=(x-130,y-130,x+130,y+130)
    with rasterio.open(root/f'review/geotiffs/{site}.tif','w',driver='GTiff',width=256,height=256,
             count=3,dtype='uint8',crs='EPSG:3857',transform=from_bounds(*extent,256,256)) as ds:
        ds.write(np.ones((3,256,256),dtype=np.uint8)*128)
    Image.new('RGB',(256,256),(150,160,170)).save(root/f'review/images/{site}.png')
    df=pd.DataFrame([{'tile_id':site,'site_id':site,'name':'Test hospital',
      'candidate_class':'known_hospital_site','latitude':lat,'longitude':lon,
      'image':f'review/images/{site}.png','geotiff':f'review/geotiffs/{site}.tif',
      'annotation':f'review/annotations/{site}.json','review_status':status,
      'reviewed_label':'HOSPITAL_PRESENT' if status=='APPROVED' else 'UNKNOWN',
      'reviewer':'Reviewer' if status=='APPROVED' else '', 'source_checked':'YES' if status=='APPROVED' else 'NO'}])
    df.to_csv(root/'review_manifest.csv',index=False)
    return root,lat,lon


def _way(lat,lon,footprint_m=35,btype='yes',id=10,tags=None):
    tx=Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True)
    inv=Transformer.from_crs('EPSG:3857','EPSG:4326',always_xy=True)
    x,y=tx.transform(lon,lat)
    raw=[(x-15,y-15),(x+footprint_m,y-15),(x+footprint_m,y+footprint_m),(x-15,y+footprint_m),(x-15,y-15)]
    return {'type':'way','id':id,'tags':tags if tags is not None else {'building':btype},
            'geometry':[{'lon':p[0],'lat':p[1]} for p in (inv.transform(a,b) for a,b in raw)]}


def test_building_footprint_converts_to_pixels(tmp_path):
    root,lat,lon=_create_test_chip(tmp_path)
    boxes=osm.extract_suggestions([_way(lat,lon,id=100,tags={'building':'hospital','amenity':'hospital'})],lat,lon,root/'review/geotiffs/A.tif')
    assert len(boxes)==1 and boxes[0]['evidence_level']==3
    assert 0<boxes[0]['bbox'][0]<boxes[0]['bbox'][2]<256
    assert 0<boxes[0]['bbox'][1]<boxes[0]['bbox'][3]<256


def test_osm_grounds_only_never_become_roof_labels(tmp_path):
    root,lat,lon=_create_test_chip(tmp_path)
    campus=_way(lat,lon,footprint_m=100,id=222,tags={'amenity':'hospital','name':'Grounds'})
    assert osm.extract_suggestions([campus],lat,lon,root/'review/geotiffs/A.tif')==[]


def test_nearest_nontagged_roof_stays_weak(tmp_path):
    root,lat,lon=_create_test_chip(tmp_path)
    boxes=osm.extract_suggestions([_way(lat,lon)],lat,lon,root/'review/geotiffs/A.tif')
    assert boxes[0]['evidence_level']==0 and boxes[0]['reason']=='near_odhf_reference_only'


def test_batched_drafts_do_not_mutate_verified_manifest(tmp_path):
    root,lat,lon=_create_test_chip(tmp_path)
    original=(root/'review_manifest.csv').read_bytes()
    def fake_fetch(query,**_):
        assert '[out:json]' in query and 'way["building"]' in query
        return {'elements':[_way(lat,lon,id=222,tags={'building':'hospital','amenity':'hospital'})]}
    got=osm.generate_osm_drafts(root,fetcher=fake_fetch,pause_s=0)
    assert got['queried_or_cached']==1 and got['images_with_drafts']==1
    assert got['approved_automatically']==0
    assert (root/'review_manifest.csv').read_bytes()==original
    assert not (root/'review/annotations/A.json').exists()
    cached=osm.generate_osm_drafts(root,fetcher=lambda *_args,**_kwargs:(_ for _ in ()).throw(AssertionError('Must cache')),pause_s=0)
    assert cached['already_cached']==1
    assert len(osm.load_draft_boxes(root,'A'))==1
    assert osm.make_draft_contact_sheet(root).is_file()


def test_draft_does_not_overwrite_approved_human_label(tmp_path):
    root,lat,lon=_create_test_chip(tmp_path,status='APPROVED')
    file=root/'review/annotations/A.json';file.write_text('{"flags":{"verified":true}}')
    before=file.read_bytes()
    got=osm.generate_osm_drafts(root,fetcher=lambda *_a,**_k:{'elements':[]},pause_s=0)
    assert got['queried_or_cached']==0 and file.read_bytes()==before
    assert got['approved_automatically']==0


def test_review_next_returns_draft_and_changes_when_saved(tmp_path):
    root,lat,lon=_create_test_chip(tmp_path)
    draft=root/'drafts/osm';draft.mkdir(parents=True)
    (draft/'A.json').write_text(json.dumps({'tile_id':'A','approved':False,'candidates':[{'bbox':[12,13,90,94]}]}))
    first=review.next_unreviewed_payload(root)
    assert first['ok'] and not first['done'] and first['suggested_boxes']==[[12,13,90,94]]
    assert first['image_src'].startswith('data:image/png;base64,')
    html,js=review.build_labeler_ui('A','A hospital','known_hospital_site',root/'review/images/A.png',first['suggested_boxes'])
    assert 'Use suggested OSM boxes' in html
    assert 'oss4008_hospital_ai.next_review' in js
    result=review.save_annotation_review(root,{'tile_id':'A','reviewer':'Someone','category':'HOSPITAL_PRESENT','confirmed':True,'boxes':[[12,13,90,94]]})
    assert result['ok'] and review.next_unreviewed_payload(root)['done']


def test_overpass_request_bounds_and_rate_limit():
    with pytest.raises(ValueError):osm.query_for_centers([(0,-90)])
    query=osm.query_for_centers([(45.4,-75.7)])
    assert 'out geom' in query and 'relation' in query
    class Reply:
        def __init__(self,status):self.status_code=status;self.headers={};self.content=b'{}'
        def json(self):return {'elements':[]}
        def raise_for_status(self):raise AssertionError('should not call for rate limited')
    class Session:
        def __init__(self):self.calls=0
        def post(self,*args,**kwargs):self.calls+=1;return Reply(429)
    session=Session()
    with pytest.raises(RuntimeError,match='rate limit'):
        osm.fetch_overpass(query,session=session,retries=1,sleep=lambda n:None)
    assert session.calls==1  # Respect explicit Overpass 429; never rotate servers


def test_no_unsupported_automatic_approval(tmp_path):
    root,lat,lon=_create_test_chip(tmp_path)
    class FakeSession:
        calls=0
        def post(self,**kwargs):raise AssertionError('Should not write to API')
    before=(root/'review_manifest.csv').read_bytes()
    assert osm.generate_osm_drafts(root,fetcher=lambda q,**kw:{'elements':[]},pause_s=0)['approved_automatically']==0
    assert (root/'review_manifest.csv').read_bytes()==before


def test_target_100_osm_suggestions_from_distinct_real_like_chips(tmp_path):
    """100 is a target for automatic *drafts*, not 100 verified hospitals."""
    root,lat,lon=_create_test_chip(tmp_path)
    initial=pd.read_csv(root/'review_manifest.csv',keep_default_na=False)
    rows=[]
    # Use one geotiff as a synthetic fixture; each of the 100 manifest sites has
    # its own unique tile ID and image to exercise deduplication/caching logic.
    for i in range(100):
        item=initial.iloc[0].to_dict();item['tile_id']=f'site{i:03}'
        item['name']=f'Test site {i}'
        item['image']='review/images/A.png';item['geotiff']='review/geotiffs/A.tif'
        rows.append(item)
    pd.DataFrame(rows).to_csv(root/'review_manifest.csv',index=False)
    queries=[]
    def fake(q,**kwargs):
        queries.append(q)
        return {'elements':[_way(lat,lon,id=888,tags={'building':'hospital'})]}
    result=osm.generate_osm_drafts(root,limit=100,batch_size=6,fetcher=fake,pause_s=0)
    assert result['requested_max']==100 and result['queried_or_cached']==100
    assert result['images_with_drafts']==100 and result['proposed_boxes']==100
    assert result['approved_automatically']==0 and len(queries)==17
    assert not list((root/'review/annotations').glob('*.json'))


def test_50_image_cap_visible_progress_and_disk_checkpoint(tmp_path, capsys):
    root, lat, lon = _create_test_chip(tmp_path)
    base = pd.read_csv(root/'review_manifest.csv', keep_default_na=False).iloc[0].to_dict()
    all_rows = [{**base,'tile_id':f'site{i:03}', 'site_id':f's{i:03}'} for i in range(55)]
    pd.DataFrame(all_rows).to_csv(root/'review_manifest.csv',index=False)
    calls=[]
    def fake_fetch(query,**kwargs):
        calls.append(query)
        assert '[timeout:20]' in query
        return {'elements':[]}
    report=osm.generate_osm_drafts(root,limit=50,batch_size=4,fetcher=fake_fetch,pause_s=0)
    output=capsys.readouterr().out
    assert 'OSM batch 1/13:' in output
    assert 'OSM batch 13/13 complete:' in output
    assert report['candidate_images']==50
    assert report['queried_or_cached']==50
    assert len(calls)==13
    state=json.loads((root/'drafts/osm_progress.json').read_text())
    assert state['stage']=='FINISHED' and state['images_checked_or_cached']==50
    def no_fetch(*_,**__):
        raise AssertionError('Cached OSM drafts must avoid repeated requests')
    repeated=osm.generate_osm_drafts(root,limit=50,batch_size=4,fetcher=no_fetch,pause_s=0)
    assert repeated['already_cached']==50


def test_short_request_timeout_and_no_default_retry():
    query=osm.query_for_centers([(45.4215,-75.6972)])
    class Response:
        status_code=429
        headers={}
    class Session:
        calls=0
        def post(self,*args,**kwargs):
            self.calls+=1
            assert kwargs['timeout']==(5,20)
            return Response()
    session=Session()
    with pytest.raises(RuntimeError,match='rate limit'):
        osm.fetch_overpass(query,session=session,sleep=lambda _n:None)
    assert session.calls==1


def test_private_coffee_primary_falls_back_after_network_timeout(capsys):
    query = osm.query_for_centers([(45.42,-75.70)])
    class Answer:
        status_code = 200
        content = b'{"elements":[]}'
        def raise_for_status(self): pass
        def json(self): return {'elements': [{'type':'way','id':1}]}
    class Session:
        def __init__(self): self.targets=[]
        def post(self,url,**kwargs):
            self.targets.append(url)
            if len(self.targets)==1: raise osm.requests.exceptions.ReadTimeout('first server stalled')
            return Answer()
    session=Session()
    result=osm.fetch_overpass(query,session=session,retries=0,sleep=lambda _:None,
                              fallback_endpoints=('https://overpass-api.de/api/interpreter',))
    assert result['elements'][0]['id']==1
    assert session.targets == [osm.OVERPASS_ENDPOINT, 'https://overpass-api.de/api/interpreter']
    assert 'trying separate public Overpass mirror' in capsys.readouterr().out

def test_rate_limit_does_not_fail_over_to_secondary_server():
    query=osm.query_for_centers([(45.42,-75.70)])
    class Response:
        status_code=429
        headers={}
    class Session:
        calls=0
        def post(self,*args,**kwargs):self.calls+=1;return Response()
    s=Session()
    with pytest.raises(RuntimeError,match='rate limit'):
        osm.fetch_overpass(query,session=s,retries=0,
           fallback_endpoints=('https://overpass-api.de/api/interpreter',))
    assert s.calls==1

def test_failover_stops_if_both_public_endpoints_timeout():
    query=osm.query_for_centers([(45.42,-75.70)])
    class Session:
        calls=0
        def post(self,*args,**kwargs):
            self.calls+=1
            raise osm.requests.exceptions.ReadTimeout('unavailable')
    s=Session()
    with pytest.raises(RuntimeError,match='2 server'):
        osm.fetch_overpass(query,session=s,retries=0,
            fallback_endpoints=('https://overpass-api.de/api/interpreter',))
    assert s.calls==2
