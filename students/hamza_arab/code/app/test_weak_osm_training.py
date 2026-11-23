import json
from pathlib import Path
import sys
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import from_origin
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from weak_osm_training import box_to_yolo,make_weak_dataset,train_weak_experiment

def make_data(root,n=10,levels=None):
    (root/'review/images').mkdir(parents=True)
    (root/'review/geotiffs').mkdir(parents=True)
    (root/'drafts/osm').mkdir(parents=True)
    rows=[]
    for i in range(n):
        name=f'osm_{i:02d}'
        img=root/f'review/images/{name}.png'
        tif=root/f'review/geotiffs/{name}.tif'
        Image.new('RGB',(100,100),color=(i*3,30,110)).save(img)
        with rasterio.open(tif,'w',driver='GTiff',width=100,height=100,count=3,dtype='uint8',
                             crs='EPSG:3857',transform=from_origin(-8200000,5600000,1,1)) as ds:
            import numpy as np
            ds.write(np.zeros((3,100,100),dtype='uint8'))
        rows.append({'tile_id':name,'site_id':f'site_{i:02d}','image':str(img.relative_to(root)),
              'geotiff':str(tif.relative_to(root)),'imagery_source':'Ontario Open Imagery',
              'imagery_license_url':'https://data.ontario.ca/',
              'review_status':'UNREVIEWED','candidate_class':'known_hospital_site'})
        level = levels[i] if levels else 3
        json_data={'tile_id':name,'approved':False,'candidates':[{'osm_id':f'way/{i+99}',
           'bbox':[18,22,63,72], 'evidence_level':level,'reason':'explicit_hospital_use_tag' if level==3 else 'near_odhf_reference_only'}]}
        (root/f'drafts/osm/{name}.json').write_text(json.dumps(json_data))
    pd.DataFrame(rows).to_csv(root/'review_manifest.csv',index=False)


def test_box_conversion():
    assert box_to_yolo([10,10,30,30],100,100)=='0 0.2000000 0.2000000 0.2000000 0.2000000'
    with pytest.raises(ValueError):box_to_yolo([40,10,30,20],100,100)


def test_weak_dataset_at_least_eight_sites(tmp_path):
    make_data(tmp_path,12)
    r=make_weak_dataset(tmp_path)
    assert r['status']=='WEAK_PSEUDO_DATASET_READY_NOT_GROUND_TRUTH'
    assert r['independent_sites']==12 and r['pseudo_boxes']==12
    assert r['test_sites']>=2 and r['val_sites']>=2 and r['train_sites']>=2
    m=pd.read_csv(tmp_path/'review_manifest.csv',keep_default_na=False)
    assert set(m.review_status)=={'UNREVIEWED'}
    assert not (tmp_path/'review/annotations').exists()
    csv=pd.read_csv(tmp_path/'weak_osm_experiment/dataset/weak_site_split_manifest.csv')
    assert csv.site_id.nunique()==12 and all(csv.num_pseudo_boxes==1)
    assert len({x for x in csv.osm_evidence})>=1
    assert not (tmp_path/'detector/training_report.json').exists()


def test_weak_dataset_does_not_accept_nearby_guess(tmp_path):
    make_data(tmp_path,10,levels=[0]*10)
    r=make_weak_dataset(tmp_path)
    assert r['status']=='INSUFFICIENT_PSEUDO_LABEL_SITES'
    assert r['matched_images']==0 and r['training_started']==False
    assert not (tmp_path/'weak_osm_experiment/dataset/data.yaml').exists()


def test_weak_dataset_does_not_accept_approved_reviews(tmp_path):
    make_data(tmp_path,12)
    p=tmp_path/'review_manifest.csv';df=pd.read_csv(p,keep_default_na=False)
    df.loc[0,'review_status']='APPROVED';df.to_csv(p,index=False)
    r=make_weak_dataset(tmp_path)
    assert r['independent_sites']==11 and r['matched_images']==11
    assert 'site_00' not in pd.read_csv(tmp_path/'weak_osm_experiment/dataset/weak_site_split_manifest.csv').site_id.to_list()


def test_training_cannot_start_if_not_ready(tmp_path):
    make_data(tmp_path,3)
    make_weak_dataset(tmp_path)
    with pytest.raises(ValueError,match='Not enough OSM pseudo labels'):
        train_weak_experiment(tmp_path)


def test_no_status_confusion(tmp_path):
    make_data(tmp_path,9)
    result=make_weak_dataset(tmp_path)
    assert 'NOT_GROUND_TRUTH' in result['status']
    assert not (tmp_path/'hospital_yolo').exists()


def test_unverified_overlay_is_suppressed_without_weights(tmp_path):
    import json
    nb=json.loads((Path(__file__).parent/'04_canada_colab_inline.ipynb').read_text())
    src=''.join(nb['cells'][2]['source'])
    start=src.index('def _add_unverified_weak_overlay():')
    end=src.index('SCREEN_MARGIN_PX=65',start)
    import types
    fake=types.SimpleNamespace(build_dashboard_html=lambda *_a,**_k:"let importedPredictions=[]; if(evaluation.ready)aiStatus.classList.add('ready');")
    scope={'AI_ROOT':tmp_path,'canada_hospital_dashboard':fake}
    exec(compile(src[start:end].replace('print(_add_unverified_weak_overlay())', ''),'weak_overlay','exec'),scope)
    assert 'No experimental weak-model overlay' in scope['_add_unverified_weak_overlay']()
    assert fake.build_dashboard_html()=="let importedPredictions=[]; if(evaluation.ready)aiStatus.classList.add('ready');"


def test_unverified_overlay_provenance_protects_map(tmp_path):
    import json,hashlib,types
    nb=json.loads((Path(__file__).parent/'04_canada_colab_inline.ipynb').read_text())
    src=''.join(nb['cells'][2]['source'])
    start=src.index('def _add_unverified_weak_overlay():');end=src.index('SCREEN_MARGIN_PX=65',start)
    exp=tmp_path/'weak_osm_experiment';(exp/'dataset').mkdir(parents=True)
    weights=exp/'weak_best.pt';weights.write_bytes(b'fixture-placeholder-not-a-trained-model')
    split=exp/'dataset'/'weak_site_split_manifest.csv';split.write_text('site_id,split\ntest,test\n')
    wd=hashlib.sha256(weights.read_bytes()).hexdigest();sd=hashlib.sha256(split.read_bytes()).hexdigest()
    (exp/'weak_training_report.json').write_text(json.dumps({'status':'TRAINED_ON_OSM_PSEUDO_LABELS_NOT_VERIFIED','weights_sha256':wd,'split_sha256':sd}))
    (exp/'experimental_unverified_predictions.geojson').write_text(json.dumps({'type':'FeatureCollection','properties':{'status':'UNVERIFIED_WEAKLY_SUPERVISED_MODEL','weights_sha256':wd},'features':[{'type':'Feature','geometry':{'type':'Point','coordinates':[-75.69,45.42]},'properties':{'confidence':0.56,'method':'osm_weak_supervision_experimental','weights_sha256':wd}}]}))
    text="let importedPredictions=[]; if(evaluation.ready)aiStatus.classList.add('ready');"
    fake=types.SimpleNamespace(build_dashboard_html=lambda *_a,**_k:text)
    scope={'AI_ROOT':tmp_path,'canada_hospital_dashboard':fake}
    exec(compile(src[start:end].replace('print(_add_unverified_weak_overlay())', ''),'weak_overlay','exec'),scope)
    status=scope['_add_unverified_weak_overlay']()
    assert 'UNVERIFIED' in status
    html=fake.build_dashboard_html()
    assert 'let importedPredictions=[{"lat":45.42,"lon":-75.69,"conf":0.56}];' in html
    assert 'Experimental OSM AI' in html
    weights.write_bytes(b'tampered')
    fake.build_dashboard_html=lambda *_a,**_k:text
    status=scope['_add_unverified_weak_overlay']()
    assert 'mismatch' in status and fake.build_dashboard_html()==text
