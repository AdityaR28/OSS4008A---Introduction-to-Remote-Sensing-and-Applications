"""End-to-end tests use SIMULATED ARTIFACTS only; no model output is real."""
from pathlib import Path
import json
import hashlib
import importlib.util
from types import SimpleNamespace
import pandas as pd
import pytest
import nbformat
from PIL import Image
import numpy as np
import rasterio

import hospital_ai as ai
import hospital_review as reviewer
import app

BASE_TEMPLATE_SHA = 'c8b2603605b3923c55ee9453825fd5783f491503ed762dc0b8c859c1aa145c6c'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fake_evaluated_artifacts(root):
    (root/'detector').mkdir(parents=True)
    (root/'hospital_yolo').mkdir()
    (root/'predictions').mkdir()
    weights=root/'detector'/'hospital_best.pt';weights.write_bytes(b'FAKE TEST WEIGHTS: NOT A MODEL')
    split=root/'hospital_yolo'/'site_split_manifest.csv';pd.DataFrame([{'tile_id':'t','site_id':'s','split':'test'}]).to_csv(split,index=False)
    original=root/'review_manifest.csv';pd.DataFrame([{'tile_id':'t','site_id':'s','review_status':'APPROVED','image':'x','geotiff':'x'}]).to_csv(original,index=False)
    report={'status':'TRAINED_AND_EVALUATED','weights_sha256':_sha(weights),
            'split_manifest_sha256':_sha(split),'source_manifest_sha256':_sha(original)}
    (root/'detector'/'training_report.json').write_text(json.dumps(report))
    return report


def test_approved_ui_template_and_screen_height_unchanged():
    assert hashlib.sha256(app.UI_TEMPLATE.encode('utf-8')).hexdigest()==BASE_TEMPLATE_SHA
    assert 'def launch_colab_dashboard(screen_margin_px=65)' in Path(app.__file__).read_text()
    assert 'height:calc(100dvh - {screen_margin_px}px)' in Path(app.__file__).read_text()
    assert '◯ Draw circle' in app.UI_TEMPLATE
    assert 'id="show-pred"' in app.UI_TEMPLATE and 'id="run-ai"' in app.UI_TEMPLATE


def test_model_predictions_are_absent_until_evaluated(tmp_path):
    points,why=app.load_verified_model_predictions(tmp_path)
    assert points==[] and 'No evaluated' in why
    frame=pd.DataFrame([{'name':'Synthetic reference','latitude':45.43,'longitude':-75.7,'type':'Hospital'}])
    page=app.build_dashboard_html(frame,{'label':'SYNTHETIC TEST'})
    assert 'let importedPredictions=[];' in page
    assert 'No trained model or generative AI chatbot' in page


def test_auto_prediction_bridge_uses_only_verified_model_hash(tmp_path):
    r=_fake_evaluated_artifacts(tmp_path)
    geo={'type':'FeatureCollection','features':[{'type':'Feature',
        'geometry':{'type':'Point','coordinates':[-75.64,45.43]},
        'properties':{'class':'hospital','confidence':.85,
                      'method':'trained_YOLO_hospital_building_detector',
                      'weights_sha256':r['weights_sha256']}}]}
    path=tmp_path/'predictions'/'trained_hospital_test_predictions.geojson'
    path.write_text(json.dumps(geo))
    points,why=app.load_verified_model_predictions(tmp_path)
    assert points==[{'lat':45.43,'lon':-75.64,'conf':.85}]
    frame=pd.DataFrame([{'name':'Known facility','latitude':45.43,'longitude':-75.7,'type':'Hospital'}])
    page=app.build_dashboard_html(frame,{'label':'TEST'},points)
    assert 'let importedPredictions=[{"lat":45.43,"lon":-75.64,"conf":0.85}];' in page
    assert 'Known facility' in page and 'Map assistant' in page
    geo['features'][0]['properties']['method']='DETECTOR_TEST_ONLY';path.write_text(json.dumps(geo))
    blocked,reason=app.load_verified_model_predictions(tmp_path)
    assert blocked==[] and 'suppressed' in reason
    geo['features'][0]['properties']['method']='trained_YOLO_hospital_building_detector';path.write_text(json.dumps(geo))
    (tmp_path/'detector'/'hospital_best.pt').write_bytes(b'WEIGHTS ALTERED')
    blocked,reason=app.load_verified_model_predictions(tmp_path)
    assert blocked==[] and 'changed' in reason


def test_skip_ambiguous_annotation_does_not_become_negative(tmp_path):
    (tmp_path/'review'/'images').mkdir(parents=True)
    image=tmp_path/'review'/'images'/'a.png';Image.new('RGB',(90,90),(255,255,255)).save(image)
    pd.DataFrame([{'tile_id':'a','site_id':'a','name':'AMBIGUOUS','candidate_class':'other_healthcare_site_UNVERIFIED',
                   'image':'review/images/a.png','annotation':'review/annotations/a.json','review_status':'UNREVIEWED',
                   'reviewed_label':'UNKNOWN','reviewer':'','source_checked':'NO'}]).to_csv(tmp_path/'review_manifest.csv',index=False)
    result=reviewer.save_annotation_review(tmp_path,{'tile_id':'a','category':'SKIP_UNCERTAIN','reviewer':'Test','confirmed':False,'boxes':[]})
    assert result['ok'] and result['remaining_unreviewed']==0
    assert not (tmp_path/'review'/'annotations'/'a.json').exists()
    manifest=pd.read_csv(tmp_path/'review_manifest.csv')
    assert manifest.iloc[0]['review_status']=='REJECTED'
    status=ai.review_workspace_status(tmp_path)
    assert status['records']==1 and status['images_found']==1
    assert status['geotiffs_found']==0


def test_batch_map_export_aggregates_heldout_with_model_provenance(tmp_path,monkeypatch):
    report=_fake_evaluated_artifacts(tmp_path)
    (tmp_path/'review'/'geotiffs').mkdir(parents=True)
    tif=tmp_path/'review'/'geotiffs'/'scene.tif'
    with rasterio.open(tif,'w',driver='GTiff',width=256,height=256,count=3,
                       dtype='uint8',crs='EPSG:4326',
                       transform=rasterio.transform.from_bounds(-75.7,45.4,-75.6,45.5,256,256)) as ds:
        ds.write(np.ones((3,256,256),dtype='uint8')*100)
    pd.DataFrame([{'tile_id':'t','site_id':'s','geotiff':'review/geotiffs/scene.tif',
                   'review_status':'APPROVED'}]).to_csv(tmp_path/'review_manifest.csv',index=False)
    report['source_manifest_sha256']=_sha(tmp_path/'review_manifest.csv')
    (tmp_path/'detector'/'training_report.json').write_text(json.dumps(report))
    def mock_model_predict(root,source,out,confidence=.35):
        return {'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'Point',
            'coordinates':[-75.66,45.45]},'properties':{'class':'hospital','confidence':.8,
            'method':'trained_YOLO_hospital_building_detector','weights_sha256':report['weights_sha256']}}]}
    monkeypatch.setattr(ai,'export_geotiff_predictions',mock_model_predict)
    output=tmp_path/'predictions'/'trained_hospital_test_predictions.geojson'
    ans=ai.export_all_heldout_predictions(tmp_path,output)
    assert ans['features']==1 and ans['test_sites']==1 and output.is_file()
    points,reason=app.load_verified_model_predictions(tmp_path)
    assert len(points)==1 and points[0]['conf']==.8
    assert not output.with_suffix('.temporary.geojson').exists()
    # NOTE: This test injected a synthetic prediction; this temp output must never
    # be used for user-facing map results. The real method does not accept mocks.


def test_every_notebook_cell_compiles_and_embedded_modules_match():
    root=Path(__file__).parent
    notebooks=root if (root/'04_canada_colab_inline.ipynb').exists() else root.parent/'notebooks'
    for filename in ['04_canada_colab_inline.ipynb','08_hospital_ai_pipeline.ipynb']:
        nb=nbformat.read(notebooks/filename,as_version=4)
        nbformat.validate(nb)
        for i,c in enumerate(nb.cells):
            if c.cell_type=='code' and not c.source.lstrip().startswith('%pip'):
                compile(c.source,f'{filename}:cell-{i}','exec')
        assert all(c.execution_count is None and c.outputs==[] for c in nb.cells if c.cell_type=='code')
    nb=nbformat.read(notebooks/'04_canada_colab_inline.ipynb',as_version=4)
    import ast
    appassign=ast.parse(nb.cells[-3].source).body[0]
    assert isinstance(appassign,ast.Assign) and ast.literal_eval(appassign.value)==(root/'app.py').read_text()
    code_cell=nb.cells[7].source
    tree=ast.parse(code_cell)
    values={t.targets[0].id:ast.literal_eval(t.value) for t in tree.body if isinstance(t,ast.Assign) and isinstance(t.targets[0],ast.Name) and t.targets[0].id in ('HOSPITAL_AI_SOURCE','HOSPITAL_REVIEW_SOURCE')}
    assert values['HOSPITAL_AI_SOURCE']==(root/'hospital_ai.py').read_text()
    assert values['HOSPITAL_REVIEW_SOURCE']==(root/'hospital_review.py').read_text()
    assert 'SCREEN_MARGIN_PX = 65' in nb.cells[-1].source


def test_small_circle_area_inference_is_guarded_and_georeferenced(tmp_path, monkeypatch):
    """Mock remote and YOLO calls; verify behavior but NEVER claim accuracy."""
    report=_fake_evaluated_artifacts(tmp_path)
    with pytest.raises(ValueError,match='20–250'):
        ai.infer_in_selected_circle(tmp_path,45.43,-75.70,25000)
    def mocked_download(lat,lon,image,tif,**kwargs):
        Image.new('RGB',(640,640),(100,120,90)).save(image)
        with rasterio.open(tif,'w',driver='GTiff',width=640,height=640,count=3,dtype='uint8',
                           crs='EPSG:4326',transform=rasterio.transform.from_bounds(-75.72,45.41,-75.68,45.45,640,640)) as ds:
            ds.write(np.ones((3,640,640),dtype='uint8')*100)
        return {'image':str(image),'geotiff':str(tif)}
    def mocked_model(root,geotiff,output,confidence=.35):
        return {'type':'FeatureCollection','features':[{'type':'Feature',
            'geometry':{'type':'Point','coordinates':[-75.7,45.43]},
            'properties':{'method':'trained_YOLO_hospital_building_detector',
                          'class':'hospital','confidence':.77,
                          'weights_sha256':report['weights_sha256']}}]}
    monkeypatch.setattr(ai,'download_ontario_chip',mocked_download)
    monkeypatch.setattr(ai,'export_geotiff_predictions',mocked_model)
    result=ai.infer_in_selected_circle(tmp_path,45.43,-75.70,100)
    assert result['ok'] and result['count']==1 and result['image_chips']==1
    saved=Path(result['saved_to'])
    assert saved.is_file()
    assert json.loads(saved.read_text())['features'][0]['properties']['method']=='trained_YOLO_hospital_building_detector'
    # No real model inference was performed: this is a simulated contract test.


def test_browser_approved_html_unmodified_and_download_safe(tmp_path):
    assert app.UI_TEMPLATE.count('<section class="filters">')==1
    assert 'map assistant' in app.UI_TEMPLATE.casefold()
    assert "SCREEN_MARGIN_PX = 65" in nbformat.read((Path(__file__).parent/'04_canada_colab_inline.ipynb') if (Path(__file__).parent/'04_canada_colab_inline.ipynb').exists() else Path(__file__).parent.parent/'notebooks'/'04_canada_colab_inline.ipynb',as_version=4).cells[-1].source


def test_resume_missing_imagery_without_overwriting_human_approval(tmp_path):
    from test_hospital_ai import FakeImagerySession
    csv=tmp_path/'odhf.csv'
    pd.DataFrame([{'id':'h','name':'Hospital','type':'Hospital','latitude':45.4,'longitude':-75.7},
                  {'id':'c','name':'Clinic','type':'Clinic','latitude':45.41,'longitude':-75.7}]).to_csv(csv,index=False)
    project=tmp_path/'pilot'
    session=FakeImagerySession()
    first=ai.prepare_review_chips(csv,project,center_lat=45.4,center_lon=-75.7,radius_km=5,
                                   hospitals=1,controls=1,session=session)
    assert len(first)==2
    first.loc[0,'review_status']='APPROVED'
    first.loc[0,'reviewed_label']='HOSPITAL_PRESENT'
    first.loc[0,'reviewer']='Test reviewer'
    first.loc[0,'source_checked']='YES'
    first.to_csv(project/'review_manifest.csv',index=False)
    existing_image=(project/first.iloc[0]['image']).read_bytes()
    missing=(project/first.iloc[1]['geotiff']);missing.unlink()
    resumed=ai.prepare_review_chips(csv,project,center_lat=45.4,center_lon=-75.7,radius_km=5,
                                     hospitals=1,controls=1,session=session)
    assert len(resumed)==2 and missing.exists()
    assert resumed.iloc[0]['review_status']=='APPROVED'
    assert (project/resumed.iloc[0]['image']).read_bytes()==existing_image
    assert resumed.iloc[1]['review_status']=='UNREVIEWED'
