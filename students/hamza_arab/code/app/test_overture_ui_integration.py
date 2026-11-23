"""Offline behavior-contract tests; no real network or model claimed."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pandas as pd

MAP = Path(__file__).parent/'04_canada_colab_inline.ipynb'
LAB = Path(__file__).parent/'08_hospital_ai_pipeline.ipynb'


def test_map_ui_layout_and_source_byte_identical():
    previous=json.load(open('/mnt/data/04_canada_colab_inline.ipynb'))
    new=json.load(open(MAP))
    def source(doc):
        s=''.join(doc['cells'][3]['source'])
        return next(n.value.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='APP_SOURCE' for t in n.targets))
    assert source(previous)==source(new)
    assert 'SCREEN_MARGIN_PX=65' in ''.join(new['cells'][3]['source'])
    assert 'def _run_hospital_detection' in ''.join(new['cells'][3]['source'])
    assert 'infer_weak_in_selected_circle' in ''.join(new['cells'][3]['source'])


def test_experimental_checkpoint_wires_run_ai_without_fake_model(tmp_path):
    import app
    exp=tmp_path/'weak_osm_experiment'; (exp/'dataset').mkdir(parents=True)
    (exp/'weak_best.pt').write_bytes(b'FAKE_WEIGHTS_FOR_RENDER_TEST_ONLY')
    (exp/'dataset/weak_site_split_manifest.csv').write_text('site_id,split\na,test\n')
    dig=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (exp/'weak_training_report.json').write_text(json.dumps({
        'status':'TRAINED_ON_OSM_PSEUDO_LABELS_NOT_VERIFIED',
        'weights_sha256':dig(exp/'weak_best.pt'),
        'split_sha256':dig(exp/'dataset/weak_site_split_manifest.csv')
    }))
    nb=json.load(open(MAP));s=''.join(nb['cells'][3]['source']);
    start=s.index('def _add_unverified_weak_overlay():')
    end=s.index('print(_add_unverified_weak_overlay())',start)
    patch=s[start:end]
    namespace={'AI_ROOT':tmp_path,'canada_hospital_dashboard':app}
    original=app.build_dashboard_html
    try:
        exec(patch,namespace)
        result=namespace['_add_unverified_weak_overlay']()
        assert '0 UNVERIFIED' in result
        dataframe=pd.DataFrame([{'name':'Hospital','latitude':45.42,'longitude':-75.70,
            'type':'Hospital'}])
        html=app.build_dashboard_html(dataframe)
        assert "if(!evaluation.ready){addChat('Experimental image model available." in html
        assert 'Running EXPERIMENTAL model inference' in html
        assert "importedPredictions=points;predictionOrigin='experimental'" in html
        assert 'height:100%' in html
        assert 'Experimental open-map AI — UNVERIFIED' in html
        assert 'No evaluated hospital AI predictions yet' not in html
    finally:
        app.build_dashboard_html=original


def test_no_weak_checkpoint_no_guard_bypass(tmp_path):
    import app
    nb=json.load(open(MAP));s=''.join(nb['cells'][3]['source']);
    start=s.index('def _add_unverified_weak_overlay():')
    end=s.index('print(_add_unverified_weak_overlay())',start)
    namespace={'AI_ROOT':tmp_path,'canada_hospital_dashboard':app}
    original=app.build_dashboard_html
    try:
        exec(s[start:end],namespace)
        result=namespace['_add_unverified_weak_overlay']()
        assert 'No experimental weak-model checkpoint' in result
        assert app.build_dashboard_html is original
    finally:
        app.build_dashboard_html=original


def test_lab_separate_notebook_never_modifies_verified_annotations_or_uses_overpass_by_default():
    nb=json.load(open(LAB))
    code=''.join(nb['cells'][13]['source'])
    assert 'RUN_PUBLIC_OVERPASS = False' in code
    assert 'overture.generate_overture_drafts(ROOT' in code
    assert 'WEAK_TRAIN_EPOCHS = 12' in code
    assert 'RUN_HANDS_FREE_AI = True' in code
    assert 'No training attempted' in code
    assert 'WEAK_OSM_SOURCE' in code
    assert 'OVERTURE_WEAK_SOURCE' in ''.join(nb['cells'][5]['source'])
    assert 'overturemaps>=2.1,<3' in ''.join(nb['cells'][2]['source'])
