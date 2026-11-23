"""Offline contracts; all trained artifacts are synthetic and NOT actual medical maps."""
from pathlib import Path
import json,hashlib
import pandas as pd
import app

def _hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _artifacts(root):
    (root/'detector').mkdir(parents=True)
    (root/'hospital_yolo').mkdir()
    weight=root/'detector'/'hospital_best.pt';weight.write_bytes(b'DUMMY_UNIT_TEST_ONLY')
    source=root/'review_manifest.csv';source.write_text('id,review_status\nt1,APPROVED\n')
    split=root/'hospital_yolo'/'site_split_manifest.csv'
    pd.DataFrame([{'site_id':'a','split':'train'},{'site_id':'b','split':'test'},{'site_id':'c','split':'test'}]).to_csv(split,index=False)
    report={'status':'TRAINED_AND_EVALUATED','weights_sha256':_hash(weight),
            'source_manifest_sha256':_hash(source),'split_manifest_sha256':_hash(split),
            'test_metrics':{'precision':.81,'recall':.52,'mAP50':.66,'mAP50_95':.31}}
    path=root/'detector'/'training_report.json';path.write_text(json.dumps(report))
    return path,weight,split,source

def test_untrained_returns_no_metrics(tmp_path):
    state=app.load_evaluated_model_summary(tmp_path)
    assert state['ready'] is False and state['metrics'] is None and state['test_sites'] is None
    page=app.build_dashboard_html(pd.DataFrame(),{'label':'Synthetic test'},evaluation=state)
    assert 'AI detections' in page and 'View AI test results' in page and 'id="run-ai"' in page
    assert 'AI detector not trained' not in json.dumps(state) or not state['ready']

def test_evaluated_only_when_hashes_and_heldout_metrics_valid(tmp_path):
    report,weights,split,source=_artifacts(tmp_path)
    state=app.load_evaluated_model_summary(tmp_path)
    assert state['ready'] is True and state['test_sites']==2 and state['metrics']['mAP50_95']==.31
    details=json.loads(report.read_text());details['test_metrics']['mAP50'] = 1.2
    report.write_text(json.dumps(details))
    assert app.load_evaluated_model_summary(tmp_path)['ready'] is False
    details['test_metrics']['mAP50'] = .66
    report.write_text(json.dumps(details))
    weights.write_bytes(b'TAMPER')
    assert app.load_evaluated_model_summary(tmp_path)['ready'] is False

def test_metric_summary_does_not_use_reference_site_count(tmp_path):
    _,_,split,_=_artifacts(tmp_path)
    state=app.load_evaluated_model_summary(tmp_path)
    assert state['test_sites']==2
    pd.DataFrame([{'site_id':'test','split':'train'}]).to_csv(split,index=False)
    assert app.load_evaluated_model_summary(tmp_path)['ready'] is False
