from pathlib import Path
import ast,hashlib,json,sys,types
import pandas as pd
import nbformat

BASE=Path(__file__).parent
PROJECT=Path('/mnt/data/oss4008_bulk')

def notebook(name):
    path=BASE/name
    obj=nbformat.read(path,as_version=4)
    nbformat.validate(obj)
    for idx,c in enumerate(obj.cells):
        if c.cell_type=='code':
            compile('\n'.join('pass' if line.lstrip().startswith('%') else line for line in c.source.splitlines()),path.name+':'+str(idx),'exec')
    return obj

def assigned_literal(code,name):
    for n in ast.parse(code).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(f'{name} not present')


def test_map_only_and_preserve_approved_ui():
    n=notebook('04_canada_colab_inline.ipynb')
    assert len(n.cells)==3
    assert 'Notebook 08' in n.cells[0].source
    assert 'Notebook 07' in n.cells[0].source
    app=assigned_literal(n.cells[2].source,'APP_SOURCE')
    old=(PROJECT/'app.py').read_text()
    assert app==old
    assert 'SCREEN_MARGIN_PX=65' in n.cells[-1].source
    assert 'SCREEN_MARGIN_PX=65' in app or 'screen_margin_px=65' in app
    assert not any('train_and_evaluate(' in c.source for c in n.cells if c.cell_type=='code' and c is not n.cells[-1])
    assert 'launch_colab_dashboard(screen_margin_px=SCREEN_MARGIN_PX)' in n.cells[-1].source


def test_cached_reference_data_requires_no_download(tmp_path,monkeypatch):
    n=notebook('04_canada_colab_inline.ipynb')
    data=tmp_path/'drive/MyDrive/OSS4008_Hospital_AI_Pilot/reference_cache'
    data.mkdir(parents=True)
    records=pd.DataFrame([{'name':'Test hospital','latitude':45.4,'longitude':-75.7,'type':'Hospital'}])
    records.to_csv(data/'odhf_facilities.csv',index=False)
    (data/'odhf_facilities.source.json').write_text('{"source_version":"Test fixture"}')
    code=n.cells[1].source.replace('/content/drive/MyDrive',str(tmp_path/'drive/MyDrive'))
    code=code.replace('/content/odhf_facilities.csv',str(tmp_path/'odhf_facilities.csv'))
    code=code.replace('/content/odhf_facilities.source.json',str(tmp_path/'odhf_facilities.source.json'))
    code=code.replace('/content/prepare_odhf.py',str(tmp_path/'prepare_odhf.py'))
    fake_google=types.ModuleType('google')
    fake_colab=types.ModuleType('google.colab')
    fake_drive=types.ModuleType('google.colab.drive')
    fake_drive.mount=lambda _: (_ for _ in ()).throw(AssertionError('Would mount despite cached location'))
    fake_colab.drive=fake_drive
    monkeypatch.setitem(sys.modules,'google',fake_google)
    monkeypatch.setitem(sys.modules,'google.colab',fake_colab)
    monkeypatch.setitem(sys.modules,'google.colab.drive',fake_drive)
    ns={}
    exec(compile(code,'cache_loader','exec'),ns)
    loaded=pd.read_csv(tmp_path/'odhf_facilities.csv')
    assert loaded.iloc[0]['name']=='Test hospital'
    assert (tmp_path/'odhf_facilities.source.json').is_file()
    assert not (tmp_path/'prepare_odhf.py').exists()


def test_training_lab_defaults_safe_and_has_return_link():
    n=notebook('08_hospital_ai_pipeline.ipynb')
    all_code='\n'.join(c.source for c in n.cells if c.cell_type=='code')
    assert 'RUN_OSM_DRAFTS = False' in all_code
    assert 'TRAIN_WHEN_READY = False' in all_code
    assert 'EXPORT_PREDICTIONS = False' in all_code
    assert "source_dir=ROOT/'source'" in all_code
    assert "cache=ROOT/'reference_cache'" in all_code
    assert 'Return to FINAL MAP' in n.cells[0].source
    assert 'OPEN FINAL HOSPITAL MAP' in n.cells[-1].source
    # Laboratory default does NOT attempt the 100-site expansion.
    assert 'AUTO_DOWNLOAD_TO_100_HOSPITALS = True' not in all_code
