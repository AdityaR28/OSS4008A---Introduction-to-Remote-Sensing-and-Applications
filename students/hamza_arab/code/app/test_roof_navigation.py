from pathlib import Path
import importlib.util, math, ast, json
import pandas as pd
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from rasterio.io import MemoryFile
import numpy as np
import pytest
import hospital_chat as chat
import hospital_area_scan as scan

ROOT=Path(__file__).parent

def records():
    return [
        {'key':'1','name':'Kitchener General Hospital','lat':43.456,'lon':-80.512,'type':'Hospital','address':'Kitchener, Ontario','province':'ON'},
        {'key':'2','name':'Other Kitchener Hospital','lat':43.47,'lon':-80.51,'type':'Hospital','address':'Kitchener, Ontario','province':'ON'},
        {'key':'3','name':'Ottawa Hospital','lat':45.4215,'lon':-75.6972,'type':'Hospital','address':'Ottawa, Ontario','province':'ON'},
        {'key':'4','name':'Kitchener small clinic','lat':43.4471,'lon':-80.4982,'type':'Clinic','address':'Kitchener','province':'ON'},
    ]

def test_victoria_park_near_hospital_overrides_previous_ottawa_city_and_focus():
    r=chat.answer({'question':'I am looking for a hospital near Kitchener Victoria Park',
                   'lat':45.4215,'lon':-75.6972,'radius_m':1000,'selected_key':'3'},records())
    assert r['ok'] and r['mode']=='reference-only'
    assert 'Victoria Park, Kitchener' in r['answer']
    assert [x['key'] for x in r['nav_results']]==['1','2']
    assert 'Ottawa' not in r['answer']
    assert r['nav_results'][0]['distance_km']<3


def test_kitchener_city_hall_alias_navigation():
    r=chat.answer({'question':'find hospital close to Kitchener Hall','lat':45.4,'lon':-75.7},records())
    assert 'Kitchener City Hall' in r['answer']
    assert r['nav_results'][0]['name']=='Kitchener General Hospital'
    assert 1 < r['nav_results'][0]['distance_km'] < 4


def test_other_landmark_resolver_is_optional_and_result_grounded():
    resolver=lambda query:{'lat':43.45,'lon':-80.50,'display_name':'Test landmark Kitchener'}
    r=chat.answer({'question':'hospital near Test Place Kitchener','lat':45.4,'lon':-75.7},records(),landmark_resolver=resolver)
    assert r['origin_label']=='Test landmark Kitchener'
    assert r['nav_results'][0]['key']=='1'


def test_no_ground_truth_hallucinations_for_unknown_landmark():
    r=chat.answer({'question':'hospital near unknown place x', 'lat':45.4215,'lon':-75.6972}, records(),
                  landmark_resolver=lambda *_: None)
    assert 'Could not locate' in r['note']
    assert r['nav_results'][0]['key']=='3'


def test_rectangle_screening_filters_houses_retains_large_roofs():
    from rasterio.transform import from_bounds
    from pyproj import Transformer
    mx,my=Transformer.from_crs('EPSG:4326','EPSG:3857',always_xy=True).transform(-80.51,43.45)
    class R:pass
    ds=R();ds.crs='EPSG:3857';ds.width=640;ds.height=640
    ds.transform=from_bounds(mx-256,my-256,mx+256,my+256,width=640,height=640)
    tiny=scan.roof_box_metrics(scan.bbox_polygon(ds,[100,100,125,125]))
    large=scan.roof_box_metrics(scan.bbox_polygon(ds,[200,200,310,310]))
    narrow=scan.roof_box_metrics(scan.bbox_polygon(ds,[200,200,215,500]))
    assert not scan.passes_roof_screen(tiny),tiny
    assert scan.passes_roof_screen(large),large
    assert not scan.passes_roof_screen(narrow),narrow
    assert large['bbox_area_m2']>750 and tiny['bbox_area_m2']<750
    assert 'v2_size_screened' in scan.SCAN_VERSION


def test_notebook_integrity_and_navigation_source():
    import nbformat
    notebook=nbformat.read(ROOT/'04_canada_colab_inline.ipynb',as_version=4)
    tree=ast.parse(notebook.cells[-1].source)
    vals={t.id:n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
          for t in n.targets if isinstance(t,ast.Name) and t.id in {'APP_SOURCE','AREA_SCAN_SOURCE','CHAT_SOURCE','WEAK_ROOF_SOURCE'}}
    for key,file in [('APP_SOURCE','app.py'),('AREA_SCAN_SOURCE','hospital_area_scan.py'),('CHAT_SOURCE','hospital_chat.py'),('WEAK_ROOF_SOURCE','weak_osm_training.py')]:
        assert vals[key]==(ROOT/file).read_text()
    assert 'SCREEN_MARGIN_PX=65' in notebook.cells[-1].source
    assert 'Navigate' in vals['APP_SOURCE']
