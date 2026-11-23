import json
from pathlib import Path
from shapely.geometry import box
import pytest
import hospital_chat as h
import hospital_footprint as f

RECORDS=[{'key':'1','name':'Vancouver Hospital','type':'Hospital','lat':49.28,'lon':-123.12,'address':'Vancouver', 'province':'BC', 'er':'', 'ownership':''},
         {'key':'2','name':'Kitchener Regional Hospital','type':'Hospital','lat':43.456,'lon':-80.512,'address':'Kitchener', 'province':'ON','er':'','ownership':''}]

def test_downtown_vancouver_landmark_navigates_from_ottawa():
    reply=h.answer({'question':'I need a hospital near downtown Vancouver','lat':45.42,'lon':-75.69,'radius_m':25000},RECORDS)
    assert reply['nav_results'][0]['key']=='1'
    assert 'Vancouver' in reply['origin_label']

def test_generic_city_supported_without_hardcoding():
    resolver=lambda q:{'lat':49.2827,'lon':-123.1207,'display_name':'Vancouver, BC'}
    reply=h.answer({'question':'Find a hospital in Vancouver','lat':45.42,'lon':-75.69},RECORDS,landmark_resolver=resolver)
    assert reply['nav_results'][0]['key']=='1'

def test_ambiguous_park_does_not_wrongly_pick_city():
    reply=h.answer({'question':'hospital near Victoria Park','lat':45.42,'lon':-75.69},RECORDS)
    assert reply['nav_results']==[] and 'specify the city' in reply['answer'].lower()

def test_undiscovered_landmark_does_not_fallback_to_wrong_city():
    reply=h.answer({'question':'hospital near Made-Up Landmark, Somewhere','lat':45.42,'lon':-75.69},RECORDS,landmark_resolver=lambda q:None)
    assert reply['nav_results']==[] and 'Could not locate' in reply['answer']

def test_actual_local_generation_is_enabled_and_navigation_is_grounded():
    called=[]
    def writer(prompt):
        called.append(prompt)
        return 'Here is the closest reference hospital, with details limited to what is listed.'
    reply=h.answer({'question':'hospital near downtown Vancouver','lat':45.42,'lon':-75.69,'enable_local_llm':True},
                   RECORDS,local_generator=writer)
    assert reply['mode']=='local-qwen' and reply['llm_enabled']
    assert reply['nav_results'][0]['key']=='1'
    assert 'official_references' in called[0]

def test_local_failure_is_explicit_and_still_has_navigation():
    def no_model(_):raise RuntimeError('model source blocked')
    reply=h.answer({'question':'hospital near downtown Vancouver','lat':45.42,'lon':-75.69,'enable_local_llm':True},
                   RECORDS,local_generator=no_model)
    assert reply['mode']=='reference-only' and 'unavailable' in reply['note']
    assert reply['nav_results'][0]['key']=='1'

def test_mapped_footprint_prefers_hospital_class_to_nearby_large_generic():
    lat,lon=43.456,-80.512
    # 0.001deg approx 80m E-W, 110m N-S; hospital and generic are plausible.
    rows=[{'geometry':box(lon-.0002,lat-.00017,lon+.0002,lat+.00017),'class':None,'id':'generic'},
          {'geometry':box(lon+.0004,lat-.0003,lon+.0011,lat+.0003),'class':'hospital','id':'hospital'}]
    results=f.select_footprints(rows,lat,lon)
    assert results and results[0]['properties']['evidence']=='overture_class_hospital'
    assert results[0]['properties']['method']=='mapped_building_near_odhf_unverified'

def test_small_house_rejected_as_footprint(tmp_path):
    lat,lon=43.456,-80.512
    tiny={'geometry':box(lon-.00008,lat-.00007,lon+.00008,lat+.00007),'class':'house'}
    result=f.mapped_footprint(tmp_path,lat,lon,provider=lambda bbox:[tiny])
    assert result['features']==[]
    again=f.mapped_footprint(tmp_path,lat,lon,provider=lambda bbox:(_ for _ in ()).throw(AssertionError('not cached')))
    assert again['cached'] and again['features']==[]

def test_no_footprint_fabricated_on_provider_failure(tmp_path):
    with pytest.raises(RuntimeError,match='offline'):
        f.mapped_footprint(tmp_path,43.456,-80.512,provider=lambda b:(_ for _ in ()).throw(RuntimeError('offline')))
    assert not list((tmp_path/'reference_footprints').glob('*.json')) if (tmp_path/'reference_footprints').exists() else True
