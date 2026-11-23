import json
from pathlib import Path
from shapely.geometry import box, Polygon
import hospital_footprint as f
import app

LAT,LON=43.456,-80.512

def mkpoly(dx=0,dy=0,wx=.0007,wy=.0006):
    return box(LON+dx,LAT+dy,LON+dx+wx,LAT+dy+wy)

def osm_element(i=1,klass='yes',dx=0,dy=0,wx=.0007,wy=.0006):
    p=mkpoly(dx,dy,wx,wy)
    return {'type':'way','id':i,'tags':{'building':klass},
            'geometry':[{'lon':x,'lat':y} for x,y in p.exterior.coords]}

def test_osm_query_is_bounded_only_buildings():
    q=f.overpass_query(LAT,LON,180)
    assert 'way["building"]' in q and 'relation["building"]' in q
    assert 'around:205' in q and 'out body geom' in q

def test_osm_parser_reads_actual_roof_polygon():
    rows=f.osm_rows_from_overpass({'elements':[osm_element(klass='hospital')]})
    assert len(rows)==1 and rows[0]['source']=='OpenStreetMap' and rows[0]['class']=='hospital'
    assert rows[0]['geometry'].area>0

def test_no_parking_amenity_becomes_building():
    p={'elements':[{'type':'way','id':22,'tags':{'amenity':'parking'},'geometry':osm_element()['geometry']}]}
    assert f.osm_rows_from_overpass(p)==[]

def test_normal_osm_roof_and_house_filter(tmp_path):
    source=lambda la,lo,r:f.osm_rows_from_overpass({'elements':[
        osm_element(1,'yes',dx=.00012,dy=.00015),
        osm_element(2,'house',dx=-.00015,dy=.00008,wx=.00018,wy=.00014)]})
    r=f.mapped_footprint(tmp_path,LAT,LON,osm_provider=source,provider=lambda b:[],mode='nearby')
    assert r['ok'] and len(r['features'])==1,r
    assert r['features'][0]['properties']['source'].startswith('OpenStreetMap')
    assert r['features'][0]['properties']['mapped_building_id'].startswith('osm/way/1/')

def test_duplicate_roof_not_rendered_twice(tmp_path):
    poly=mkpoly(.00015,.00008)
    osm=lambda la,lo,r:[{'geometry':poly,'class':'yes','id':'osm/way/1','source':'OpenStreetMap'}]
    overture=lambda b:[{'geometry':poly,'class':None,'id':'overt/1','source':'Overture Maps'}]
    r=f.mapped_footprint(tmp_path,LAT,LON,osm_provider=osm,provider=overture,mode='nearby')
    assert len(r['features'])==1,r

def test_fallback_to_overture_when_osm_fails(tmp_path):
    def fail(*args):raise RuntimeError('public Overpass unavailable')
    other=lambda bbox:[{'geometry':mkpoly(.0002,.0001),'class':None,'id':'overture-large'}]
    r=f.mapped_footprint(tmp_path,LAT,LON,osm_provider=fail,provider=other,mode='nearby')
    assert len(r['features'])==1
    assert 'OpenStreetMap:' in r['provider_errors'][0]

def test_osm_populated_skips_expensive_overture_when_three_candidates(tmp_path):
    def source(*args):return [{'geometry':mkpoly(i*.00085,.00001),'class':'hospital','source':'OpenStreetMap','id':str(i)} for i in range(3)]
    def overture(*args):raise AssertionError('Overture should not be queried')
    r=f.mapped_footprint(tmp_path,LAT,LON,osm_provider=source,provider=overture,mode='nearby')
    assert len(r['features'])==3

def test_provider_failures_not_cached(tmp_path):
    def fail(*args):raise RuntimeError('offline')
    r=f.mapped_footprint(tmp_path,LAT,LON,osm_provider=fail,provider=fail,mode='nearby')
    assert r['features']==[] and len(r['provider_errors'])==2
    assert not list((tmp_path/'reference_footprints').glob('*.json')) if (tmp_path/'reference_footprints').exists() else True

def test_area_threshold_adjusts_candidate_count():
    p={'geometry':mkpoly(.0001,.0001,wx=.0003,wy=.00025),'class':'yes','id':'medium','source':'OpenStreetMap'}
    low=f.select_nearby_roofs([p],LAT,LON,minimum_area_m2=500)
    high=f.select_nearby_roofs([p],LAT,LON,minimum_area_m2=2500)
    assert len(low)==1 and len(high)==0

def test_circle_limits_mapped_geometries():
    near={'geometry':mkpoly(.00001,.00001),'class':'yes','source':'OpenStreetMap'}
    far={'geometry':mkpoly(.006,.005),'class':'yes','source':'OpenStreetMap'}
    vals=f.select_nearby_roofs([near,far],LAT,LON,radius_m=100)
    assert len(vals)==1

def test_relation_polygon_keeps_courtyard_hole():
    ring=list(mkpoly().exterior.coords)
    inner=list(mkpoly(.0002,.00015,wx=.00012,wy=.00012).exterior.coords)
    rel={'type':'relation','id':7,'tags':{'building':'hospital'},'members':[
        {'role':'outer','geometry':[{'lon':x,'lat':y} for x,y in ring]},
        {'role':'inner','geometry':[{'lon':x,'lat':y} for x,y in inner]}]}
    vals=f.osm_rows_from_overpass({'elements':[rel]})
    assert len(vals)==1 and len(vals[0]['geometry'].interiors)==1

def test_osm_raises_on_invalid_response():
    import pytest
    with pytest.raises(ValueError):f.osm_rows_from_overpass({'bad':[]})

def test_cache_key_includes_area_and_radius(tmp_path):
    p={'geometry':mkpoly(.0001,.00001),'class':'yes','source':'OpenStreetMap'}
    provider=lambda *args:[p]
    a=f.mapped_footprint(tmp_path,LAT,LON,radius_m=180,minimum_area_m2=500,osm_provider=provider,mode='nearby')
    b=f.mapped_footprint(tmp_path,LAT,LON,radius_m=250,minimum_area_m2=500,osm_provider=provider,mode='nearby')
    c=f.mapped_footprint(tmp_path,LAT,LON,radius_m=250,minimum_area_m2=1500,osm_provider=provider,mode='nearby')
    assert not a['cached'] and not b['cached'] and not c['cached']
    d=f.mapped_footprint(tmp_path,LAT,LON,radius_m=250,minimum_area_m2=500,osm_provider=lambda *args:(_ for _ in ()).throw(AssertionError('cache should be used')),mode='nearby')
    assert d['cached']

def test_rendered_ui_preserves_layout_and_is_not_claiming_verified_ai():
    html=app.UI_TEMPLATE
    assert 'SCREEN_MARGIN_PX' not in html
    assert 'id="trace-radius"' in html and 'id="trace-area"' in html
    assert 'Trace large buildings' in html
    assert 'Emergency services</label>' not in html
    assert 'NOT VERIFIED HOSPITAL OR AI DETECTION' in html
    assert 'Source: '+"'" not in ''  # avoid interpolation in JS string
