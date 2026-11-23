import sys, json, ast, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import pytest
from shapely.geometry import Polygon, MultiPolygon, box
import hospital_footprint as f

LAT, LON = 43.456, -80.512
# ~0.0001deg latitude ~= 11.1m, longitude ~8.1m in Kitchener.
def poly(dx,dy,rx=.0003,ry=.0003,category='yes',name=''):
    return {'geometry':box(LON+dx-rx,LAT+dy-ry,LON+dx+rx,LAT+dy+ry),'class':category,'id':name}

def test_trace_multiple_nearby_large_roofs_even_if_not_connected():
    rows=[poly(.0003,0,name='main'),poly(-.0010,.0007,name='annex'),poly(.0015,0,name='other'),
          poly(.0030,0,name='outside'),poly(.0005,.0002,rx=.00006,ry=.00005,name='small_house'),
          poly(.0002,.0004,category='house',name='tagged_house')]
    result=f.select_nearby_roofs(rows,LAT,LON,180)
    names={p['properties']['overture_id'] for p in result}
    assert {'main','annex','other'}<=names, names
    assert not {'outside','small_house','tagged_house'} & names
    assert all(p['geometry']['type']=='Polygon' for p in result)

def test_circle_radius_in_metres_not_tile_squares():
    # A 45m off-centre large building can be inside a 60m circle but excluded by 25m circle.
    rows=[poly(.00060,0,rx=.00035,ry=.0003,name='east')]
    assert len(f.select_nearby_roofs(rows,LAT,LON,60))==1
    assert len(f.select_nearby_roofs(rows,LAT,LON,20))==0
    with pytest.raises(ValueError,match='radius'):
        f.select_nearby_roofs(rows,LAT,LON,500)

def test_courtyard_hole_is_retained_for_real_roof_shape():
    outer=box(LON-.001,LAT-.001,LON+.001,LAT+.001)
    hole=box(LON-.0002,LAT-.0002,LON+.0002,LAT+.0002)
    roof=Polygon(outer.exterior.coords,[hole.exterior.coords])
    result=f.select_nearby_roofs([{'geometry':roof,'class':'hospital','id':'courtyard'}],LAT,LON)
    assert len(result)==1
    assert len(result[0]['geometry']['coordinates'])==2
    assert result[0]['properties']['evidence']=='overture_class_hospital'

def test_large_road_or_parking_not_auto_generated():
    # Without an actual building feature the source returns nothing, not a roof contour.
    result=f.select_nearby_roofs([],LAT,LON)
    assert result==[]

def test_cached_new_radii_and_source_not_approving_labels(tmp_path):
    rows=[poly(.0002,0,name='a')]
    count={'n':0}
    def provider(bb):
        count['n']+=1
        return rows
    first=f.mapped_footprint(tmp_path,LAT,LON,provider=provider,radius_m=180,mode="nearby")
    second=f.mapped_footprint(tmp_path,LAT,LON,provider=provider,radius_m=180,mode="nearby")
    third=f.mapped_footprint(tmp_path,LAT,LON,provider=provider,radius_m=200,mode="nearby")
    assert count['n']==2 and not first['cached'] and second['cached'] and not third['cached']
    assert all('verified hospital' in p['properties']['warning'] for p in first['features'])
    assert list((tmp_path/'reference_footprints').glob('*.json'))
    assert not (tmp_path/'review_manifest.csv').exists()

def test_invalid_radius_and_source_failure(tmp_path):
    with pytest.raises(ValueError):f.mapped_footprint(tmp_path,LAT,LON,radius_m=400)
    with pytest.raises(RuntimeError,match='offline'):
        f.mapped_footprint(tmp_path,LAT,LON,provider=lambda bbox:(_ for _ in ()).throw(RuntimeError('offline')))
    assert not list((tmp_path/'reference_footprints').glob('*.json'))

def test_ui_no_unused_controls_and_has_trace_action():
    src=(Path(__file__).parent/'app.py').read_text()
    assert 'id="trace-roofs"' in src and 'Trace nearby buildings' in src
    assert 'mappedFootprintLayer=L.featureGroup()' in src
    assert 'L.geoJSON(f.geometry' in src
    for obsolete in ['id="er"', 'id="availability-note"', 'id="view-evaluation"', 'id="prediction-file"', "$('view-evaluation')"]:
        assert obsolete not in src,obsolete

def test_notebook_embeds_current_source_and_both_pin_circle_callback_paths():
    book=json.loads((Path(__file__).parent/'04_canada_colab_inline.ipynb').read_text())
    src=''.join(book['cells'][3]['source'])
    defs=ast.parse(src)
    sources={target.id:node.value.value for node in defs.body if isinstance(node,ast.Assign)
             for target in node.targets if isinstance(target,ast.Name) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)}
    assert sources['APP_SOURCE']==(Path(__file__).parent/'app.py').read_text()
    assert sources['FOOTPRINT_SOURCE']==(Path(__file__).parent/'hospital_footprint.py').read_text()
    assert "request.get('radius_m',180)" in src
    assert "if 'key' in request:" in src
    assert "lat,lon=float(request['lat']),float(request['lon'])" in src
    assert 'SCREEN_MARGIN_PX=65' in src
