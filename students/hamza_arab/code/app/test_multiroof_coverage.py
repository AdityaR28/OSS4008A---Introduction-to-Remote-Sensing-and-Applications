import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from shapely.geometry import box, MultiPolygon
import hospital_footprint as f

LAT,LON=43.456,-80.512
# Produce rough geospatial rect dimensions 80m = .001deg E/W, 110m = .001deg N/S.
def outline(dx,dy,rx=.00045,ry=.0004,kind='yes',id=''):
    return {'geometry':box(LON+dx-rx,LAT+dy-ry,LON+dx+rx,LAT+dy+ry),'class':kind,'id':id}

def test_multiple_wings_not_two():
    # Hospital body + four adjacent wings separated by 4–15m gaps.
    pieces=[outline(0,0,kind='hospital',id='a'),
            outline(.001,0,kind='yes',id='b'),
            outline(.002,0,kind='yes',id='c'),
            outline(.002,.0009,kind='yes',id='d'),
            outline(.001,.0009,kind='yes',id='e'),
            outline(.0035,.002,kind='house',id='house')]
    vals=f.select_footprints(pieces,LAT,LON)
    assert len(vals)>=5,len(vals)
    assert vals[0]['properties']['evidence']=='overture_class_hospital'
    assert 'house' not in [v['properties']['overture_id'] for v in vals]

def test_distant_unconnected_large_commercial_excluded():
    vals=f.select_footprints([outline(0,0,kind='hospital'),outline(.003,0,rx=.00025,ry=.0004,id='unrelated')],LAT,LON)
    assert len(vals)==1

def test_multi_polygon_wings_preserved():
    a=box(LON-.0006,LAT-.0004,LON+.0004,LAT+.0004)
    b=box(LON+.00048,LAT-.0004,LON+.0013,LAT+.0004)
    vals=f.select_footprints([{'geometry':MultiPolygon([a,b]),'class':'hospital','id':'complex'}],LAT,LON)
    assert len(vals)==2

def test_cache_new_version_and_all_wings(tmp_path):
    rows=[outline(0,0,kind='hospital')]+[outline(.001*i,0,kind='yes',id=str(i)) for i in range(1,5)]
    first=f.mapped_footprint(tmp_path,LAT,LON,provider=lambda bb:rows)
    assert first['ok'] and len(first['features'])==5 and not first['cached']
    second=f.mapped_footprint(tmp_path,LAT,LON,provider=lambda bb:(_ for _ in ()).throw(AssertionError('would refetch')))
    assert second['cached'] and len(second['features'])==5

if __name__=='__main__':
 test_multiple_wings_not_two();test_distant_unconnected_large_commercial_excluded();test_multi_polygon_wings_preserved();print('standalone ok')

def test_many_connected_wings_bounded_and_no_overflow():
    # A dense campus with more than two separately mapped buildings, plus a
    # hard cap that prevents an excessive number of outlines in the browser.
    hospital=outline(0,0,rx=.0004,ry=.0004,kind='hospital',id='main')
    others=[outline(i*.00082,j*.00078,rx=.00038,ry=.00035,kind='yes',id=f'{i}-{j}')
            for i in range(4) for j in range(4) if (i,j)!=(0,0)]
    vals=f.select_footprints([hospital,*others],LAT,LON)
    assert len(vals)>=10 and len(vals)<=f.MAX_FOOTPRINTS,len(vals)
    assert all(v['properties']['warning'].startswith('Mapped roof/') for v in vals)
