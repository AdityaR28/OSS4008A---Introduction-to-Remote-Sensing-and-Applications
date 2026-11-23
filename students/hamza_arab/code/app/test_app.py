"""Offline smoke tests for compact Colab dashboard and official data fallbacks."""
from io import BytesIO
import json
from pathlib import Path
import zipfile
import pandas as pd
import app
import prepare_odhf as prep

fixture=pd.DataFrame([
 {'name':'Fixture Hospital','latitude':45.43,'longitude':-75.70,'type':'hospital','emergency_services':'yes','ownership':'Example operator'},
 {'name':'Fixture Clinic','latitude':45.40,'longitude':-75.72,'type':'clinic','emergency_services':'no'},
])
assert len(app.normalize_records(fixture))==2
html=app.build_dashboard_html(fixture,{'label':'Test reference source','message':'Test only'})
assert 'cartocdn' not in html.lower()
assert 'tile.openstreetmap.org' in html
assert 'Facilities in selected area' not in html
assert 'Filters &amp; map layers' in html and 'Map assistant' in html
assert 'Draw circle' in html and 'leaflet.draw.js' in html
assert '"lat":45.43' in html
assert 'Test reference source' in html
assert 'grid-template-columns' in html and 'height:100%' in html
assert 'prediction-file' in html and 'show-pred' in html
assert 'class="results"' not in html
assert 'tiles.openfreemap.org/styles/liberty' in html
assert 'World_Imagery/MapServer/tile/' in html
assert 'World_Street_Map/MapServer/tile/' in html
assert 'referrer" content="strict-origin-when-cross-origin' in html
assert 'activateBase' in html
payload=html.split('<script type="application/json" id="facility-data">',1)[1].split('</script>',1)[0]
assert len(json.loads(payload))==2
unsafe=fixture.copy()
unsafe.loc[0,'name']='</script><img src=x onerror=alert(1)>'
assert '</script><img' not in app.build_dashboard_html(unsafe)

# V2 unit fixture uses a real-shaped GeoJSON record in a mock ZIP, not a fake production result.
blob=BytesIO()
with zipfile.ZipFile(blob,'w') as archive:
    archive.writestr('odhf_v2.geojson',json.dumps({'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{'id':'fixture','name':'Fixture Hospital','type':'hospital','emergency_services':'yes'},
         'geometry':{'type':'Point','coordinates':[-75.70,45.43]}}]}))
class Resp:
    def __init__(self,content=None,payload=None):self.content=content or b'';self.payload=payload
    def raise_for_status(self):pass
    def json(self):return self.payload
class V2Session:
    def get(self,*args,**kwargs):return Resp(content=blob.getvalue())
v2=prep._from_v2(V2Session())
assert len(v2)==1 and v2.iloc[0]['source_version']=='ODHF v2 (2026)'
assert v2.iloc[0]['emergency_services']=='yes'

# V1 pagination: verify 301 records are all gathered across two pages and no ER values are invented.
class V1Session:
    def get(self,url,params,**kwargs):
        if params.get('returnIdsOnly')=='true':return Resp(payload={'objectIds':list(range(1,302))})
        ids=[int(x) for x in params['objectIds'].split(',')]
        return Resp(payload={'features':[{'attributes':{'OBJECTID':i,'name_nom':f'Fixture {i}',
          'facility_type':'Hospitals','latitude':45.4+i/100000,'longitude':-75.7,
          'address':'Fixture','province_en':'Ontario'}} for i in ids]})
v1=prep._from_v1_service(V1Session(),'https://test.local/map/query')
assert len(v1)==301 and v1.emergency_services.eq('').all()
assert v1.source_version.eq('ODHF v1 (2020 fallback)').all()

print('PASS: compact UI, OpenFreeMap + raster fallback, no result panel, safety, V2 parsing, V1 pagination')
