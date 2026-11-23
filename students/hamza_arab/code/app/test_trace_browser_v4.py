"""Actual Chromium JS click smoke test with mocked map provider and Colab response."""
import json,sys,re
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).parent))
import app

MOCK_JS=r'''
(function(){
const mkLayer=()=>({events:{},addTo(parent){if(parent.layers)parent.layers.push(this);return this;},addLayer(child){if(this.layers)this.layers.push(child);return this;}, bindPopup(content){this.popup=content;return this;},clearLayers(){this.layers=[];return this;}, getBounds(){return {isValid:()=>true}}, on(name,cb){this.events[name]=cb;return this},setLatLng(){return this},setRadius(){return this},getRadius(){return this},getLatLng(){return this},openPopup(){return this}});
let m=mkLayer();m.layers=[];m.getZoom=()=>16;m.zoom=16;
m.invalidateSize=()=>m;m.setView=function(){return this};m.flyTo=function(pos,z){this.zoom=z;this.flyPos=pos;return this};m.fitBounds=function(b,o){this.lastFit={b,o};return this};m.hasLayer=()=>true;m.removeLayer=()=>m;
m.distance=(a,b)=>{const x=Array.isArray(a)?{lat:a[0],lng:a[1]}:a;const y=Array.isArray(b)?{lat:b[0],lng:b[1]}:b;return Math.hypot((x.lat-y.lat)*111100,(x.lng-y.lng)*81200)};
m.fire=function(name,data){if(this.events[name])this.events[name](data)};
window._map=m;window._markers=[];window._geojsons=[];window._calls=[];
const L={map:()=>m,latLng:(lat,lng)=>({lat,lng}),tileLayer:()=>mkLayer(),marker:(coords,opts)=>{const a=mkLayer();a.pos=coords;window._markers.push(a);return a;},circle:(coords,opts)=>{const a=mkLayer();a.pos=coords;a.radius=opts.radius;a.getRadius=()=>a.radius;a.getLatLng=()=>({lat:coords[0],lng:coords[1]});return a;},divIcon:()=>({}),markerClusterGroup:()=>{const l=mkLayer();l.layers=[];l.addLayer=function(child){this.layers.push(child);return this};l.zoomToShowLayer=(child,cb)=>cb();return l;},layerGroup:()=>{const a=mkLayer();a.layers=[];return a;},featureGroup:()=>{const a=mkLayer();a.layers=[];return a;},geoJSON:(geom,opts)=>{let a=mkLayer();a.shape=geom;window._geojsons.push(a);return a;},control:{zoom:()=>mkLayer(),layers:()=>mkLayer()}};
L.maplibreGL=undefined;
L.TileLayer=function TileLayer(){};window.L=L;
window.google={colab:{kernel:{invokeFunction:async(name,args,kwargs)=>{
  window._calls.push({name,args,kwargs});
  const ring=[[-80.513,43.456],[-80.513,43.457],[-80.512,43.457],[-80.512,43.456],[-80.513,43.456]];
  return {data:{'application/json':{ok:true,features:[{type:'Feature',geometry:{type:'Polygon',coordinates:[ring]},properties:{area_m2:6500,evidence:'large_mapped_building_near_reference_only',distance_from_search_centre_m:20}}]}}};
}}}};
})();
'''

def test_browser_trace():
    df=pd.DataFrame([{'id':'x','name':'Kitchener Hospital','latitude':45.4215,'longitude':-75.6972,
                      'type':'Hospital','address':'Kitchener','province':'ON'}])
    raw=app.build_dashboard_html(df)
    raw=re.sub(r'<script src="[^"]+"></script>','',raw)
    raw=re.sub(r'<link rel="stylesheet" href="[^"]+"/>','',raw)
    raw=raw.replace('<script>\n(function()', '<script>'+MOCK_JS+'</script>\n<script>\n(function()',1)
    errors=[]
    with sync_playwright() as p:
        b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=b.new_page(viewport={'width':1280,'height':800})
        page.on('pageerror',lambda err:errors.append(str(err)))
        page.set_content(raw,wait_until='load')
        assert page.locator('#er').count()==0
        assert page.locator('#trace-roofs').count()==1
        assert page.locator('#view-evaluation').count()==0
        # Click red hospital marker's popup control: uses official pin key.
        count=page.evaluate('window._markers?.length || 0')
        assert count>=1,count
        page.evaluate('''() => {const m=window._markers.find(p=>p.popup?.textContent?.includes('Kitchener'));m.popup.querySelector('button:last-child').click();}''')
        page.wait_for_timeout(100)
        assert 'oss4008.hospital_footprint' in page.evaluate('window._calls.at(-1).name')
        assert page.evaluate('window._calls.at(-1).args[0].key') == '0'
        assert page.evaluate('window._calls.at(-1).args[0].radius_m') == 180
        assert page.evaluate('window._calls.at(-1).args[0].minimum_area_m2') == 900
        assert page.evaluate('window._geojsons.length')>=1
        assert 'amber mapped building outlines' in page.locator('#messages').inner_text()
        # Draw a 200m circle and click main toolbar trace button.
        page.evaluate("""() => {const l={getLatLng:()=>({lat:43.456, lng:-80.512}),getRadius:()=>200,addTo:()=>l};window._map.fire('draw:created',{layerType:'circle',layer:l})}""")
        page.select_option('#trace-area','1500')
        page.click('#trace-roofs')
        page.wait_for_timeout(100)
        q=page.evaluate('window._calls.at(-1).args[0]')
        assert q['radius_m']==200 and abs(q['lat']-43.456)<1e-5 and q['mode']=='nearby' and q['minimum_area_m2']==1500
        assert errors==[],errors
        b.close()

if __name__=='__main__':
 test_browser_trace();print('Chromium browser trace integration PASS')
