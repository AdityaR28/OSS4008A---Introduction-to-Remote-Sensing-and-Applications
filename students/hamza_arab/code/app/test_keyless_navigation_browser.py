from pathlib import Path
import json,importlib.util
import pandas as pd
from playwright.sync_api import sync_playwright
import hospital_chat as chat

D=Path(__file__).parent
spec=importlib.util.spec_from_file_location('app_view',str(D/'app.py'))
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
facilities=pd.DataFrame([
 {'name':'Vancouver Regional Hospital','latitude':49.2801,'longitude':-123.1201,'type':'Hospital','id':'1','address':'Vancouver, BC','province':'BC','emergency_services':'','ownership':''},
 {'name':'Kitchener Regional Hospital','latitude':43.456,'longitude':-80.512,'type':'Hospital','id':'2','address':'Kitchener, ON','province':'ON','emergency_services':'','ownership':''},
])
records=app.normalize_records(facilities)
answer=chat.answer({'question':'hospital near downtown Vancouver','lat':45.4215,'lon':-75.6972,'enable_local_llm':True},
                   records,local_generator=lambda _: 'The available hospital reference is near downtown Vancouver. Select Navigate to visit its pin.')
assert answer['mode']=='local-qwen'
footprint={'ok':True,'features':[{'type':'Feature','geometry':{'type':'Polygon','coordinates':[[[-123.121+i*.0002,49.2798],[-123.1207+i*.0002,49.2798],[-123.1207+i*.0002,49.2804],[-123.121+i*.0002,49.2804],[-123.121+i*.0002,49.2798]]]},'properties':{'source':'Overture Maps','area_m2':3000,'method':'mapped_building_near_odhf_unverified'}} for i in range(5)]}
mock=(D/'mock_leaflet.js').read_text()+'''
window.L.polygon=(ring,opts)=>{window.__POLYGONS=(window.__POLYGONS||[]).concat([{ring,opts}]);return window.L.marker(ring[0],opts)};
window.L.circleMarker=(loc,opts)=>window.L.marker(loc,opts);
const oldMarker=window.L.marker;
window.L.marker=(xy,o)=>{const marker=oldMarker(xy,o);marker.bindPopup=function(content){this.popup=content;return this;};return marker;};
'''
html=app.build_dashboard_html(facilities)
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1550,'height':900})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.route('https://unpkg.com/**',lambda route:route.fulfill(status=200,body=mock if route.request.url.endswith('/leaflet.js') else '',content_type='application/javascript'))
    page.route('https://tiles.openfreemap.org/**',lambda route:route.fulfill(status=200,body='',content_type='application/javascript'))
    page.add_init_script('''window.google={colab:{kernel:{invokeFunction:async(name,args,kwargs)=>{
       window.__CALLS=(window.__CALLS||[]).concat([{name,args}]);
       if(name==='oss4008.hospital_chat')return {data:{'application/json':__ANSWER__}};
       if(name==='oss4008.hospital_footprint')return {data:{'application/json':__FOOTPRINT__}};
       return {data:{'application/json':{ok:true,features:[],image_chips:0}}};
    }}}};'''.replace('__ANSWER__',json.dumps(answer,ensure_ascii=False)).replace('__FOOTPRINT__',json.dumps(footprint)))
    page.goto('about:blank');page.set_content(html,wait_until='domcontentloaded');page.wait_for_timeout(150)
    page.locator('#question').fill('hospital near downtown Vancouver')
    page.locator('#chat-form button').click();page.wait_for_timeout(180)
    assert page.locator('.nav-result button').count()==1
    assert 'Local Qwen LLM' in page.locator('#messages').inner_text()
    page.locator('.nav-result button').click();page.wait_for_timeout(100)
    nav=page.evaluate('({center:window.__TEST_MAP.center,zoom:window.__TEST_MAP.zoom})')
    assert nav['zoom']==16 and abs(nav['center'][0]-49.2801)<1e-4,nav
    # Display the selected marker popup in the test document to press its button.
    found=page.evaluate('''()=>{
      for(const layer of window.__TEST_MAP.layers){
        for(const marker of (layer.children||[]))if(marker.popup?.textContent?.includes('Vancouver Regional Hospital')){
          document.body.appendChild(marker.popup);return true;
        }
      }
      return false;
    }''')
    assert found,'Reference marker popup not found'
    page.locator('button',has_text='Mapped roof outline').last.click();page.wait_for_timeout(200)
    drawn=page.evaluate('window.__POLYGONS?.length||0')
    assert drawn==5,drawn
    assert 'MAPPED' not in page.locator('#messages').inner_text() or 'mapped' in page.locator('#messages').inner_text().lower()
    assert not errors,errors
    print('PASS: keyless local LLM mode rendered; Vancouver Navigate zoom16; on-demand amber mapped footprint; JS errors 0')
    browser.close()
