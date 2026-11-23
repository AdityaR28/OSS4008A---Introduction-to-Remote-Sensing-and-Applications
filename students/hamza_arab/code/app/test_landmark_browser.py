from pathlib import Path
import json, importlib.util
import pandas as pd
from playwright.sync_api import sync_playwright
import hospital_chat as chat

D=Path(__file__).parent
spec=importlib.util.spec_from_file_location('landmark_map_app',str(D/'app.py'))
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
records=pd.DataFrame([
 {'name':'Kitchener General Hospital','latitude':43.456,'longitude':-80.512,'type':'Hospital','id':'1','address':'Kitchener hospital street','province':'ON','emergency_services':'','ownership':''},
 {'name':'Ottawa Hospital','latitude':45.4215,'longitude':-75.6972,'type':'Hospital','id':'2','address':'Ottawa','province':'ON','emergency_services':'','ownership':''},
 {'name':'Kitchener clinic','latitude':43.447,'longitude':-80.498,'type':'Clinic','id':'3','address':'Kitchener','province':'ON','emergency_services':'','ownership':''},
])
normalized=app.normalize_records(records)
reply=chat.answer({'question':'I need a hospital near Kitchener Victoria Park','lat':45.4215,'lon':-75.6972,'radius_m':25000},normalized)
assert reply['nav_results'][0]['name']=='Kitchener General Hospital',reply
html=app.build_dashboard_html(records)
mock=(D/'mock_leaflet.js').read_text()+"\nwindow.L.polygon=(ring,opts)=>window.L.marker([ring[0][0],ring[0][1]],opts);window.L.circleMarker=(loc,opts)=>window.L.marker(loc,opts);\n"
errs=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1550,'height':880})
 page.on('pageerror',lambda e:errs.append(str(e)))
 page.route('https://unpkg.com/**',lambda route:route.fulfill(status=200,body=mock if route.request.url.endswith('/leaflet.js') else '',content_type='application/javascript'))
 page.route('https://tiles.openfreemap.org/**',lambda route:route.fulfill(status=200,body='',content_type='application/javascript'))
 page.add_init_script('''window.google={colab:{kernel:{invokeFunction:async(name,args,kwargs)=>{
   window.__CALLS=(window.__CALLS||[]).concat([{name,args}]);
   if(name==='oss4008.hospital_chat')return {data:{'application/json':__REPLY__}};
   return {data:{'application/json':{ok:true,features:[],image_chips:0}}};
 }}}};'''.replace('__REPLY__',json.dumps(reply,ensure_ascii=False)))
 page.goto('about:blank')
 page.set_content(html,wait_until='domcontentloaded')
 page.wait_for_timeout(140)
 page.locator('#question').fill('I need a hospital near Kitchener Victoria Park')
 page.locator('#chat-form button').click()
 page.wait_for_timeout(250)
 assert page.locator('.nav-result button').count()==1
 assert 'Kitchener General Hospital' in page.locator('#messages').inner_text()
 page.locator('.nav-result button').first.click()
 pos=page.evaluate('({center:window.__TEST_MAP.center, zoom:window.__TEST_MAP.zoom})')
 assert pos['zoom']==16 and abs(pos['center'][0]-43.456)<1e-5 and abs(pos['center'][1]+80.512)<1e-5,pos
 assert 'Map moved to Kitchener General Hospital' in page.locator('#messages').inner_text()
 assert not errs,errs
 print('PASS: landmark Q&A -> grounded hospital nav button -> correct pin (43.456,-80.512), zoom16; JS errors 0')
 browser.close()
