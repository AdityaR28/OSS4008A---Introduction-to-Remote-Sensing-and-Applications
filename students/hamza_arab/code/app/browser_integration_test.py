import asyncio
import json
import sys
from pathlib import Path
import pandas as pd
from playwright.async_api import async_playwright
import app

ROOT=Path(__file__).parent

async def main():
    df=pd.DataFrame([{'name':'Ottawa Civic Hospital TEST-REFERENCE','latitude':45.43,'longitude':-75.70,'type':'Hospital'},
      {'name':'Community Clinic TEST-REFERENCE','latitude':45.44,'longitude':-75.71,'type':'Clinic'}])
    result=app.build_dashboard_html(df, {'label':'MOCK TEST - NOT REAL'},[
        {'lat':45.429,'lon':-75.701,'conf':.81}])
    # Simulate Colab callback for an evaluated model; generated features are MOCK ONLY.
    callback_stub="""<script>window.google={colab:{kernel:{invokeFunction:async (name,args)=>({data:{'application/json':{ok:true,image_chips:1,features:[{geometry:{coordinates:[-75.7001,45.4302]},properties:{confidence:0.88}}]}}})}}};</script>"""
    result=result.replace('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>', callback_stub + '<script>'+ (ROOT/'mock_leaflet.js').read_text()+'</script>')
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'],headless=True)
        page=await browser.new_page(viewport={'width':1400,'height':800},device_scale_factor=1)
        errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        await page.route('**/*', lambda route:route.abort())
        await page.set_content(result, wait_until='domcontentloaded')
        print('REFERENCE_COUNT',await page.locator('#location-count').inner_text())
        print('LAYERS',await page.evaluate('Array.from(window.__TEST_MAP.layers).map(x=>x.children ? x.children.length : 0)'))
        print('INJECTED_PREDICTIONS',await page.evaluate("Array.from(window.__TEST_MAP.layers).flatMap(x=>x.children||[]).filter(x=>x.options?.icon?.html?.includes('pin pred')).length"))
        print('APP_RECTANGLES',await page.locator('.main').bounding_box(),await page.locator('#map').bounding_box(), await page.locator('.filters').bounding_box())
        assert (await page.locator('#map').bounding_box())['height']>250
        assert (await page.locator('.filters').bounding_box())['y']>(await page.locator('#map').bounding_box())['y']
        assert await page.evaluate('document.documentElement.scrollHeight<=document.documentElement.clientHeight+1')
        assert await page.evaluate("Array.from(window.__TEST_MAP.layers).flatMap(x=>x.children||[]).filter(x=>x.options?.icon?.html?.includes('pin pred')).length")==1
        await page.locator('#facility-type').select_option('Clinic')
        print('FILTERED_COUNT',await page.locator('#location-count').inner_text())
        await page.locator('#draw-circle').click()
        assert await page.evaluate('window.__DRAW_TRIGGERED')
        # Emulate the user drawing a 120-metre circle before asking for actual AI.
        await page.evaluate("window.__TEST_MAP.emit('draw:created',{layerType:'circle',layer:L.circle([45.43,-75.7],{radius:120})})")
        await page.locator('#question').fill('run AI in selected circle')
        await page.locator('#chat-form button').click()
        await page.wait_for_timeout(120)
        print('INFERENCE_ASSISTANT', (await page.locator('.msg').all_inner_texts())[-1])
        print('AI_POINT_COUNT', await page.evaluate("Array.from(window.__TEST_MAP.layers).flatMap(x=>x.children||[]).filter(x=>x.options?.icon?.html?.includes('pin pred')).length"))
        assert await page.evaluate("Array.from(window.__TEST_MAP.layers).flatMap(x=>x.children||[]).filter(x=>x.options?.icon?.html?.includes('pin pred')).length")==1
        assert 'AI completed' in (await page.locator('.msg').all_inner_texts())[-1]
        print('JAVASCRIPT_ERRORS',errors)
        assert not errors,errors
        await browser.close()

asyncio.run(main())
