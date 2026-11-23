"""Colab-native, multi-source road-map viewer of official Canadian healthcare reference records.
No Gradio service. This code and the notebook copy share the same template.
"""
from __future__ import annotations
import html
import json
import math
from pathlib import Path
import pandas as pd
from IPython.display import HTML, display

DATA_PATHS=(Path('/content/odhf_facilities.csv'),Path('/content/odhf_hospitals.csv'),Path('odhf_facilities.csv'),Path('odhf_hospitals.csv'))


def load_facilities():
    path=next((p for p in DATA_PATHS if p.is_file()),None)
    if path is None:
        return pd.DataFrame(columns=['name','latitude','longitude','type','emergency_services','ownership']), 'Reference dataset unavailable. The data setup cell did not complete.'
    data=pd.read_csv(path,low_memory=False).fillna('')
    for field in ('name','latitude','longitude'):
        if field not in data:
            raise ValueError(f'Reference file is missing required column {field}: {path}')
    for optional in ('id','type','address','province','emergency_services','ownership'):
        if optional not in data: data[optional]=''
    data['latitude']=pd.to_numeric(data['latitude'],errors='coerce')
    data['longitude']=pd.to_numeric(data['longitude'],errors='coerce')
    data=data[data.latitude.between(41,84)&data.longitude.between(-142,-52)].copy()
    data['type']=data['type'].astype(str).str.strip().replace('','Unspecified')
    return data,f'{len(data):,} reference records from {path}'


def normalize_records(data):
    if len(data)>40000:raise ValueError('Unexpectedly large reference dataset')
    result=[]
    for _,r in data.iterrows():
        try:
            lat,lon=float(r['latitude']),float(r['longitude'])
            if not (math.isfinite(lat) and math.isfinite(lon) and 41<=lat<=84 and -142<=lon<=-52):continue
        except (ValueError,TypeError):continue
        def val(field):
            x=str(r.get(field,'') or '').strip()
            return '' if x.lower() in ('nan','none','null') else x[:280]
        result.append({'key':str(len(result)),'name':val('name') or 'Unnamed facility','lat':lat,'lon':lon,
          'type':val('type') or 'Unspecified','address':val('address'),'province':val('province'),
          'er':val('emergency_services'),'ownership':val('ownership')})
    return result



DEFAULT_AI_PROJECT_DIR = Path('/content/drive/MyDrive/OSS4008_Hospital_AI_Pilot')


def load_verified_model_predictions(project_dir=DEFAULT_AI_PROJECT_DIR):
    """Load only predictions from the real, evaluated checkpoint matching reviewed data.

    A missing model is a normal state. Synthetic predictor results and ODHF
    reference locations NEVER become green model markers.
    """
    import hashlib
    root=Path(project_dir)
    report_path=root/'detector'/'training_report.json'
    weights=root/'detector'/'hospital_best.pt'
    geojson_path=root/'predictions'/'trained_hospital_test_predictions.geojson'
    split_path=root/'hospital_yolo'/'site_split_manifest.csv'
    review_path=root/'review_manifest.csv'
    if not all(f.is_file() for f in [report_path,weights,geojson_path,split_path,review_path]):
        return [], 'No evaluated hospital AI predictions yet; reference map only.'
    def digest(file):
        h=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
        return h.hexdigest()
    try:
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if (report.get('status')!='TRAINED_AND_EVALUATED' or
            report.get('weights_sha256')!=digest(weights) or
            report.get('split_manifest_sha256')!=digest(split_path) or
            report.get('source_manifest_sha256')!=digest(review_path)):
            return [], 'AI checkpoint or labels changed; predictions hidden pending re-evaluation.'
        geo=json.loads(geojson_path.read_text(encoding='utf-8'))
        if geo.get('type')!='FeatureCollection' or not isinstance(geo.get('features'),list):
            raise ValueError('Expected GeoJSON FeatureCollection')
        if len(geo['features'])>3000:
            raise ValueError('Prediction count exceeds map safety limit')
        output=[]
        for feature in geo['features']:
            if feature.get('geometry',{}).get('type')!='Point':
                raise ValueError('Only georeferenced detector Point features are supported')
            coords=feature['geometry'].get('coordinates')
            if not isinstance(coords,list) or len(coords)!=2:
                raise ValueError('Invalid geographic coordinates')
            lon,lat=map(float,coords)
            properties=feature.get('properties') or {}
            confidence=float(properties.get('confidence',float('nan')))
            if (not math.isfinite(lon) or not math.isfinite(lat) or not math.isfinite(confidence) or
                not (-142 <= lon <= -52 and 41 <= lat <= 84 and 0 <= confidence <= 1) or
                properties.get('method')!='trained_YOLO_hospital_building_detector' or
                properties.get('weights_sha256')!=report['weights_sha256']):
                raise ValueError('Prediction origin / coordinates / confidence not verified')
            output.append({'lat':lat,'lon':lon,'conf':confidence})
        return output, f'{len(output)} evaluated model detections available on the map.'
    except (OSError,ValueError,KeyError,TypeError) as e:
        return [], 'Unverified predictions suppressed: '+str(e)[:180]


def load_evaluated_model_summary(project_dir=DEFAULT_AI_PROJECT_DIR):
    """Read genuine held-out detector metrics only when model/dataset hashes match.

    Absence of training is an expected, honest status; no fictional numbers.
    These are object-detection metrics, not a hospital identification accuracy rate.
    """
    import hashlib
    root = Path(project_dir)
    report_file = root/'detector'/'training_report.json'
    weights = root/'detector'/'hospital_best.pt'
    split = root/'hospital_yolo'/'site_split_manifest.csv'
    manifest = root/'review_manifest.csv'
    missing = [x for x in (report_file, weights, split, manifest) if not x.is_file()]
    if missing:
        return {'ready': False, 'status': 'Not trained — review hospital roofs first',
                'metrics': None, 'test_sites': None}
    try:
        def digest(path):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1048576), b''):
                    h.update(chunk)
            return h.hexdigest()
        report = json.loads(report_file.read_text(encoding='utf-8'))
        if (report.get('status') != 'TRAINED_AND_EVALUATED' or
            report.get('weights_sha256') != digest(weights) or
            report.get('split_manifest_sha256') != digest(split) or
            report.get('source_manifest_sha256') != digest(manifest)):
            raise ValueError('Checkpoint or reviewed dataset does not match the evaluation')
        values = report.get('test_metrics')
        required = ('precision','recall','mAP50','mAP50_95')
        if not isinstance(values,dict) or any(k not in values for k in required):
            raise ValueError('Held-out evaluation metrics not recorded')
        metrics = {k:float(values[k]) for k in required}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in metrics.values()):
            raise ValueError('Invalid held-out metric value')
        splits = pd.read_csv(split, keep_default_na=False)
        if not {'split','site_id'}.issubset(splits.columns):
            raise ValueError('Test-site provenance is incomplete')
        count = int(splits.loc[splits['split'].eq('test'),'site_id'].nunique())
        if count < 1:
            raise ValueError('No independent test site found')
        return {'ready': True, 'status': 'Evaluated hospital-building detector',
                'metrics': metrics, 'test_sites': count,
                'scope': 'Held-out test sites; rooftop candidates only, not nationwide clinical-site accuracy'}
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        return {'ready': False, 'status': 'AI evaluation unavailable: '+str(exc)[:140],
                'metrics': None, 'test_sites': None}

def build_dashboard_html(data,status=None,model_predictions=None,evaluation=None):
    status=status or {'label':'Canadian healthcare reference data','message':''}
    payload=json.dumps(normalize_records(data),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    source=json.dumps(status,ensure_ascii=False,separators=(',',':'))
    evaluation_json=json.dumps(evaluation or {'ready':False,'status':'AI detector not trained','metrics':None,'test_sites':None},ensure_ascii=False,separators=(',',':'),allow_nan=False)
    def safe_json(x):return x.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    rendered=(UI_TEMPLATE.replace('__FACILITY_DATA_JSON__',safe_json(payload))
                         .replace('__DATASET_STATUS_JSON__',safe_json(source))
                         .replace('__MODEL_EVALUATION_JSON__',safe_json(evaluation_json)))
    if model_predictions:
        # Preload only model-verified candidates into the existing green marker layer.
        signature='let importedPredictions=[];'
        if rendered.count(signature)!=1:
            raise RuntimeError('Map overlay integration signature changed; cannot preload safely')
        sanitized=[]
        for point in model_predictions:
            lat,lon,conf=float(point['lat']),float(point['lon']),float(point['conf'])
            if not all(math.isfinite(v) for v in (lat,lon,conf)) or not (-142<=lon<=-52 and 41<=lat<=84 and 0<=conf<=1):
                raise ValueError('Invalid AI prediction location/confidence')
            sanitized.append({'lat':lat,'lon':lon,'conf':conf})
        rendered=rendered.replace(signature,'let importedPredictions='+safe_json(json.dumps(sanitized,allow_nan=False,separators=(',',':')))+';',1)
    # UI owns Run AI/chat commands directly; do not inject handlers twice.
    return rendered


def launch_colab_dashboard(screen_margin_px=65):
    data,message=load_facilities()
    source_file=Path('/content/odhf_facilities.source.json') if Path('/content').exists() else Path('odhf_facilities.source.json')
    source={'label':'Statistics Canada references','message':message}
    if source_file.is_file():
        try:
            meta=json.loads(source_file.read_text())
            source['label']=meta.get('source_version','Statistics Canada references')
        except (OSError,ValueError):pass
    print('Dashboard source:',source['label'],'|',message)
    predictions, prediction_status=load_verified_model_predictions()
    evaluation=load_evaluated_model_summary()
    print('AI overlay:', prediction_status)
    print('AI evaluation:', evaluation['status'])
    output=build_dashboard_html(data,source,model_predictions=predictions,evaluation=evaluation)
    if 'evaluated model detections available' in prediction_status:
        output=output.replace(
            'Image-based hospital detection is planned, but no trained detector is connected.',
            'An evaluated hospital-building detector is connected for the Ontario imagery pilot. Draw a small circle and type run AI to request new inference. Green markers are unverified model candidates, never confirmed hospital identity.')
    # Smaller screen margin => taller dashboard; 65px fits the approved Colab canvas.
    screen_margin_px = int(screen_margin_px)
    if not 40 <= screen_margin_px <= 240:
        raise ValueError("screen_margin_px must be between 40 and 240")
    markup=('<iframe title="Hospital Infrastructure Canada dashboard" '
            f'style="display:block;width:100%;height:calc(100dvh - {screen_margin_px}px);min-height:480px;border:0;border-radius:10px" '
            'sandbox="allow-scripts allow-same-origin allow-forms" '
            'referrerpolicy="strict-origin-when-cross-origin" srcdoc="'+html.escape(output,quote=True)+'"></iframe>')
    display(HTML(markup))
    return {'records_loaded':len(data),'data_source':source['label'],'ui':'Notebook-native Leaflet (no Gradio)'}

UI_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="referrer" content="strict-origin-when-cross-origin"/>
<title>Hospital Infrastructure Canada</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.9.0/dist/maplibre-gl.css"/>
<style>

:root{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#203248;background:#f4f7fb;font-size:13px}
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden}
body{background:#f8fafc}
button,input,select{font:inherit}
button{cursor:pointer}
.page{width:100%;height:100%;min-height:0;display:flex;flex-direction:column;padding:12px 14px;gap:9px;background:#f8fafc}
.header{display:flex;justify-content:space-between;align-items:center;gap:8px;flex:0 0 auto;min-height:33px}
.header-title{font-size:18px;font-weight:750;line-height:1.15;letter-spacing:-.25px}
.subtitle,.muted{font-size:10px;color:#657387}
.tag{border:1px solid #dce3ea;border-radius:14px;color:#5b6980;font-size:10px;padding:4px 8px;white-space:nowrap}
.main{order:1;flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1.9fr) minmax(255px,.86fr);gap:12px}
.panel{min-width:0;min-height:0;border:1px solid #dce3ea;border-radius:11px;overflow:hidden;background:#fff}
.main>.panel:first-child{display:flex;flex-direction:column}
.searchbar,.toolbar{display:flex;align-items:center;gap:6px;padding:6px 8px;border-bottom:1px solid #e2e7ed;flex:0 0 auto}
.searchbar input{flex:1;min-width:0}
input,select{border:1px solid #d9e1e8;border-radius:6px;background:#fff;color:#253449;padding:0 8px;height:30px;width:100%;font-size:11px}
.btn{border:1px solid #d9e1e8;border-radius:6px;padding:6px 8px;background:#f6f8fa;color:#2c415a;font-weight:650;font-size:10px;white-space:nowrap}
.btn:hover{background:#eaf1ff}.btn.blue{background:#295fd8;color:#fff;border-color:#295fd8}.btn.active{background:#e0eafb;color:#1f56c2;border-color:#6792f0}
.toolbar{min-height:37px;flex-wrap:wrap}.toolbar .selection{font-size:10px;color:#245ab6;font-weight:650;margin-left:auto}
#map{position:relative;flex:1 1 auto;min-height:255px;width:100%;background:#eaf0e8;z-index:1}
.map-footer{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px 8px;color:#69768b;font-size:10px;flex:0 0 auto;min-height:28px}
.assistant{display:flex;flex-direction:column;height:100%}
.assistant-title{flex:0 0 auto;min-height:39px;padding:6px 9px;border-bottom:1px solid #e2e7ed;display:flex;align-items:center;justify-content:space-between;gap:8px}.assistant-title strong{font-size:12px}
.assistant-body{flex:1;min-height:0;overflow:auto;padding:8px;display:flex;flex-direction:column;gap:8px;background:#fff}
.msg{max-width:94%;font-size:11px;line-height:1.45;padding:8px;border:1px solid #e0e6ef;border-radius:9px 9px 9px 3px;background:#f7f9fc;white-space:pre-wrap}
.msg.user{align-self:flex-end;background:#e8efff;color:#274271;border-color:#cbd9f7;border-radius:9px 9px 3px 9px}
 .nav-result-list{margin-top:7px;display:flex;flex-direction:column;gap:5px}.nav-result{display:flex;align-items:center;justify-content:space-between;gap:8px;border-top:1px solid #dee5ec;padding-top:5px}.nav-result span{min-width:0;flex:1}.nav-result button{flex:0 0 auto}
.prompts{display:flex;flex-direction:column;gap:5px;margin-top:7px}.prompts button{border:1px solid #dce3ea;border-radius:6px;background:#fff;color:#374c66;font-size:10px;padding:6px;text-align:left}
.ask{display:flex;gap:5px;padding:6px;border-top:1px solid #e2e7ed;flex:0 0 auto}.ask input{min-width:0;flex:1}.ask button{border:0;background:#295fd8;color:#fff;border-radius:6px;padding:0 12px}
.filters{order:2;border:1px solid #dce3ea;border-radius:10px;overflow:visible;background:#fff;flex:0 0 auto}
.heading{padding:5px 9px;display:flex;justify-content:space-between;align-items:center;gap:8px;border-bottom:1px solid #e2e7ed}.heading strong{font-size:11px}
.filter-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:6px 9px}
label{display:block;font-weight:650;font-size:10px;color:#657387;margin-bottom:4px}select:disabled{opacity:.48;cursor:not-allowed}
.filter-foot{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:0 9px 5px}
.ai-state{font-size:10px;color:#50617b;font-weight:600}.ai-state.ready{color:#167b56}.filter-foot label{color:#364d68;display:flex;align-items:center;gap:5px;margin:0;font-weight:500;font-size:10px}.filter-foot input[type=checkbox]{width:13px;height:13px;accent-color:#295fd8}.filter-foot .muted{flex:1;font-size:9px}
.advanced{padding:4px 8px;border-top:1px solid #eef1f4}.advanced summary{cursor:pointer;color:#445d7c;font-size:10px;font-weight:600}.advanced input{margin-top:7px;padding:7px;height:auto}
#result-rows,.results{display:none!important}
.notice{order:3;flex:0 0 auto;font-size:10px;color:#596a83;margin:0;padding:1px 3px;max-height:22px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.notice.error{color:#aa3333}
.pin{display:block;width:15px;height:15px;border:2px solid #fff;border-radius:50%;background:#cf4949;box-shadow:0 1px 5px #0005}.pin.clinic{background:#d3944a}.pin.other{background:#718aa0}.pin.pred{background:#1e9e6d}
.leaflet-popup-content{font-size:12px;line-height:1.5}.leaflet-control-attribution{font-size:9px!important}
@media(max-width:740px){html,body{overflow:auto;height:auto}.page{height:auto;min-height:700px}.main{grid-template-columns:1fr;min-height:590px}.main>.panel:first-child{min-height:350px}#map{min-height:255px}.assistant{height:230px}.filter-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.toolbar .selection{margin-left:0}}
</style>
</head>
<body>
<div class="page">
<div class="header">
 <div><div class="header-title">Hospital Infrastructure Canada</div><div class="subtitle">Remote sensing · Facility explorer · OSS 4008</div></div>
 <span class="tag">Interactive research map</span>
</div>
<div class="main">
 <section class="panel">
  <form class="searchbar" id="search-form"><span aria-hidden="true">⌕</span><input id="search" aria-label="Search Canadian city" value="Ottawa" placeholder="Search any Canadian city or town"/><button class="btn blue" type="submit" id="search-btn">Search</button></form>
  <div class="toolbar"><button class="btn" id="draw-circle" type="button">◯ Draw circle</button><button class="btn" id="clear-area" type="button">Clear selection</button><button class="btn" id="reset-view" type="button">⌖ Reset view</button><button class="btn blue" id="run-ai" type="button" title="Scan small drawn area or hospital-reference image sites across the selected city">✦ Run AI</button><button class="btn" id="trace-roofs" type="button" title="Trace real map building polygons near a selected pin or drawn circle; NOT confirmed hospital buildings">Trace large buildings</button><select id="trace-radius" aria-label="Tracing distance from hospital pin" title="Distance from red reference pin" style="width:72px;height:27px;padding:2px;font-size:10px"><option value="100">100 m</option><option value="180" selected>180 m</option><option value="250">250 m</option><option value="350">350 m</option></select><select id="trace-area" aria-label="Minimum mapped building area" title="Minimum building area, not roof height" style="width:103px;height:27px;padding:2px;font-size:10px"><option value="500">500 m²+</option><option selected value="900">900 m²+</option><option value="1500">1500 m²+</option><option value="2500">2500 m²+</option></select><span class="selection" id="selection">25 km study radius</span></div>
  <div id="map" aria-label="Interactive map showing Statistics Canada reference healthcare facilities"></div>
  <div class="map-footer"><span><span id="location-label">Ottawa, Ontario</span> · <span id="map-provider">Loading map…</span></span><span id="location-count">Loading references…</span></div>
 </section>
 <section class="panel assistant">
  <div class="assistant-title"><div><strong>✦ Map assistant</strong><div class="subtitle">Hospital map expert · reference-grounded answers</div></div><span class="tag">Grounded references + local LLM</span></div>
  <div class="assistant-body" id="messages"><div class="msg">Search a city or draw a circle to explore Canadian healthcare facilities. I can help navigate the map and count known reference sites.<div class="prompts"><button data-prompt="Show hospitals in Ottawa" type="button">Show hospitals in Ottawa</button><button data-prompt="How many facilities are in my selected area?" type="button">Count selected facilities</button><button data-prompt="Can AI identify hospitals from satellite images?" type="button">Explain AI detection</button></div></div></div>
  <form class="ask" id="chat-form"><input id="question" placeholder="Ask about a hospital or selected area…" aria-label="Ask map assistant"/><button type="submit" aria-label="Send message">➤</button></form>
 </section>
</div>
<section class="filters">
 <div class="heading"><strong>Filters &amp; map layers</strong><span class="muted" id="data-state">Source: waiting for data</span></div>
 <div class="filter-grid">
  <div><label for="facility-type">Facility type</label><select id="facility-type"><option value="all">All facility types</option></select></div>
  <div><label for="study-radius">Area radius</label><select id="study-radius"><option value="5">5 km</option><option value="10">10 km</option><option selected value="25">25 km</option><option value="50">50 km</option><option value="100">100 km</option></select></div>
 </div>
 <div class="filter-foot"><label><input type="checkbox" checked id="show-refs"/> Known facilities</label><label><input type="checkbox" checked id="show-pred"/> AI detections <span id="ai-count">(0)</span></label><span class="ai-state" id="ai-state">AI not trained</span><button class="btn" id="reset-filters">Reset filters</button></div>

</section>
<div hidden id="results-count"></div><div hidden id="area-description"></div><div hidden id="result-rows"></div>
<div class="notice" id="source-note">Red = official facility reference · Green = experimental AI bounding boxes · Amber = mapped building candidates (not verified hospitals).</div>
<script type="application/json" id="facility-data">__FACILITY_DATA_JSON__</script><script type="application/json" id="dataset-status">__DATASET_STATUS_JSON__</script><script type="application/json" id="model-evaluation">__MODEL_EVALUATION_JSON__</script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script src="https://unpkg.com/maplibre-gl@5.9.0/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/@maplibre/maplibre-gl-leaflet@0.1.4/leaflet-maplibre-gl.js"></script>
<script>
(function(){'use strict';
 const $=id=>document.getElementById(id);
 const records=JSON.parse($('facility-data').textContent);
 const datasetStatus=JSON.parse($('dataset-status').textContent);
 const evaluation=JSON.parse($('model-evaluation').textContent);
 let experimentalReady=false;
 let selectedFacilityKey=null;
 const centerFallback={lat:45.4215,lon:-75.6972,name:'Ottawa, Ontario'};
 const commonCities={ottawa:[45.4215,-75.6972,'Ottawa, Ontario'],halifax:[44.6488,-63.5752,'Halifax, Nova Scotia'],toronto:[43.6532,-79.3832,'Toronto, Ontario'],vancouver:[49.2827,-123.1207,'Vancouver, British Columbia'],montreal:[45.5017,-73.5673,'Montréal, Quebec'],calgary:[51.0447,-114.0719,'Calgary, Alberta'],edmonton:[53.5461,-113.4938,'Edmonton, Alberta'],winnipeg:[49.8951,-97.1384,'Winnipeg, Manitoba']};
 let city={...centerFallback};let selectedCircle=null;let selected=[];let importedPredictions=[];let drawing=false;let busy=false;
 const typeSelect=$('facility-type');
 if(!window.L){$('map').textContent='Map tiles could not load. Check network access to the Leaflet CDN.';return;}
 const map=L.map('map',{zoomControl:false,scrollWheelZoom:true,preferCanvas:true}).setView([city.lat,city.lon],11);
 L.control.zoom({position:'topright'}).addTo(map);
 // OpenFreeMap is a keyless vector basemap. Leaflet keeps markers and drawing on top.
 // Use live raster fallbacks when the MapLibre bridge, vector service or WebGL fails.
 const esriStreets=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Basemap © Esri, HERE, Garmin, OpenStreetMap contributors'});
 const esriImagery=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Imagery © Esri, Vantor, Earthstar Geographics, GIS User Community'});
 const osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'});
 let vector=null;
 try {if(window.maplibregl && L.maplibreGL && maplibregl.supported()) vector=L.maplibreGL({style:'https://tiles.openfreemap.org/styles/liberty',interactive:false});}
 catch(err){console.warn('Vector basemap unsupported:',err);}
 const providerOptions=vector?{'Streets · OpenFreeMap':vector,'Satellite · Esri':esriImagery,'Streets · Esri (legacy)':esriStreets,'Streets · OpenStreetMap':osm}
   :{'Satellite · Esri':esriImagery,'Streets · Esri (legacy)':esriStreets,'Streets · OpenStreetMap':osm};
 const providers=[...Object.entries(providerOptions)];
 let currentBase=null,baseReady=false,baseChoice=0;
 const failures=new Map();
 function activateBase(index, explanation='') {
   if(index>=providers.length){$('map-provider').textContent='Map background unavailable — check internet/CDN access';return;}
   baseChoice=index;
   const [name,layer]=providers[index];
   if(currentBase && map.hasLayer(currentBase)){map.removeLayer(currentBase);}
   currentBase=layer; baseReady=false;
   $('map-provider').textContent=(explanation?explanation+' · ':'')+'Loading '+name;
   try{layer.addTo(map);}catch(err){console.warn('Map layer unavailable',name,err);activateBase(index+1,'Fallback');}
 }
 for(const [i,p] of providers.entries()) {
   const [name,layer]=p;
   if(layer instanceof L.TileLayer){
     layer.on('tileload',()=>{if(currentBase===layer){baseReady=true;$('map-provider').textContent=name;}});
     layer.on('tileerror',()=>{if(currentBase!==layer||baseReady)return;
        const count=(failures.get(name)||0)+1;failures.set(name,count);
        if(count>=3)activateBase(i+1,'Map source blocked');
     });
   }
 }
 activateBase(0);
 if(vector){
   try {const ml=vector.getMaplibreMap();
       ml.on('load',()=>{if(currentBase===vector){baseReady=true;$('map-provider').textContent='Streets · OpenFreeMap';}});
       ml.on('error',(event)=>{if(currentBase===vector && !baseReady){console.warn('OpenFreeMap load error',event);activateBase(1,'Fallback');}});
   } catch(err) {activateBase(1,'Fallback');}
   // If a vector style silently fails to load, switch to a real raster map.
   setTimeout(()=>{if(currentBase===vector && !baseReady)activateBase(1,'Fallback');},9000);
 }
 // Leaflet layer selector permits users to choose streets or satellite imagery.
 L.control.layers(providerOptions,null,{position:'bottomright',collapsed:true}).addTo(map);
 map.on('baselayerchange',(event)=>{
   const index=providers.findIndex(([name,layer])=>layer===event.layer);
   if(index>=0){currentBase=event.layer;baseChoice=index;baseReady=false;failures.set(providers[index][0],0);$('map-provider').textContent='Loading '+providers[index][0];}
 });
 const studyCircle=L.circle([city.lat,city.lon],{radius:25000,color:'#2563db',weight:1.7,fillColor:'#2563db',fillOpacity:.045,dashArray:'6 4'}).addTo(map);
 const drawnGroup=L.featureGroup().addTo(map);
 const cluster=(L.markerClusterGroup?L.markerClusterGroup({chunkedLoading:true,showCoverageOnHover:false,maxClusterRadius:45}):L.layerGroup()).addTo(map);
 const predictionLayer=L.layerGroup().addTo(map);
 const mappedFootprintLayer=L.featureGroup().addTo(map);
 const availableTypes=Array.from(new Set(records.map(r=>r.type||'Unspecified'))).sort((a,b)=>a.localeCompare(b));
 for(const type of availableTypes){const opt=document.createElement('option');opt.value=type;opt.textContent=type;typeSelect.appendChild(opt);}
 const defaultHospital=availableTypes.find(t=>/^hospitals?$/i.test(t.trim()));
 if(defaultHospital)typeSelect.value=defaultHospital;
 function esc(value){const span=document.createElement('span');span.textContent=String(value??'');return span.innerHTML;}
 const icon=(kind,pred=false)=>L.divIcon({className:'',html:'<span class="pin '+(pred?'pred':kind.includes('hospital')?'':kind.includes('clinic')?'clinic':'other')+'"></span>',iconSize:[18,18],iconAnchor:[9,9]});
 function activeArea(){if(selectedCircle)return {center:selectedCircle.getLatLng(),radius:selectedCircle.getRadius()};return {center:L.latLng(city.lat,city.lon),radius:Number($('study-radius').value)*1000};}
 function getMatches(){const a=activeArea();const kind=typeSelect.value;
 return records.filter(r=>Number.isFinite(r.lat)&&Number.isFinite(r.lon)&&map.distance(a.center,[r.lat,r.lon])<=a.radius&&
 (kind==='all'||r.type===kind))
 .map(r=>({...r,dist:map.distance(a.center,[r.lat,r.lon])/1000})).sort((a,b)=>a.dist-b.dist);
 }
 function updateAreaOverlay(){if(selectedCircle){map.removeLayer(studyCircle);$('selection').textContent='Drawn circle: '+(selectedCircle.getRadius()/1000).toFixed(1)+' km';}
 else {const a=activeArea();studyCircle.setLatLng(a.center).setRadius(a.radius);if(!map.hasLayer(studyCircle))studyCircle.addTo(map);$('selection').textContent=Number($('study-radius').value)+' km study radius';}}
 function popupContent(r){const div=document.createElement('div');div.innerHTML='<strong>'+esc(r.name)+'</strong><div>'+esc(r.type)+'</div>'+
 (r.address?'<div>'+esc(r.address)+'</div>':'')+(r.ownership?'<div>Owner: '+esc(r.ownership)+'</div>':'')+
 '<div style="font-size:10px;color:#777;margin-top:5px">Statistics Canada reference — NOT an AI detection</div>';
 const ask=document.createElement('button');ask.type='button';ask.className='btn';
 ask.textContent='Ask about this facility';ask.style.marginTop='7px';
 ask.onclick=()=>{selectedFacilityKey=r.key;addChat('Selected '+r.name+'. Ask a question in the map assistant about this official reference.');};
 div.appendChild(ask);
 if((r.type||'').toLowerCase()==='hospital'){
   const roof=document.createElement('button');roof.type='button';roof.className='btn';roof.style.marginLeft='5px';
   roof.textContent='Trace large buildings';roof.title='Show mapped building footprints around this red hospital reference, not AI detections';
   roof.onclick=()=>{selectedFacilityKey=r.key;showMappedFootprint({key:r.key,radius_m:Number($('trace-radius').value),minimum_area_m2:Number($('trace-area').value)});};div.appendChild(roof);
 }
 return div;}
 function locate(r){map.flyTo([r.lat,r.lon],Math.max(map.getZoom(),14),{duration:.55});const m=markersById.get(r.key);if(m&&cluster.zoomToShowLayer)cluster.zoomToShowLayer(m,()=>m.openPopup());else if(m)m.openPopup();}
 const markersById=new Map();
 function navigateToReference(key){
   // Only local, loaded ODHF references are eligible; never trust LLM-provided coordinates.
   const target=records.find(x=>x.key===String(key) && (x.type||'').toLowerCase()==='hospital');
   if(!target){addChat('This hospital reference is not available in the current dataset.');return;}
   city={lat:target.lat,lon:target.lon,name:target.name};
   selectedCircle=null;drawnGroup.clearLayers();selectedFacilityKey=target.key;
   $('show-refs').checked=true;typeSelect.value=defaultHospital||'all';
   $('study-radius').value='5';
   $('search').value=target.name;
   // Navigating only moves the map; it does not automatically start a costly AI scan.
   map.flyTo([target.lat,target.lon],16,{duration:.65});refresh();
   const mark=markersById.get(target.key);
   if(mark&&cluster.zoomToShowLayer)cluster.zoomToShowLayer(mark,()=>mark.openPopup());
   else if(mark)mark.openPopup();
   addChat('Map moved to '+target.name+' ('+(target.address||'official reference location')+'). This is an ODHF reference pin, not a verified AI-detected roof.');
 }
 async function showMappedFootprint(request){
   let kernel=null;
   try{kernel=window.google?.colab?.kernel||window.parent?.google?.colab?.kernel||null;}catch(_e){kernel=window.google?.colab?.kernel||null;}
   if(!kernel||typeof kernel.invokeFunction!=='function'){
     addChat('Building tracing needs a running Colab notebook.');return;
   }
   const radius=Number(request?.radius_m);
   if(!Number.isFinite(radius)||radius<20||radius>350){
     addChat('Draw a circle between 20 and 350 metres, or choose a red hospital pin (180-metre search). Map tile squares are not distance units.');return;
   }
   let title='selected area';
   if(request.key!==undefined){
      const ref=records.find(x=>x.key===String(request.key)&&(x.type||'').toLowerCase()==='hospital');
      if(!ref){addChat('Select a red official hospital pin or draw a circle.');return;}
      title=ref.name;
   }else{
      const lat=Number(request.lat),lon=Number(request.lon);
      if(!Number.isFinite(lat)||!Number.isFinite(lon)||lat<41||lat>84||lon< -142||lon> -52){
        addChat('The drawn area must be inside Canada.');return;
      }
   }
   mappedFootprintLayer.clearLayers();
   addChat('Tracing mapped large buildings within '+Math.round(radius)+' m of '+title+'… These are map geometry, not AI-labelled hospitals.');
   const minArea=Number(request.minimum_area_m2||$('trace-area').value);
   if(!Number.isFinite(minArea)||minArea<200||minArea>5000){addChat('Choose a building size between 200 and 5000 m².');return;}
   const [args,kwargs]=hostArgumentEnvelope(kernel,{...request,minimum_area_m2:minArea,mode:'nearby'});
   try{
     const raw=await kernel.invokeFunction('oss4008.hospital_footprint',args,kwargs);
     const data=raw?.data?.['application/json'];
     if(!data?.ok){addChat('Building tracing unavailable: '+String(data?.error||'unknown data source error').slice(0,190));return;}
     let n=0,explicit=0;
     for(const f of (data.features||[]).slice(0,40)){
       if(f.geometry?.type!=='Polygon')continue;
       const rings=f.geometry.coordinates;
       if(!Array.isArray(rings)||!rings.length||rings.length>30)continue;
       if(!rings.every(r=>Array.isArray(r)&&r.length>=4&&r.length<=1500&&r.every(p=>Array.isArray(p)&&p.length===2&&Number.isFinite(Number(p[0]))&&Number.isFinite(Number(p[1]))&&p[0]>=-142&&p[0]<=-52&&p[1]>=41&&p[1]<=84)))continue;
       const props=f.properties||{};
       const isHospitalClass=props.evidence==='overture_class_hospital'||props.evidence==='osm_hospital_tag';
       const layer=L.geoJSON(f.geometry,{style:{color:isHospitalClass?'#be6c09':'#cf9522',weight:3,dashArray:'7 4',fillColor:'#fbbf24',fillOpacity:.08}});
       const note='MAP-DERIVED BUILDING OUTLINE — NOT VERIFIED HOSPITAL OR AI DETECTION'+
         (props.area_m2?' · '+Math.round(Number(props.area_m2))+' m²':'')+
         (props.distance_from_search_centre_m!==undefined?' · '+Math.round(Number(props.distance_from_search_centre_m))+' m from centre':'')+
         (isHospitalClass?' · map class tagged hospital':' · large nearby building')+
         ' · Source: '+String(props.source||data.source||'mapped building data').slice(0,100);
       layer.bindPopup(note).addTo(mappedFootprintLayer);n++;if(isHospitalClass)explicit++;
     }
     if(n&&mappedFootprintLayer.getBounds?.().isValid?.()){
       map.fitBounds(mappedFootprintLayer.getBounds(),{maxZoom:18,padding:[20,20]});
     }
     const from=String((data.sources_attempted||[]).join(' + ')||'no available source');
     const issues=(data.provider_errors||[]).map(x=>String(x).slice(0,110)).join('; ');
     addChat(n?n+' amber mapped building outlines ('+explicit+' explicitly tagged hospital). Sources: '+from+'. These are proximity candidates, NOT validated hospital buildings.'+(issues?' Partial source issue: '+issues:''):
       'No mapped building met the '+Math.round(minArea)+' m² minimum within '+Math.round(radius)+' m. Sources: '+from+'. '+(issues?'Service errors: '+issues:'Try a larger radius or a lower size threshold. The map may lack building geometry.')); 
   }catch(err){addChat('Building tracing failed: '+String(err).slice(0,200));}
 }
 async function traceRoofsFromSelection(){
   if(selectedCircle){
      const c=selectedCircle.getLatLng();
      return showMappedFootprint({lat:c.lat,lon:c.lng,radius_m:selectedCircle.getRadius(),minimum_area_m2:Number($('trace-area').value)});
   }
   if(selectedFacilityKey!==null){
      return showMappedFootprint({key:selectedFacilityKey,radius_m:Number($('trace-radius').value),minimum_area_m2:Number($('trace-area').value)});
   }
   addChat('For tracing, draw a 20–350 m circle or click a red hospital marker and select Trace large buildings. Satellite tile squares do not represent metres.');
 }
 $('trace-roofs').onclick=traceRoofsFromSelection;

 function appendNavigation(message,items){
   if(!Array.isArray(items)||!items.length)return;
   // A dedicated, readable Navigate control for each official hospital result.
   const panel=document.createElement('div');panel.className='nav-result-list';
   for(const item of items.slice(0,5)){
     const r=records.find(x=>x.key===String(item.key) && (x.type||'').toLowerCase()==='hospital');
     if(!r)continue;
     const row=document.createElement('div');row.className='nav-result';
     const label=document.createElement('span');label.textContent=r.name+' · '+Number(item.distance_km||0).toFixed(1)+' km (straight-line)';
     const btn=document.createElement('button');btn.type='button';btn.className='btn';
     btn.textContent='Navigate';btn.setAttribute('aria-label','Navigate map to '+r.name);
     btn.onclick=()=>navigateToReference(r.key);
     row.append(label,btn);panel.appendChild(row);
   }
   if(panel.childElementCount)message.appendChild(panel);
 }

 function refresh(){updateAreaOverlay();selected=getMatches();cluster.clearLayers();markersById.clear();
 if($('show-refs').checked){for(const r of selected.slice(0,1800)){const m=L.marker([r.lat,r.lon],{icon:icon(r.type.toLowerCase())}).bindPopup(popupContent(r));cluster.addLayer(m);markersById.set(r.key,m);}}
 const area=activeArea();$('location-label').textContent=city.name;$('location-count').textContent=records.length?selected.length.toLocaleString()+' matching reference facilities':'Reference data unavailable';
 $('results-count').textContent=selected.length.toLocaleString()+' results';$('area-description').textContent=selectedCircle?'Within drawn circle ('+(area.radius/1000).toFixed(1)+' km)':('Within '+(area.radius/1000).toFixed(0)+' km of '+city.name);
 const list=$('result-rows');list.replaceChildren();
 if(!selected.length){const none=document.createElement('div');none.className='result-line';none.textContent=records.length?'No reference facilities match these filters in the selected area.':'No reference records loaded. Check the dataset preparation cell above.';list.appendChild(none);}
 for(const r of selected.slice(0,12)){const line=document.createElement('div');line.className='result-line';const info=document.createElement('div');const name=document.createElement('div');name.className='name';name.textContent=r.name;const sub=document.createElement('div');sub.className='info';sub.textContent=r.type+' · '+r.dist.toFixed(1)+' km';info.append(name,sub);const btn=document.createElement('button');btn.textContent='Locate';btn.onclick=()=>locate(r);line.append(info,btn);list.append(line);}
 if(selected.length>12){const extra=document.createElement('div');extra.className='result-line muted';extra.textContent='Showing nearest 12 records. All '+selected.length.toLocaleString()+' matching markers are included (up to 1,800 markers).';list.append(extra);}
 renderPredictions();}
 let predictionOrigin=evaluation.ready && importedPredictions.length?'evaluated':'none';
 function candidateGeometry(feature){
   // Accept actual georeferenced model bbox polygons and legacy point predictions.
   if(feature && feature.geometry?.type==='Polygon'){
     const ring=feature.geometry.coordinates?.[0];
     if(!Array.isArray(ring)||ring.length<4)return null;
     const coords=ring.map(p=>({lon:Number(p[0]),lat:Number(p[1])}));
     if(!coords.every(p=>Number.isFinite(p.lat)&&Number.isFinite(p.lon)&&p.lat>=41&&p.lat<=84&&p.lon>=-142&&p.lon<=-52))return null;
     const center=coords.slice(0,-1).reduce((acc,p)=>({lat:acc.lat+p.lat/(coords.length-1),lon:acc.lon+p.lon/(coords.length-1)}),{lat:0,lon:0});
     const conf=Number(feature.properties?.confidence);
     if(!Number.isFinite(conf)||conf<0||conf>1)return null;
     return {kind:'polygon',lat:center.lat,lon:center.lon,conf,ring:coords.map(p=>[p.lat,p.lon]),
             name:String(feature.properties?.source_name||'').slice(0,140)};
   }
   if(feature && feature.geometry?.type==='Point'){
     const pair=feature.geometry.coordinates;
     if(!Array.isArray(pair)||pair.length!==2)return null;
     const lat=Number(pair[1]),lon=Number(pair[0]),conf=Number(feature.properties?.confidence);
     if(!Number.isFinite(lat)||!Number.isFinite(lon)||!Number.isFinite(conf)||conf<0||conf>1)return null;
     return {kind:'point',lat,lon,conf};
   }
   return null;
 }
 function renderPredictions(){
   predictionLayer.clearLayers();const a=activeArea();let inArea=0;
   for(const p of importedPredictions){
     if(map.distance(a.center,[p.lat,p.lon])>a.radius)continue;
     inArea++;
     if(!$('show-pred').checked)continue;
     const info=(predictionOrigin==='evaluated'?'Evaluated roof candidate':'EXPERIMENTAL candidate, NOT verified hospital')+
       (p.conf===null?'':' · detector confidence '+p.conf.toFixed(2))+
       (p.name?' · near '+p.name:'')+
       (p.kind==='polygon'?' · MODEL BOUNDING RECTANGLE (not exact roof outline)':' · model point');
     if(p.kind==='polygon'){
       const shape=L.polygon(p.ring,{color:'#00b77b',weight:3,opacity:1,fillColor:'#10b981',fillOpacity:.16});
       shape.bindPopup(info);shape.addTo(predictionLayer);
       // At city scale small roofs are subpixel; make the locations discoverable.
       if(map.getZoom()<14)L.circleMarker([p.lat,p.lon],{radius:4,color:'#fff',weight:1.5,fillColor:'#00aa76',fillOpacity:.85}).bindPopup(info).addTo(predictionLayer);
     }else L.marker([p.lat,p.lon],{icon:icon('',true)}).bindPopup(info).addTo(predictionLayer);
   }
   $('ai-count').textContent='('+inArea+')';
 }
 map.on('zoomend',renderPredictions);

 function setDrawing(active){drawing=active;$('draw-circle').classList.toggle('active',active);}
 $('draw-circle').onclick=()=>{if(!L.Draw||!L.Draw.Circle){addChat('Circle-drawing library did not load. Check network access.');return;}setDrawing(true);new L.Draw.Circle(map,{shapeOptions:{color:'#2861d8',weight:2,fillOpacity:.1}}).enable();};
 map.on('draw:created',e=>{if(e.layerType!=='circle')return;drawnGroup.clearLayers();selectedCircle=e.layer;drawnGroup.addLayer(selectedCircle);setDrawing(false);refresh();});
 map.on('draw:drawstop',()=>setDrawing(false));
 $('clear-area').onclick=()=>{selectedCircle=null;drawnGroup.clearLayers();refresh();};
 $('reset-view').onclick=()=>{map.setView([city.lat,city.lon],11);};
 $('study-radius').onchange=refresh;typeSelect.onchange=refresh;$('show-refs').onchange=refresh;$('show-pred').onchange=renderPredictions;
 $('reset-filters').onclick=()=>{typeSelect.value=defaultHospital||'all';$('study-radius').value='25';$('show-refs').checked=true;$('show-pred').checked=true;selectedCircle=null;drawnGroup.clearLayers();refresh();};
 async function searchCity(query){const q=String(query||'').trim().split(',')[0].trim();if(q.length<2)throw new Error('Enter a city or town name.');
 let found=null;try{const u='https://geocoding-api.open-meteo.com/v1/search?name='+encodeURIComponent(q)+'&count=10&language=en&countryCode=CA';const result=await fetch(u);if(!result.ok)throw Error('geocoding service '+result.status);const payload=await result.json();const hits=(payload.results||[]).filter(x=>x.country_code==='CA');const exact=hits.find(x=>x.name.toLowerCase()===q.toLowerCase());const item=exact||hits[0];if(item)found={lat:item.latitude,lon:item.longitude,name:item.name+', '+(item.admin1||'Canada')};}
 catch(err){const offline=commonCities[q.toLowerCase()];if(offline)found={lat:offline[0],lon:offline[1],name:offline[2]};else throw new Error('Geocoding unavailable. Check connection or try Ottawa, Halifax, Toronto, Vancouver, Montréal, Calgary, Edmonton or Winnipeg.');}
 if(!found)throw new Error('No Canadian location found for '+q+'. Try a city name.');
 city=found;selectedFacilityKey=null;selectedCircle=null;drawnGroup.clearLayers();map.setView([city.lat,city.lon],11);$('search').value=city.name;refresh();return city.name;}
 $('search-form').onsubmit=async e=>{
   e.preventDefault();const b=$('search-btn');if(busy)return;busy=true;b.disabled=true;
   try{
     await searchCity($('search').value);
     // A city search can start a bounded pilot scan automatically, if an actual
     // experimental image checkpoint is available. No untrained marker guesses.
     if(experimentalReady && city.lat>=41.7 && city.lat<=57 && city.lon>=-95.2 && city.lon<=-73.8)
       await runAiSearch();
     else if(experimentalReady)
       addChat('The imagery-based pilot currently supports Ontario. Official reference markers remain available across Canada.');
   }catch(err){addChat('Search: '+String(err.message).slice(0,180));}
   finally{busy=false;b.disabled=false;}
 };
 function addChat(message,user=false){const el=document.createElement('div');el.className='msg'+(user?' user':'');el.textContent=message;$('messages').appendChild(el);$('messages').scrollTop=$('messages').scrollHeight;return el;}
 function showEvaluation(){
   if(!evaluation.ready){addChat('AI detector: not trained/evaluated yet. Accuracy cannot be reported. Complete verified roof annotations, then train and evaluate on independent sites. '+evaluation.status);return;}
   const m=evaluation.metrics;
   const pct=x=>(100*x).toFixed(1)+'%';
   addChat('Independent test-set detection results ('+evaluation.test_sites+' sites):\n'+
     'Precision: '+pct(m.precision)+'\nRecall: '+pct(m.recall)+'\nmAP@0.50: '+pct(m.mAP50)+'\nmAP@0.50–0.95: '+pct(m.mAP50_95)+
     '\nThese are held-out object-detection metrics, not the probability a selected map marker is truly a hospital and not an overall Canada-wide accuracy.');
 }
 // One city search selects the real hospital-reference locations in its study radius.
 // The model scans genuine nearby image chips, not the entire city's every roof.
 function referenceSiteTargets(){
   const a=activeArea();const ordered=records.filter(r=>/^hospitals?$/i.test(r.type.trim()) &&
       map.distance(a.center,[r.lat,r.lon])<=a.radius)
       .sort((u,v)=>map.distance(a.center,[u.lat,u.lon])-map.distance(a.center,[v.lat,v.lon]));
   const unique=[];
   for(const p of ordered){
     if(unique.some(u=>map.distance([p.lat,p.lon],[u.lat,u.lon])<35))continue;
     unique.push({lat:p.lat,lon:p.lon,name:p.name});
   }
   return unique;
 }
 function hostArgumentEnvelope(kernel,payload){
   // Colab rejects iframe-realm objects in some versions; create both in host realm.
   let host=window;
   try{if(window.parent&&window.parent!==window&&window.parent.google?.colab?.kernel===kernel)host=window.parent;}catch(e){}
   return [host.JSON.parse(JSON.stringify([payload])),host.JSON.parse('{}')];
 }
 async function runAiSearch(){
   if(!evaluation.ready&&!experimentalReady){
     addChat('AI model unavailable. Train an experimental model in Notebook 08 first; reference markers are not AI detections.');return;
   }
   const a=activeArea();
   let kernel=null;
   try{kernel=window.google?.colab?.kernel||window.parent?.google?.colab?.kernel||null;}catch(e){kernel=window.google?.colab?.kernel||null;}
   if(!kernel||typeof kernel.invokeFunction!=='function'){
     addChat('Colab notebook callback is unavailable; reopen the current Notebook 04 map cell.');return;
   }
   const isSmall=selectedCircle&&a.radius>=20&&a.radius<=250;
   const choices=isSmall?[]:referenceSiteTargets();
   if(!isSmall&&!choices.length){
     addChat('No official hospital references fall in this selection. This PILOT scans roof imagery near known reference sites; it cannot yet exhaustively inspect every building in an empty city-wide selection. Draw a 20–250 m circle for a local image test.');return;
   }
   if(!isSmall && !experimentalReady){
     addChat('City-wide reference-guided scan currently requires the experimental saved model. A verified small-circle detector may still be available.');return;
   }
   if(selectedCircle && a.radius<20){addChat('Circle must be at least 20 metres.');return;}
   if(selectedCircle && a.radius>100000){addChat('Select a city study radius of 100 km or less.');return;}
   const btn=$('run-ai');btn.disabled=true;btn.textContent='Scanning…';
   importedPredictions=[];predictionOrigin=evaluation.ready?'evaluated':'experimental';
   $('show-pred').checked=true;renderPredictions();
   try{
     if(isSmall){
       addChat('Running a local image-model scan; any green rectangles are unverified building candidates.');
       const [args,kwargs]=hostArgumentEnvelope(kernel,{lat:a.center.lat,lon:a.center.lng,radius_m:a.radius});
       const reply=await kernel.invokeFunction('oss4008.hospital_detection',args,kwargs);
       const res=reply?.data?.['application/json'];
       if(!res?.ok){addChat('Image-model scan failed: '+String(res?.error||'No Colab result').slice(0,210));return;}
       importedPredictions=(res.features||[]).map(candidateGeometry).filter(Boolean);
       renderPredictions();
       addChat('Local model scan: '+importedPredictions.length+' size-screened roof bounding rectangles from '+res.image_chips+' images. Experimental predictions do not prove hospital function.');
       return;
     }
     const cap=24, sites=choices.slice(0,cap),batchSize=4;
     addChat('Scanning aerial images near '+sites.length+' known hospital reference sites around '+city.name+
       ', '+Math.ceil(sites.length/batchSize)+' batches. This is REFERENCE-GUIDED (NOT every building in the city).'+
       (choices.length>cap?' First '+cap+' of '+choices.length+' sites only.':''));
     let done=0,failed=0,cache=0;const errs=[];
     for(let i=0;i<sites.length;i+=batchSize){
       const subset=sites.slice(i,i+batchSize);
       const payload={mode:'reference_sites',sites:subset};
       const [args,kwargs]=hostArgumentEnvelope(kernel,payload);
       try{
         const response=await kernel.invokeFunction('oss4008.hospital_detection',args,kwargs);
         const res=response?.data?.['application/json'];
         if(!res?.ok)throw Error(res?.error||'No Colab result');
         done+=Number(res.sites_processed||0);
         failed+=(res.errors||[]).length;
         cache+=Number(res.cached_sites||0);
         errs.push(...(res.errors||[]).map(x=>x.site+': '+x.error));
         importedPredictions.push(...(res.features||[]).map(candidateGeometry).filter(Boolean));
         renderPredictions();
       }catch(err){failed+=subset.length;errs.push('Batch '+(i/batchSize+1)+': '+String(err).slice(0,110));}
       addChat('Scan progress: '+Math.min(i+batchSize,sites.length)+'/'+sites.length+' locations attempted; '+importedPredictions.length+' larger-roof candidate boxes so far (small roof boxes screened out).');
     }
     addChat('Area scan finished: '+done+'/'+sites.length+' hospital-reference sites imaged ('+cache+' reused from cache); '+
       importedPredictions.length+' size-screened model bounding rectangles (unverified). '+failed+' imagery errors.'+
       (errs.length?' First error: '+errs[0]:'')+
       ' Only areas near the known sites were scanned. These are UNVERIFIED model candidates; zoom in to inspect their boundaries.');
   }catch(err){addChat('Area scan error: '+String(err).slice(0,240));}
   finally{btn.disabled=false;btn.textContent='✦ Run AI';}
 }
 $('run-ai').onclick=()=>runAiSearch();

 const aiStatus=$('ai-state');aiStatus.textContent=evaluation.status;
 if(evaluation.ready)aiStatus.classList.add('ready');

 async function answer(query){const text=String(query).trim();if(!text)return;addChat(text,true);const low=text.toLowerCase();
   // Keep original map navigation actions responsive without invoking an LLM.
   if(/\b(run|start|perform)\s+(?:the\s+)?(?:ai|hospital\s+detection|detector)\b/.test(low)){
     await runAiSearch();return;
   }
   if(/(clear|remove).*(circle|selection)/.test(low)){$('clear-area').click();addChat('Selection cleared.');return;}
   if(/^(draw|circle)(\b|\s)/.test(low)){addChat('Select Draw circle above the map, then drag on the map. Run AI searches images near hospital reference sites in that area.');return;}
   if(/^show hospitals? in /i.test(text)){
     const q=text.replace(/^show hospitals? in /i,'');
     try {const name=await searchCity(q);addChat('Showing official hospital references near '+name+'. Click Run AI for an experimental image-based scan.');}
     catch(err){addChat('City search: '+String(err.message).slice(0,160));}
     return;
   }
   const area=activeArea();
   let kernel=null;try{kernel=window.google?.colab?.kernel||window.parent?.google?.colab?.kernel||null;}catch(_error){kernel=window.google?.colab?.kernel||null;}
   if(!kernel||typeof kernel.invokeFunction!=='function'){
     const r=records.find(x=>x.key===selectedFacilityKey);
     if(r) addChat('Official reference: '+r.name+' · '+(r.address||'address not listed')+
       ' · '+r.lat.toFixed(5)+', '+r.lon.toFixed(5)+'. Parking, ER, pricing and helipad details are not verified by this map.');
     else addChat(selected.length+' official healthcare reference facilities are visible. Click a marker and choose Ask about this facility. No language model is connected in this browser session.');
     return;
   }
   const [args,kwargs]=hostArgumentEnvelope(kernel,{
     question:text, lat:area.center.lat, lon:area.center.lng, radius_m:area.radius,
     selected_key:selectedFacilityKey
   });
   try{
     const reply=await kernel.invokeFunction('oss4008.hospital_chat',args,kwargs);
     const data=reply?.data?.['application/json'];
     if(!data?.ok){addChat('Hospital assistant: '+String(data?.error||'No response from notebook').slice(0,200));return;}
     const provider=(data.mode==='gemini'?'Gemini LLM':data.mode==='local-qwen'?'Local Qwen LLM (keyless)':'Official reference-data assistant');
     const message=addChat(data.answer+'\n— '+provider+
       (data.note?'\n'+data.note:''));
     appendNavigation(message,data.nav_results);
   }catch(err){addChat('Hospital assistant unavailable: '+String(err).slice(0,160));}
 }
 $('chat-form').onsubmit=async e=>{e.preventDefault();const x=$('question');const q=x.value;x.value='';await answer(q);};
 document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>answer(b.dataset.prompt));
 $('source-note').textContent=(records.length?datasetStatus.label+' · '+records.length.toLocaleString()+' records. ':'No records loaded. '+datasetStatus.message+' ')+ 'Red = reference pins. Green = experimental AI boxes. Amber = mapped building polygons, not confirmed hospital roofs.';
 $('data-state').textContent=records.length?datasetStatus.label+' ('+records.length.toLocaleString()+')':'No reference data';
 if(!records.length){$('source-note').classList.add('error');$('location-count').textContent='No reference dataset loaded';}
 refresh();setTimeout(()=>map.invalidateSize(),250);
})();
</script>
</div>
</body></html>
"""
