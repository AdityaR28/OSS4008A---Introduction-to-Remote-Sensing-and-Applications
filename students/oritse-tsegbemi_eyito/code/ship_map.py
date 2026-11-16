"""Bounded-area Earth Engine -> Colab PyTorch -> geographic result workflow."""
import asyncio
import csv
import hashlib
import io
import json
import math
import html
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import ee
import geemap
import ipywidgets as widgets
import numpy as np
import rasterio
import requests
import torch
import torchvision
from IPython.display import display, clear_output
from PIL import Image, ImageDraw
from pyproj import Transformer
from ipyleaflet import DrawControl, GeoJSON, CircleMarker, LayerGroup

from ship_backend import predict_rgb, write_json, checksum

COLLECTION = 'COPERNICUS/S2_HARMONIZED'  # L1C, matching the training source.
HALIFAX_BOUNDS = [-63.58, 44.615, -63.545, 44.64]
PORT_SOURCE = 'https://github.com/smdg-org/Terminal-Code-List/blob/master/SMDG%20Terminal%20Code%20List.csv'
PORTS = [('Fairview Cove Container Terminal', 44.667778, -63.627222),
         ('South End Container Terminal', 44.628333, -63.5625)]


def validate_bounds(bounds, max_area_km2=16):
    west, south, east, north = map(float, bounds)
    if not all(math.isfinite(v) for v in [west, south, east, north]):
        raise ValueError('Coordinates must be finite numbers.')
    if not (-141 <= west < east <= -52 and 41 <= south < north <= 84):
        raise ValueError('Select a small area within the Canadian map extent.')
    width = (east-west)*111.32*math.cos(math.radians((south+north)/2))
    height = (north-south)*111.32
    if width*height > max_area_km2 or max(width, height) > 6:
        raise ValueError('Area too large. Select at most 16 square kilometres and 6 km per side.')
    if min(width, height) < 0.1:
        raise ValueError('Area too small. Select at least 100 m per side.')
    return [west, south, east, north]


def fetch_observation(bounds, start, end, scene_cloud_max=30):
    """Retrieve one native 10 m observation; never use a multi-date composite."""
    bounds = validate_bounds(bounds)
    if not math.isfinite(scene_cloud_max) or not 0 <= scene_cloud_max <= 100:
        raise ValueError('Scene cloud maximum must be between 0 and 100.')
    if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
        raise ValueError('The end date must be later than the start date (end exclusive).')
    region = ee.Geometry.Rectangle(bounds, 'EPSG:4326', False)
    collection = (ee.ImageCollection(COLLECTION).filterBounds(region).filterDate(start, end)
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', scene_cloud_max))
                  .sort('CLOUDY_PIXEL_PERCENTAGE'))
    candidates = collection.limit(12).getInfo()['features']
    if not candidates:
        raise ValueError('No matching images. Adjust dates or choose another area.')
    # Check actual AOI coverage; a scene may merely intersect the requested box.
    raw = None
    for candidate in candidates:
        image = ee.Image(candidate['id'])
        mask = image.select(['B4', 'B3', 'B2']).mask().reduce(ee.Reducer.min())
        coverage = mask.unmask(0, sameFootprint=False).reduceRegion(
            ee.Reducer.mean(), region, 60, maxPixels=100000).getInfo()
        if float(next(iter(coverage.values())) or 0) >= 0.98:
            raw, selected = image, candidate
            break
    if raw is None:
        raise ValueError('No candidate covers at least 98% of the requested area.')
    # Approximate the source dataset's 1st/99th percentile band stretch. Source
    # percentiles were not supplied; this reproducible scene-level approximation
    # remains a domain-shift limitation and is recorded in every result.
    bands = ['B4', 'B3', 'B2']
    stats = raw.select(bands).reduceRegion(
        ee.Reducer.percentile([1, 99]), raw.geometry(), 120,
        bestEffort=True, maxPixels=2000000).getInfo()
    lows = np.array([stats[b+'_p1'] for b in bands], dtype=np.float32)
    highs = np.array([stats[b+'_p99'] for b in bands], dtype=np.float32)
    if not np.isfinite(lows).all() or not np.isfinite(highs).all() or np.any(highs <= lows):
        raise ValueError('Cannot determine a valid contrast stretch for this scene.')
    projection = raw.select('B2').projection().getInfo()
    cloud_collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filter(
        ee.Filter.eq('system:index', selected['id'].split('/')[-1]))
    if cloud_collection.size().getInfo() == 0:
        raise ValueError('No matching pixel-cloud assessment. Choose another date or area.')
    cloud = ee.Image(cloud_collection.first()).select('probability')
    # Download RGB and cloud probability on exactly the same native grid.
    download = raw.select(bands).addBands(cloud).toFloat()
    url = download.getDownloadURL({'region': region, 'scale': 10,
                                           'crs': projection['crs'], 'format': 'GEO_TIFF'})
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with rasterio.MemoryFile(response.content) as memory:
        with memory.open() as raster:
            pixels = raster.read([1,2,3]).astype(np.float32)
            available = np.all(raster.read_masks() > 0, axis=0) & np.any(pixels > 0, axis=0)
            cloud_probability = raster.read(4)
            valid = available & (cloud_probability < 65)
            transform, crs = raster.transform, raster.crs.to_string()
    clear_fraction = float(valid.sum()/max(available.sum(),1))
    if clear_fraction < 0.8:
        raise ValueError('Less than 80% of this area passes the pixel-cloud screen. Choose another date or area.')
    if pixels.shape[1]*pixels.shape[2] > 300000:
        raise ValueError('Downloaded raster exceeds the interactive pixel limit.')
    rgb = np.clip((pixels-lows[:, None, None])/(highs-lows)[:, None, None], 0, 1)
    rgb = np.rint(rgb*255).astype(np.uint8).transpose(1, 2, 0)
    rgb[~valid] = 0
    metadata = {'image_id': selected['id'], 'collection': COLLECTION,
                'acquired_utc': datetime.fromtimestamp(selected['properties']['system:time_start']/1000,
                                                       timezone.utc).isoformat(),
                'scene_cloud_percent': selected['properties']['CLOUDY_PIXEL_PERCENTAGE'],
                'scene_cloud_max': scene_cloud_max,
                'bounds_wgs84': bounds, 'pixel_size_m': 10,
                'raster_crs': crs, 'raster_transform': list(transform)[:6],
                'raster_width': rgb.shape[1], 'raster_height': rgb.shape[0],
                'available_pixel_fraction': float(available.mean()),
                'clear_pixel_fraction': clear_fraction,
                'cloud_probability_collection': 'COPERNICUS/S2_CLOUD_PROBABILITY',
                'cloud_probability_max_exclusive': 65,
                'preprocessing': 'Scene-level 1/99 percentiles estimated at 120 m; RGB uint8',
                'stretch_low': lows.tolist(), 'stretch_high': highs.tolist(),
                'quality_warning': 'Pixel-cloud screening is imperfect; bright ships can be masked and land can produce false detections.'}
    return rgb, valid, transform, crs, metadata


def window_starts(size, chip=128, stride=96):
    if size <= chip:
        return [0]
    return sorted(set(list(range(0, size-chip+1, stride)) + [size-chip]))


def detect_raster(model, device, rgb, valid, threshold=0.3):
    """Overlapping native-resolution chips; global NMS removes repeat boxes."""
    boxes, scores = [], []
    height, width = rgb.shape[:2]
    for y in window_starts(height):
        for x in window_starts(width):
            crop = rgb[y:y+128, x:x+128]
            chip = np.zeros((128, 128, 3), np.uint8)
            chip[:crop.shape[0], :crop.shape[1]] = crop
            found, confidence = predict_rgb(model, device, chip, threshold)
            for box, score in zip(found, confidence):
                box = box + np.array([x, y, x, y])
                box[[0, 2]] = box[[0, 2]].clip(0, width)
                box[[1, 3]] = box[[1, 3]].clip(0, height)
                cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                if box[2] > box[0] and box[3] > box[1] and valid[min(cy,height-1), min(cx,width-1)]:
                    boxes.append(box.tolist())
                    scores.append(float(score))
    if not boxes:
        return np.empty((0, 4)), np.empty(0)
    keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.3).numpy()
    return np.array(boxes)[keep], np.array(scores)[keep]


def geographic_results(boxes, scores, transform, crs, metadata):
    converter = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    features = []
    for index, (box, score) in enumerate(zip(boxes, scores), 1):
        x1, y1, x2, y2 = map(float, box)
        corners = [transform*(x1,y1), transform*(x2,y1), transform*(x2,y2), transform*(x1,y2)]
        ring = [list(converter.transform(x, y)) for x, y in corners]
        ring.append(ring[0])
        features.append({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [ring]},
                         'properties': {'detection_id': index, 'class': 'ship', 'confidence': float(score),
                                        'image_id': metadata['image_id'], 'acquired_utc': metadata['acquired_utc'],
                                        'pixel_box_xyxy': [x1,y1,x2,y2]}})
    return {'type': 'FeatureCollection', 'features': features}


def save_detection(project, rgb, boxes, scores, geojson, metadata):
    project = Path(project)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:12]
    stem = 'ships_' + metadata['acquired_utc'][:10] + '_' + key
    folder = project / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f'{stem}.geojson', geojson)
    write_json(folder / f'{stem}.json', metadata)
    with (folder / f'{stem}.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['detection_id','class','confidence','longitude','latitude','image_id','acquired_utc'])
        for feature in geojson['features']:
            ring = feature['geometry']['coordinates'][0][:-1]
            lon, lat = np.array(ring).mean(axis=0)
            prop = feature['properties']
            writer.writerow([prop['detection_id'],'ship',prop['confidence'],lon,lat,prop['image_id'],prop['acquired_utc']])
    preview = Image.fromarray(rgb)
    draw = ImageDraw.Draw(preview)
    for box, score in zip(boxes, scores):
        draw.rectangle(tuple(map(float, box)), outline='#ff3636', width=2)
        draw.text((float(box[0]), max(0., float(box[1])-11)), f'{score:.2f}', fill='#ff3636')
    preview.save(folder / f'{stem}.png')
    print('Saved CSV, GeoJSON, PNG and metadata:', stem)
    return stem


def build_detection_app(project, model, device, metrics):
    """Return the Colab application; model and Earth Engine are caller-owned."""
    project = Path(project)
    reference_features = []
    with (project/'data/halifax_port_reference.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['name','latitude','longitude','source_url','source_row_date','record_type'])
        for name, lat, lon in PORTS:
            writer.writerow([name,lat,lon,PORT_SOURCE,'2020-01-24','published reference point'])
            reference_features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[lon,lat]},
                                       'properties':{'name':name,'source_url':PORT_SOURCE,
                                                     'source_row_date':'2020-01-24','record_type':'reference'}})
    write_json(project/'data/halifax_port_reference.geojson',{'type':'FeatureCollection','features':reference_features})
    Map = geemap.Map(center=[58, -96], zoom=3, ee_initialize=False,
                     draw_ctrl=False, height='550px')
    Map.clear_layers()
    Map.add_basemap('Esri.WorldGrayCanvas')
    status, output = widgets.Output(), widgets.Output()
    fields = [widgets.FloatText(value=value, description=name, layout=widgets.Layout(width='240px'))
              for name, value in zip(['West:', 'South:', 'East:', 'North:'], HALIFAX_BOUNDS)]
    start = widgets.Text(value='2025-06-01', description='From:')
    end = widgets.Text(value='2025-09-01', description='To (excl.):')
    threshold = widgets.FloatSlider(value=0.55, min=0.05, max=0.95, step=0.05, description='Confidence:')
    selection_path = project/'results/threshold_selection.json'
    if selection_path.exists():
        threshold.value = float(json.loads(selection_path.read_text())['validation']['threshold'])
    cloud_max = widgets.IntSlider(value=30, min=0, max=100, step=5, description='Cloud max %:')
    detect = widgets.Button(description='Detect ships', button_style='primary')
    preview = widgets.Button(description='Load imagery')
    canada = widgets.Button(description='Canada overview')
    halifax = widgets.Button(description='Halifax example')
    pick = widgets.ToggleButton(value=False, description='Pick area on map')
    auto = widgets.Checkbox(value=False, description='Detect after moving map (zoom 15+)')
    ports = widgets.Checkbox(value=True, description='Show port/marina references')
    count = widgets.HTML('<b>Ship detections:</b> not run')
    details = widgets.HTML('Click a predicted box to inspect its confidence.')
    state = {'last': None, 'layers': [], 'cache': OrderedDict(), 'busy': False, 'task': None}

    presets = {
        'Halifax (Atlantic)': HALIFAX_BOUNDS,
        'Vancouver (Pacific)': [-123.115,49.285,-123.08,49.31],
        'Montreal (St Lawrence)': [-73.55,45.53,-73.515,45.555],
        'Hamilton (Great Lakes)': [-79.875,43.270,-79.862,43.280],
        'Churchill (Hudson Bay)': [-94.23,58.765,-94.195,58.79],
        'Iqaluit (Arctic)': [-68.54,63.725,-68.505,63.75],
        'Prince Rupert (Pacific North)': [-130.345,54.29,-130.31,54.315],
        'St Johns (Newfoundland)': [-52.72,47.555,-52.685,47.58],
        'Simpson Point (Yukon)': [-138.955,69.554,-138.877,69.584]}
    preset = widgets.Dropdown(options=['Choose area']+list(presets), description='Region:')
    from national_data import read_map_references
    references = read_map_references(project)
    province = widgets.Dropdown(options=['All']+sorted({r['province'] for r in references}), description='Province:')
    subtype = widgets.Dropdown(options=['All']+sorted({r['subtype'] for r in references}), description='Facility:')
    search = widgets.Text(description='Port name:', placeholder='Search national references')
    port_select = widgets.Dropdown(options=[('Select a reference', '')], description='Reference:')
    reference_count = widgets.HTML()
    download = widgets.Dropdown(options=['GeoJSON','CSV','PNG','Metadata'], description='Export:')
    download_button = widgets.Button(description='Download current result', disabled=True)

    terminal_markers = []
    for name, lat, lon in PORTS:
        marker = CircleMarker(location=(lat,lon), radius=7, color='white', fill_color='#bd3154', fill_opacity=1)
        marker.popup = widgets.HTML('<div style="color:#17212b;background:white"><b>'+name+'</b><br>'
                                   'Published reference point, not a detection.<br>Source row: 2020-01-24<br>'
                                   '<a href="'+PORT_SOURCE+'" target="_blank">SMDG reference list</a></div>')
        terminal_markers.append(marker)
    port_layer = LayerGroup(layers=tuple(terminal_markers), name='Terminal references')
    Map.add_layer(port_layer)
    ports.observe(lambda change: setattr(port_layer, 'layers', tuple(terminal_markers) if change['new'] else ()), names='value')

    national_layer = GeoJSON(data={'type':'FeatureCollection','features':[]}, name='National port/marina references',
                            point_style={'radius':4,'color':'#265f9e','fillOpacity':0.7})
    Map.add_layer(national_layer)

    def update_references(*_):
        from national_data import filter_ports
        rows = filter_ports(references, province.value, subtype.value, search.value)
        reference_count.value = (f'<b>References:</b> {len(rows):,} of {len(references):,}<br>'
                                'Published facilities; coverage varies. Blue points are not detections.') if references else (
                                'National inventory is not loaded. Run acquire_ports first and rebuild the dashboard.')
        port_select.options = [('Select a reference','')]+[(r['name']+' | '+r['province'],r['reference_id']) for r in rows]
        national_layer.data = {'type':'FeatureCollection','features':[
            {'type':'Feature','geometry':{'type':'Point','coordinates':[float(r['longitude']),float(r['latitude'])]},
             'properties':{'name':r['name'],'province':r['province'],'subtype':r['subtype'],
                           'provider':r.get('provider',''),'source_url':r.get('source_url',''),
                           'record_type':r.get('record_type','published reference'),
                           'validation_status':r.get('validation_status','')}} for r in rows]} if ports.value else {
                               'type':'FeatureCollection','features':[]}
    for widget in [province, subtype, search, ports]:
        widget.observe(update_references,names='value')
    update_references()

    def choose_preset(change):
        if change['new'] in presets:
            bounds = presets[change['new']]
            for field,value in zip(fields,bounds):
                field.value=value
            Map.center=((bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2)
            Map.zoom=14
    preset.observe(choose_preset,names='value')

    def choose_port(change):
        row=next((r for r in references if r['reference_id']==change['new']),None)
        if row:
            lat,lon=float(row['latitude']),float(row['longitude'])
            dx=.009/math.cos(math.radians(lat))
            for field,value in zip(fields,[lon-dx,lat-.009,lon+dx,lat+.009]):
                field.value=value
            Map.center,Map.zoom=(lat,lon),14
    port_select.observe(choose_port,names='value')

    def clear_results():
        for layer in state['layers']:
            if layer in Map.layers:
                Map.remove_layer(layer)
        state['layers'] = []
        state['last'] = None
        state.pop('output_stem', None)
        download_button.disabled = True
        count.value = '<b>Ship detections:</b> not run for current settings'
        details.value = 'Click a predicted box to inspect its confidence.'
        with output:
            clear_output(wait=False)

    def changed(_):
        if not state['busy']:
            clear_results()

    for field in fields + [start, end, threshold, cloud_max]:
        field.observe(changed, names='value')

    def run(inference=True):
        if state['busy']:
            return
        state['busy'] = True
        detect.disabled = preview.disabled = True
        clear_results()
        with status:
            clear_output(wait=True)
            print('Loading the selected satellite observation...')
        try:
            if inference and model is None:
                raise ValueError('Restore the preserved checkpoint before Detect ships. References and saved results remain available.')
            bounds = validate_bounds([field.value for field in fields])
            key = (tuple(bounds), start.value, end.value, cloud_max.value)
            if key not in state['cache']:
                state['cache'][key] = fetch_observation(bounds, start.value, end.value, cloud_max.value)
                if len(state['cache']) > 4:
                    state['cache'].popitem(last=False)
            rgb, valid, transform, crs, source_metadata = state['cache'][key]
            metadata = dict(source_metadata)
            west, south, east, north = bounds
            selection = GeoJSON(data={'type':'Feature','properties':{},'geometry':{'type':'Polygon',
                                'coordinates':[[[west,south],[east,south],[east,north],[west,north],[west,south]]]}},
                                name='Selected area', style={'color':'#ff9900','weight':2,'fillOpacity':0})
            Map.add_layer(selection)
            state['layers'].append(selection)
            # Draw the same RGB pixels used by inference, not a different composite.
            # GeoTIFF pixel grid is projected; use geemap's EE visualization for
            # correctly placed context, and GeoJSON for exact detection geometry.
            raw = ee.Image(metadata['image_id']).select(['B4','B3','B2'])
            low = ee.Image.constant(metadata['stretch_low']).rename(['B4','B3','B2'])
            high = ee.Image.constant(metadata['stretch_high']).rename(['B4','B3','B2'])
            visible = raw.subtract(low).divide(high.subtract(low)).clamp(0,1).clip(ee.Geometry.Rectangle(bounds))
            Map.addLayer(visible, {'min':0,'max':1}, 'Selected Sentinel-2 observation')
            state['layers'].append(Map.layers[-1])
            if inference:
                started = time.perf_counter()
                boxes, scores = detect_raster(model, device, rgb, valid, threshold.value)
                result = geographic_results(boxes, scores, transform, crs, metadata)
                metadata.update({'threshold': threshold.value, 'detection_count': len(boxes),
                                 'inference_seconds': time.perf_counter()-started,
                                 'model': 'Fine-tuned Faster R-CNN',
                                 'test_ap50': metrics.get('ap_50'), 'canadian_accuracy_measured': False,
                                 'run_utc': datetime.now(timezone.utc).isoformat(),
                                 'model_checkpoint_sha256': checksum(project/'results/models/vessel_detector.pt', 'sha256')})
                layer = GeoJSON(data=result, name='Predicted ships', style={'color':'#ff3636','weight':2,'fillOpacity':0.1})
                def clicked(**event):
                    feature = event.get('feature', {})
                    prop = feature.get('properties', event.get('properties', {}))
                    if 'confidence' in prop:
                        details.value = (f"Prediction #{prop['detection_id']} | ship | confidence {prop['confidence']:.3f}"
                                         f" | acquired {metadata['acquired_utc']}")
                layer.on_click(clicked)
                Map.add_layer(layer)
                state['layers'].append(layer)
                state['last'] = (result, metadata)
                count.value = f'<b>Predicted ships:</b> {len(boxes)} | confidence ≥ {threshold.value:.2f}'
                with output:
                    clear_output(wait=True)
                    saved_stem = save_detection(project, rgb, boxes, scores, result, metadata)
                    state['output_stem'] = saved_stem
                    download_button.disabled = False
                    with Image.open(project/'results'/f'{saved_stem}.png') as saved_preview:
                        saved_preview.thumbnail((600,600))
                        display(saved_preview.copy())
            with status:
                clear_output(wait=True)
                print('Acquired:', metadata['acquired_utc'])
                print('Scene:', metadata['image_id'])
                print('10 m imagery. Results are experimental; Canadian accuracy is not yet validated.')
                if not inference:
                    print('Imagery loaded. Click Detect ships to run the model.')
        except Exception as error:
            with status:
                clear_output(wait=True)
                # Do not print signed download URLs or credential-bearing errors.
                print(str(error) if isinstance(error, ValueError) else 'Request failed: '+type(error).__name__+'. Retry or choose another area.')
        finally:
            state['busy'] = False
            detect.disabled = preview.disabled = False

    def go_halifax(_):
        for field, value in zip(fields, HALIFAX_BOUNDS):
            field.value = value
        Map.center, Map.zoom = (44.628,-63.5625), 14

    draw = DrawControl(rectangle={'shapeOptions': {'color':'#ff9900'}},
                       polygon={}, polyline={}, circle={}, circlemarker={}, marker={})

    def drawn(_, action, geo_json):
        clear_results()
        if action == 'deleted':
            return
        coords = np.asarray(geo_json['geometry']['coordinates'][0])
        bounds = [coords[:,0].min(), coords[:,1].min(), coords[:,0].max(), coords[:,1].max()]
        for field, value in zip(fields, bounds):
            field.value = float(value)
        with status:
            clear_output(wait=True)
            print('Selection updated. Click Detect ships.')

    draw.on_draw(drawn)
    Map.add_control(draw)

    def map_clicked(**event):
        if event.get('type') != 'click' or Map.zoom < 12 or not pick.value:
            return
        lat, lon = event['coordinates']
        dx = 0.009/math.cos(math.radians(lat))
        for field, value in zip(fields, [lon-dx,lat-0.009,lon+dx,lat+0.009]):
            field.value = value
        pick.value = False

    Map.on_interaction(map_clicked)

    def moved(_):
        if state['task'] and not state['task'].done():
            state['task'].cancel()
        if not auto.value or Map.zoom < 15 or state['busy']:
            return

        async def after_pause():
            await asyncio.sleep(1.)
            if not auto.value or Map.zoom < 15:
                return
            (south, west), (north, east) = Map.bounds
            for field, value in zip(fields, [west,south,east,north]):
                field.value = value
            run(True)
        state['task'] = asyncio.get_running_loop().create_task(after_pause())

    Map.observe(moved, names='bounds')
    detect.on_click(lambda _: run(True))
    preview.on_click(lambda _: run(False))
    canada.on_click(lambda _: (setattr(Map,'center',(58,-96)), setattr(Map,'zoom',3)))
    halifax.on_click(go_halifax)
    def export_current(_):
        from google.colab import files
        suffix={'GeoJSON':'.geojson','CSV':'.csv','PNG':'.png','Metadata':'.json'}[download.value]
        if state.get('output_stem'):
            files.download(str(project/'results'/(state['output_stem']+suffix)))
    download_button.on_click(export_current)

    saved_paths = sorted((project/'results').glob('ships_*.geojson'))
    saved = widgets.Dropdown(options=[('Select saved prediction','')]+[(p.stem,p.stem) for p in saved_paths],description='Saved run:')
    def show_saved(change):
        if not change['new']:
            return
        clear_results()
        stem=change['new']
        data=json.loads((project/'results'/f'{stem}.geojson').read_text())
        metadata=json.loads((project/'results'/f'{stem}.json').read_text())
        layer=GeoJSON(data=data,name='Saved experimental predictions',style={'color':'#ff3636','weight':2,'fillOpacity':.1})
        Map.add_layer(layer)
        state['layers'].append(layer)
        # Keep last=None: archived output must not masquerade as current-settings inference.
        state['output_stem']=stem
        download_button.disabled=False
        count.value=f'<b>Saved predictions:</b> {len(data["features"])} | '+html.escape(metadata['acquired_utc'])
        details.value='Archived run; dates, area and confidence come from its metadata, not current controls.'
        bounds=metadata.get('bounds_wgs84')
        if bounds:
            Map.center=((bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2)
            Map.zoom=14
        with output:
            clear_output(wait=True)
            image_path=project/'results'/f'{stem}.png'
            if image_path.exists():
                with Image.open(image_path) as image:
                    image.thumbnail((700,550))
                    display(image.copy())
    saved.observe(show_saved,names='value')
    sidebar=widgets.VBox([widgets.HTML('<h4>Explore Canada</h4>'),preset,province,subtype,search,port_select,
                          ports,reference_count,widgets.HTML('<h4>Ship detection</h4>'),start,end,threshold,cloud_max,
                          widgets.HTML('End date is exclusive. Native 10 m input; small boats may be unresolved.'),
                          *fields,widgets.HBox([preview,detect]),pick,auto,
                          widgets.HTML('<h4>Results and exports</h4>'),saved,download,download_button],
                         layout=widgets.Layout(width='340px',min_width='300px',padding='8px'))
    main=widgets.VBox([widgets.HBox([canada,halifax]),count,details,status,Map,output],layout=widgets.Layout(flex='1'))
    panel = widgets.VBox([widgets.HTML('<h3>Ships and ports in Canada</h3>'
                         '<p>Experimental ship detector | Port and marina references | Canadian accuracy pending</p>'),
                         widgets.HBox([sidebar,main],layout=widgets.Layout(width='100%',align_items='flex-start'))])
    return panel, {'map':Map, 'run':run, 'state':state, 'fields':fields,
                   'threshold':threshold, 'go_halifax':go_halifax, 'auto':auto,
                   'draw_callback':drawn, 'port_layer':port_layer, 'ports':ports,
                   'pick':pick,'details':details,'preset':preset,'province':province,'subtype':subtype,
                   'search':search,'port_select':port_select,'national_layer':national_layer,
                   'cloud_max':cloud_max,'saved':saved,'download_button':download_button}
