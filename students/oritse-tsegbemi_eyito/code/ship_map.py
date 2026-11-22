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


class _RejectedObservation(ValueError):
    """A scene failed a quality check and another candidate may work."""


def fetch_observation(bounds, start, end, scene_cloud_max=30):
    """Retrieve one native 10 m observation; never use a multi-date composite."""
    bounds = validate_bounds(bounds)
    if not math.isfinite(scene_cloud_max) or not 0 <= scene_cloud_max <= 100:
        raise ValueError('Scene cloud maximum must be between 0 and 100.')
    if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
        raise ValueError('The end date must be later than the start date (end exclusive).')
    started = time.perf_counter()
    region = ee.Geometry.Rectangle(bounds, 'EPSG:4326', False)
    collection = (ee.ImageCollection(COLLECTION).filterBounds(region).filterDate(start, end)
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', scene_cloud_max))
                  .sort('CLOUDY_PIXEL_PERCENTAGE'))
    candidates = collection.limit(12).getInfo()['features']
    search_seconds = time.perf_counter() - started
    if not candidates:
        raise ValueError('No matching images. Adjust dates or choose another area.')
    skipped = []
    for candidate in candidates:
        try:
            rgb, valid, transform, crs, metadata = _fetch_candidate(
                candidate, region, bounds, scene_cloud_max)
        except _RejectedObservation as error:
            skipped.append({'scene': candidate['id'], 'reason': str(error)})
            continue
        metadata['skipped_scenes'] = skipped
        metadata['scene_search_seconds'] = search_seconds
        metadata['imagery_total_seconds'] = time.perf_counter() - started
        return rgb, valid, transform, crs, metadata
    reasons = '; '.join(dict.fromkeys(item['reason'] for item in skipped))
    raise ValueError(f'No usable scene among {len(skipped)} candidates. {reasons} Adjust dates or area.')


def _cloud_probability_image(image_id):
    return (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filter(ee.Filter.eq('system:index', image_id.split('/')[-1])))


def _fetch_candidate(selected, region, bounds, scene_cloud_max):
    bands = ['B4', 'B3', 'B2']
    raw = ee.Image(selected['id'])
    mask = raw.select(bands).mask().reduce(ee.Reducer.min())
    coverage = mask.unmask(0, sameFootprint=False).reduceRegion(
        ee.Reducer.mean(), region, 60, maxPixels=100000).getInfo()
    if float(next(iter(coverage.values()), None) or 0) < 0.98:
        raise _RejectedObservation('Area coverage below 98%.')

    cloud_collection = _cloud_probability_image(selected['id'])
    if cloud_collection.size().getInfo() == 0:
        raise _RejectedObservation('No matching pixel-cloud assessment.')
    cloud = ee.Image(cloud_collection.first()).select('probability').unmask(100)
    projection = raw.select('B2').projection().getInfo()
    # Download RGB and cloud probability on exactly the same native grid.
    download = raw.select(bands).addBands(cloud).toFloat()
    download_started = time.perf_counter()
    url = download.getDownloadURL({'region': region, 'scale': 10,
                                           'crs': projection['crs'], 'format': 'GEO_TIFF'})
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    download_seconds = time.perf_counter() - download_started
    with rasterio.MemoryFile(response.content) as memory:
        with memory.open() as raster:
            pixels = raster.read([1,2,3]).astype(np.float32)
            available = np.all(raster.read_masks() > 0, axis=0) & np.any(pixels > 0, axis=0)
            cloud_probability = raster.read(4)
            valid = available & (cloud_probability < 65)
            transform, crs = raster.transform, raster.crs.to_string()
    clear_fraction = float(valid.sum()/max(available.sum(),1))
    if clear_fraction < 0.8:
        raise _RejectedObservation('Less than 80% of available pixels pass the pixel-cloud screen.')
    if pixels.shape[1]*pixels.shape[2] > 300000:
        raise ValueError('Downloaded raster exceeds the interactive pixel limit.')
    # Approximate the training data's 1st/99th percentile stretch for usable scenes.
    stats = raw.select(bands).reduceRegion(
        ee.Reducer.percentile([1, 99]), raw.geometry(), 120,
        bestEffort=True, maxPixels=2000000).getInfo()
    lows = np.array([stats.get(b+'_p1', np.nan) for b in bands], dtype=np.float32)
    highs = np.array([stats.get(b+'_p99', np.nan) for b in bands], dtype=np.float32)
    if not np.isfinite(lows).all() or not np.isfinite(highs).all() or np.any(highs <= lows):
        raise _RejectedObservation('Cannot determine a valid contrast stretch.')
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
                'selected_scene_download_seconds': download_seconds,
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
                     draw_ctrl=False, height='620px')
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
    summary = widgets.HTML('<p class="sp-muted">Select an area or open a saved run to inspect ship predictions.</p>')
    scene_info = widgets.HTML('<span class="sp-muted">No satellite scene selected</span>')

    def update_summary(features, metadata):
        scores = [float(item['properties']['confidence']) for item in features]
        bars = []
        for low, high in [(0, .6), (.6, .8), (.8, 1.000001)]:
            amount = sum(low <= score < high for score in scores)
            width = 100 * amount / max(len(scores), 1)
            bars.append(f'<div class="sp-bar-row"><span>{low:.0%}–{min(high, 1):.0%}</span>'
                        f'<span class="sp-track"><i style="width:{width:.1f}%"></i></span><b>{amount}</b></div>')
        mean = f'{np.mean(scores):.0%}' if scores else '—'
        summary.value = (f'<div class="sp-stats"><div><strong>{len(scores)}</strong><span>Predicted ships</span></div>'
                         f'<div><strong>{mean}</strong><span>Mean confidence</span></div></div>'
                         '<div class="sp-label">Prediction confidence</div>' + ''.join(bars) +
                         '<p class="sp-muted">Model scores describe predictions. Canadian accuracy requires independent labels.</p>')
        scene_info.value = (f'<b>{html.escape(metadata["acquired_utc"][:10])}</b> · Sentinel-2 · 10 m'
                            f' · threshold {metadata.get("threshold", threshold.value):.0%}')
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
        summary.value = '<p class="sp-muted">Run detection for this area to see the analysis.</p>'
        scene_info.value = '<span class="sp-muted">No result for current settings</span>'
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
        run_details.layout.display = ''
        try:
            if inference and model is None:
                raise ValueError('Restore the preserved checkpoint before Detect ships. References and saved results remain available.')
            bounds = validate_bounds([field.value for field in fields])
            key = (tuple(bounds), start.value, end.value, cloud_max.value)
            cache_hit = key in state['cache']
            if not cache_hit:
                state['cache'][key] = fetch_observation(bounds, start.value, end.value, cloud_max.value)
                if len(state['cache']) > 4:
                    state['cache'].popitem(last=False)
            rgb, valid, transform, crs, source_metadata = state['cache'][key]
            metadata = dict(source_metadata)
            metadata['imagery_cache_hit'] = cache_hit
            map_started = time.perf_counter()
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
            cloud = ee.Image(_cloud_probability_image(metadata['image_id']).first()).select('probability')
            rgb_valid = raw.mask().reduce(ee.Reducer.min()).And(raw.gt(0).reduce(ee.Reducer.max()))
            visible = (raw.subtract(low).divide(high.subtract(low)).clamp(0,1)
                       .updateMask(rgb_valid.And(cloud.unmask(100).lt(65)))
                       .clip(ee.Geometry.Rectangle(bounds)))
            Map.addLayer(visible, {'min':0,'max':1}, 'Selected Sentinel-2 observation')
            state['layers'].append(Map.layers[-1])
            metadata['map_layer_setup_seconds'] = time.perf_counter() - map_started
            scene_info.value = f'<b>{html.escape(metadata["acquired_utc"][:10])}</b> · Sentinel-2 · 10 m'
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
                layer = GeoJSON(data=result, name='Predicted ships', style={'color':'#ff3636','weight':2,'fillOpacity':0.1,'bubblingMouseEvents':False})
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
                update_summary(result['features'], metadata)
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
                print(f"Clear coverage: {metadata['clear_pixel_fraction']:.0%} of available pixels.")
                skipped_count = len(metadata.get('skipped_scenes', []))
                if skipped_count:
                    print(f'Skipped {skipped_count} other scene(s) that failed quality checks.')
                print('10 m imagery. Results are experimental; Canadian accuracy is not yet validated.')
                if not inference:
                    print('Imagery loaded. Click Detect ships to run the model.')
            run_details.layout.display = 'none'
        except Exception as error:
            run_details.layout.display = ''
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
        if event.get('type') != 'click':
            return
        lat, lon = event['coordinates']
        if not pick.value:
            details.value = f'Selected location | {lat:.5f}, {lon:.5f} | Enable Pick area on map to select a detection area.'
            return
        if Map.zoom < 12:
            details.value = 'Zoom to level 12 or closer, then click to select a detection area.'
            return
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
        layer=GeoJSON(data=data,name='Saved experimental predictions',style={'color':'#ff3636','weight':2,'fillOpacity':.1,'bubblingMouseEvents':False})
        def inspect_saved(**event):
            prop = event.get('feature', {}).get('properties', event.get('properties', {}))
            if 'confidence' in prop:
                details.value = (f'Archived prediction #{prop.get("detection_id", "")} | ship | '
                                 f'confidence {prop["confidence"]:.3f} | acquired {metadata["acquired_utc"]}')
        layer.on_click(inspect_saved)
        Map.add_layer(layer)
        state['layers'].append(layer)
        # Keep last=None: archived output must not masquerade as current-settings inference.
        state['output_stem']=stem
        download_button.disabled=False
        count.value=f'<b>Saved predictions:</b> {len(data["features"])} | '+html.escape(metadata['acquired_utc'])
        details.value='Archived run; dates, area and confidence come from its metadata, not current controls.'
        update_summary(data['features'], metadata)
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
    for widget in [preset, province, subtype, search, port_select, start, end, saved, download]:
        widget.layout.width = '100%'
        widget.style.description_width = '90px'
    for field in fields:
        field.layout.width = '100%'
        field.style.description_width = '90px'
    for widget in [threshold, cloud_max]:
        widget.layout.width = '100%'
        widget.style.description_width = '90px'
    detect.button_style = 'primary'
    detect.icon = 'search'
    preview.icon = 'image'
    download_button.icon = 'download'
    canada.icon = 'globe'
    halifax.icon = 'map-marker'
    pick.layout.width = '100%'
    auto.layout.width = '100%'
    coordinate_toggle = widgets.ToggleButton(description='Area coordinates', icon='crosshairs')
    coordinate_fields = widgets.VBox(fields, layout=widgets.Layout(display='none'))
    coordinate_toggle.observe(
        lambda change: setattr(coordinate_fields.layout, 'display', '' if change['new'] else 'none'),
        names='value')
    coordinates = widgets.VBox([coordinate_toggle, coordinate_fields])
    analysis = widgets.VBox([
        summary, preset, widgets.HTML('<div class="sp-label">Satellite observation</div>'),
        start, end, threshold, cloud_max, coordinates,
        widgets.HBox([preview, detect], layout=widgets.Layout(flex_flow='row wrap', gap='8px')),
        pick, auto], layout=widgets.Layout(gap='8px'))
    places = widgets.VBox([province, subtype, search, port_select, ports, reference_count],
                           layout=widgets.Layout(gap='10px'))
    exports = widgets.VBox([saved, download, download_button,
                            widgets.HTML('<p class="sp-muted">Saved runs use their original acquisition date, area and confidence threshold.</p>'),
                            output], layout=widgets.Layout(gap='12px'))
    tabs = widgets.ToggleButtons(options=['Analyse', 'Ports', 'Results'], value='Analyse')
    tabs.layout.width = '100%'
    tabs.style.button_width = '88px'
    sections = widgets.VBox([analysis])
    def switch_section(change):
        sections.children = ({'Analyse': analysis, 'Ports': places, 'Results': exports}[change['new']],)
    tabs.observe(switch_section, names='value')
    sidebar = widgets.VBox([widgets.HTML('<h3>Ship analysis</h3>'), tabs, sections],
                            layout=widgets.Layout(width='330px', min_width='300px',
                                                  max_height='710px', overflow='auto', padding='18px'))
    sidebar.add_class('sp-sidebar')
    scene_info.add_class('sp-scene')
    details.add_class('sp-details')
    run_details = widgets.VBox([count, status], layout=widgets.Layout(display='none'))
    main = widgets.VBox([
        widgets.HBox([canada, halifax], layout=widgets.Layout(gap='8px', flex_flow='row wrap')),
        scene_info, Map, details,
        run_details
    ], layout=widgets.Layout(flex='1 1 330px', min_width='300px', padding='14px', gap='10px'))
    project_button = widgets.Button(description='Ships and ports', icon='ship', button_style='primary',
                                    layout=widgets.Layout(width='185px', height='42px'))
    project_button.on_click(lambda _: (clear_results(), setattr(Map, 'center', (58,-96)), setattr(Map, 'zoom', 3)))
    footer = widgets.HBox([widgets.HTML('<span class="sp-label">Project</span>'), project_button],
                          layout=widgets.Layout(gap='16px', align_items='center', padding='12px 20px'))
    footer.add_class('sp-footer')
    css = widgets.HTML("""<style>
    .sp-dashboard{background:#f2f5f6;border:1px solid #d4dfe2;border-radius:14px;
      color:#162f3c;font-family:Inter,Arial,sans-serif;overflow:hidden}
    .sp-dashboard h2,.sp-dashboard h3{color:#162f3c;margin:0}
    .sp-dashboard h2{font-size:24px;font-weight:650}
    .sp-dashboard h3{font-size:18px;margin-bottom:12px}
    .sp-sidebar{background:white;border-right:1px solid #dce5e8}
    .sp-dashboard .widget-tab>.p-TabBar .p-TabBar-tab.p-mod-current{border-bottom:3px solid #167a83}
    .sp-dashboard .widget-button.mod-primary{background:#126d79;color:white;border:0;border-radius:6px}
    .sp-dashboard .widget-button{border-radius:6px;min-height:32px}
    .sp-dashboard .widget-button,.sp-dashboard .widget-toggle-button{
      background:#e7eff1;color:#24414e;border:1px solid #d2dfe3}
    .sp-dashboard .widget-toggle-button.mod-active{background:#126d79;color:white}
    .sp-dashboard input,.sp-dashboard select{background:white!important;color:#24414e!important;border-color:#cbd9df!important}
    .sp-dashboard .widget-readout{color:#405b68!important}
    .sp-dashboard .widget-label{color:#405b68}
    .sp-dashboard .widget-html-content{line-height:1.5;color:#24414e}
    .sp-stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:8px 0 18px}
    .sp-stats>div{background:#edf5f5;border:1px solid #d6e6e7;border-radius:9px;padding:12px}
    .sp-stats strong{display:block;font-size:28px;line-height:1.2;color:#126d79}
    .sp-stats span{display:block;font-size:11px;color:#47616d;margin-top:6px}
    .sp-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-weight:700;color:#526e7a}
    .sp-muted{color:#58717b;font-size:12px}
    .sp-bar-row{display:grid;grid-template-columns:65px 1fr 20px;align-items:center;gap:9px;font-size:11px;margin:7px 0;color:#405b68}
    .sp-track{background:#e7eef0;border-radius:5px;overflow:hidden;height:7px}
    .sp-track i{display:block;background:#258b96;height:7px;border-radius:5px}
    .sp-scene{font-size:12px;color:#4d6672}
    .sp-details{background:white;border:1px solid #dce5e8;padding:9px;border-radius:6px;font-size:12px}
    .sp-footer{background:#e8eff1;border-top:1px solid #d3dfe3}
    .sp-dashboard .leaflet-container{border-radius:9px}
    </style>""")
    header = widgets.HBox([
        widgets.HTML('<div class="sp-label">OSS4008A · Canada</div><h2>Ships and ports</h2>'),
        widgets.HTML(f'<span class="sp-muted">{"Model loaded" if model is not None else "Load demo files to enable AI"}'
                     ' · 10 m Sentinel-2</span>')
    ], layout=widgets.Layout(justify_content='space-between', align_items='center', padding='18px 22px',
                              flex_flow='row wrap'))
    panel = widgets.VBox([css, header,
                          widgets.HBox([sidebar, main], layout=widgets.Layout(width='100%', flex_flow='row wrap')),
                          footer], layout=widgets.Layout(width='100%', max_width='1400px'))
    panel.add_class('sp-dashboard')
    return panel, {'map':Map, 'run':run, 'state':state, 'fields':fields,
                   'threshold':threshold, 'go_halifax':go_halifax, 'auto':auto,
                   'draw_callback':drawn, 'port_layer':port_layer, 'ports':ports,
                   'pick':pick,'details':details,'preset':preset,'province':province,'subtype':subtype,
                   'search':search,'port_select':port_select,'national_layer':national_layer,
                   'cloud_max':cloud_max,'saved':saved,'download_button':download_button,
                   'summary':summary,'scene_info':scene_info,'project_button':project_button,'tabs':tabs}

