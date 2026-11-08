"""Canada hospital infrastructure map — OSS4008 research application.

Reference layer: Statistics Canada ODHF v2.0 (2026).
Model layer: uploaded GeoJSON produced by a separately validated detector.
Geocoding: Open-Meteo Geocoding API, limited to deliberate user searches.
"""
from __future__ import annotations

from functools import lru_cache
import html
import json
import math
from pathlib import Path
import time

import folium
from folium.plugins import Draw, Fullscreen, MarkerCluster
import gradio as gr
import pandas as pd
import requests

DATA_FILE = Path('/content/odhf_hospitals.csv') if Path('/content').exists() else Path('odhf_hospitals.csv')
DATA_SOURCE = 'Statistics Canada — Open Database of Healthcare Facilities, v2.0 (2026)'
CANADA_CENTER = (56.1304, -106.3468)
DEFAULT_CENTER = (44.6488, -63.5752)
MAX_MARKERS = 900


def validate_coordinates(lat: float, lon: float, radius_km: float):
    lat, lon, radius_km = float(lat), float(lon), float(radius_km)
    if not all(math.isfinite(n) for n in (lat, lon, radius_km)):
        raise ValueError('Coordinates must be finite numbers.')
    if not (41 <= lat <= 84 and -142 <= lon <= -52):
        raise ValueError('Select a location within Canada.')
    if not (1 <= radius_km <= 100):
        raise ValueError('Radius must be from 1 to 100 km.')
    return lat, lon, radius_km


@lru_cache(maxsize=64)
def search_place(query: str):
    """Search Canadian municipalities. Deliberate search only, no autocomplete."""
    query = str(query).strip()
    if len(query) < 2:
        raise ValueError('Enter a Canadian city or town.')
    response = requests.get(
        'https://geocoding-api.open-meteo.com/v1/search',
        params={'name': query, 'count': 10, 'language': 'en', 'countryCode': 'CA'},
        timeout=12,
        headers={'User-Agent': 'OSS4008-Canada-Hospitals-CourseProject/1.0'},
    )
    response.raise_for_status()
    matches = response.json().get('results') or []
    if not matches:
        raise ValueError(f'No Canadian location found for "{query}". Try including the province.')
    exact = [m for m in matches if m.get('name', '').casefold() == query.split(',')[0].strip().casefold()]
    chosen = (exact or matches)[0]
    return (float(chosen['latitude']), float(chosen['longitude']),
            f"{chosen.get('name', query)}, {chosen.get('admin1', 'Canada')}")


@lru_cache(maxsize=1)
def load_hospitals():
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=['name', 'latitude', 'longitude', 'type', 'address', 'province']), \
               'The Statistics Canada reference CSV has not been prepared. Run the dataset setup cell in the Colab notebook.'
    data = pd.read_csv(DATA_FILE)
    for col in ('name', 'latitude', 'longitude'):
        if col not in data:
            raise ValueError(f'Healthcare dataset missing column: {col}')
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data = data.dropna(subset=['latitude', 'longitude'])
    data = data[data['latitude'].between(41, 84) & data['longitude'].between(-142, -52)]
    return data, None


def haversine_km(lat, lon, latitudes, longitudes):
    import numpy as np
    p1 = np.radians(lat)
    p2 = np.radians(latitudes.to_numpy(dtype=float))
    dlat = p2 - p1
    dlon = np.radians(longitudes.to_numpy(dtype=float) - lon)
    a = np.sin(dlat / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1, np.sqrt(a)))


def get_hospitals(lat, lon, radius_km):
    data, error = load_hospitals()
    if error or data.empty:
        return data, error
    selected = data.copy()
    selected['distance_km'] = haversine_km(lat, lon, selected.latitude, selected.longitude)
    selected = selected[selected.distance_km <= radius_km].sort_values('distance_km')
    return selected, None


def read_predictions(path):
    if not path:
        return []
    if Path(path).stat().st_size > 8_000_000:
        raise ValueError('Prediction file exceeds 8 MB.')
    with open(path, encoding='utf-8') as file:
        obj = json.load(file)
    if obj.get('type') != 'FeatureCollection':
        raise ValueError('Prediction file must be a GeoJSON FeatureCollection.')
    features = obj.get('features', [])
    if len(features) > 3000:
        raise ValueError('Prediction file exceeds 3,000 features.')
    points = []
    for f in features:
        geom = f.get('geometry') or {}
        if geom.get('type') != 'Point' or len(geom.get('coordinates', [])) < 2:
            continue
        lon, lat = map(float, geom['coordinates'][:2])
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        raw_conf = (f.get('properties') or {}).get('confidence')
        confidence = float(raw_conf) if raw_conf is not None else None
        if confidence is not None and not (0 <= confidence <= 1):
            continue
        points.append((lat, lon, confidence))
    return points


def render_map(lat=DEFAULT_CENTER[0], lon=DEFAULT_CENTER[1], radius_km=25,
               include_reference=True, predictions_file=None, location_label='Halifax, Nova Scotia'):
    lat, lon, radius_km = validate_coordinates(lat, lon, radius_km)
    zoom = 11 if radius_km <= 30 else 9
    map_obj = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='CartoDB positron', control_scale=True)
    folium.TileLayer('OpenStreetMap', name='Street map').add_to(map_obj)
    folium.Circle([lat, lon], radius=radius_km * 1000, color='#1763ae',
                  weight=2, fill=False, tooltip='Study area').add_to(map_obj)
    total = 0
    error = None
    if include_reference:
        refs, error = get_hospitals(lat, lon, radius_km)
        total = len(refs)
        group = folium.FeatureGroup(name='Hospital references (Statistics Canada)', show=True)
        cluster = MarkerCluster().add_to(group)
        for _, row in refs.head(MAX_MARKERS).iterrows():
            title = html.escape(str(row['name']))
            address = html.escape(str(row.get('address', '')))
            folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=5,
                color='#b91c1c', fill=True, fill_opacity=0.85,
                tooltip=title, popup=f'<strong>{title}</strong><p>{address}</p><small>Statistics Canada reference — not an AI prediction</small>').add_to(cluster)
        group.add_to(map_obj)
    pred = read_predictions(predictions_file)
    if pred:
        layer = folium.FeatureGroup(name='Uploaded model predictions', show=True)
        for plat, plon, score in pred:
            score_text = f'{score:.2f}' if score is not None else 'not provided'
            folium.CircleMarker([plat, plon],radius=7,color='#16803d',fill=True,fill_opacity=0.9,
                tooltip=f'Model prediction — confidence {score_text}').add_to(layer)
        layer.add_to(map_obj)
    Fullscreen().add_to(map_obj)
    Draw(export=False, draw_options={'polyline':False,'circle':False,'circlemarker':False,'marker':False},
         edit_options={'edit':False}).add_to(map_obj)
    folium.LayerControl(collapsed=False).add_to(map_obj)
    status = (f'**{location_label}** · {total} hospital reference records within {radius_km:g} km · '
              f'{len(pred)} uploaded model predictions.  '
              'Reference records come from Statistics Canada; no model inference was performed.')
    if total > MAX_MARKERS:
        status += f' Showing the nearest {MAX_MARKERS} reference markers.'
    if error:
        status += f'  {error}'
    return map_obj._repr_html_(), status


def handle_search(place, radius, show_refs, prediction_file):
    try:
        lat, lon, name = search_place(place)
        view, status = render_map(lat, lon, radius, show_refs, prediction_file, name)
        return view, status, lat, lon, name
    except Exception as exc:
        return gr.skip(), f'**Location search failed:** {type(exc).__name__}: {exc}', gr.skip(), gr.skip(), gr.skip()


def handle_refresh(lat, lon, label, radius, show_refs, prediction_file):
    try:
        return render_map(lat, lon, radius, show_refs, prediction_file, label)
    except Exception as exc:
        return gr.skip(), f'**Map update failed:** {type(exc).__name__}: {exc}'


def assistant_reply(message, history, radius, show_refs, prediction_file):
    """Handle simple navigation commands. Does not call an LLM or infer from imagery."""
    history = list(history or [])
    query = str(message or '').strip()
    if not query:
        return history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), ''
    history.append({'role':'user','content': query})
    import re
    place = re.sub(r'^(please\s+)?(find|show|locate|map|search|go to)(\s+for)?\s+(hospitals?\s+)?(in|near|around|at)?\s*', '', query, flags=re.I).strip()
    if not place or re.search(r'^(help|what can you do|how does this work)\??$',query, re.I):
        text = 'I can navigate the map to a Canadian city or town. Try “Find hospitals in Ottawa”. Hospital markers are Statistics Canada reference records, not detections from imagery.'
        history.append({'role':'assistant','content':text})
        return history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), ''
    view, status, lat, lon, name = handle_search(place, radius, show_refs, prediction_file)
    if isinstance(lat, (int,float)):
        response = f'Map moved to {name}. The markers show known hospital facilities within {radius:g} km. Image-based detection is not connected yet.'
    else:
        response = f'I could not find that Canadian location. Please try the town name and province. {status}'
    history.append({'role':'assistant','content':response})
    return history, view, status, lat, lon, name, ''

CUSTOM_CSS = """
/* Desktop application shell: keep the map, controls, and assistant in one viewport. */
.gradio-container {max-width: 1800px !important; margin: 0 auto !important; padding: 9px 22px !important;}
#page-title {margin: 1px 0 8px !important; padding: 0 !important;}
#page-title h1 {font-size: 1.5rem !important; line-height: 1.35 !important; margin: 0 !important; letter-spacing: -.025em;}
#main-layout {gap: 14px !important; align-items: stretch !important;}
#map-pane {min-width: 0 !important; gap: 8px !important;}
#search-group {padding: 6px !important; border-radius: 11px !important;}
#search-fields {gap: 7px !important; align-items: center !important;}
#search-fields > div:first-child {flex: 1 1 auto !important; min-width: 0 !important;}
#place-field {min-width: 0 !important;}
#place-field textarea, #place-field input {min-height: 38px !important;}
#place-button {min-width: 96px !important; width: 110px !important; max-width: 116px !important; height: 40px !important; flex: 0 0 110px !important;}
#map-box {overflow: hidden !important; min-height: 0 !important;}
#map-box iframe {width: 100% !important; height: calc(100dvh - 255px) !important; min-height: 310px !important; border: none !important; border-radius: 10px !important; display: block !important;}
#map-status {font-size: .76rem !important; opacity: .8; max-height: 36px; overflow-y: auto;}
#map-status p {margin: 0 !important;}
#quick-settings {padding: 4px 8px !important; gap: 10px !important; align-items: center !important; border-radius: 9px !important;}
#quick-settings .wrap {gap: 5px !important;}
#settings-label {min-width: 65px !important; flex: 0 0 auto !important;}
#settings-label p {font-weight: 600; font-size: .86rem; margin: 0 !important;}
#radius-control {min-width: 160px !important; flex: 1 1 auto !important;}
#radius-control label {font-size: .79rem !important;}
#reference-toggle {min-width: 150px !important; flex: 0 1 210px !important;}
#apply-button {min-width: 75px !important; flex: 0 0 84px !important;}
#assistant-panel {min-width: 285px !important; gap: 7px !important;}
#assistant-heading {margin: 0 !important;}
#assistant-heading h3 {margin: 0 !important; font-size: 1.04rem !important;}
#assistant-chat {min-height: 0 !important;}
#assistant-chat .message {font-size: .9rem !important;}
#chat-compose {gap: 6px !important; align-items: center !important;}
#chat-message {min-width: 0 !important;}
#send-button {min-width: 70px !important; max-width: 84px !important; height: 40px !important;}
#research-options {font-size: .8rem !important;}
@media (min-width: 1024px) and (min-height: 650px) {
  body {overflow-y: hidden !important;}
  #assistant-chat {height: calc(100dvh - 172px) !important; max-height: calc(100dvh - 172px) !important;}
}
@media (max-width: 1023px), (max-height: 649px) {
  body {overflow-y: auto !important;}
  #map-box iframe {height: 440px !important;}
  #assistant-chat {height: 320px !important;}
  #quick-settings {flex-wrap: wrap !important;}
}
"""

with gr.Blocks(title='Hospital infrastructure in Canada', fill_width=True, fill_height=True,
               **({'css': CUSTOM_CSS, 'theme': gr.themes.Soft()} if int(gr.__version__.split('.')[0]) < 6 else {})) as demo:
    gr.Markdown('# Hospital infrastructure in Canada', elem_id='page-title')
    latitude = gr.State(DEFAULT_CENTER[0])
    longitude = gr.State(DEFAULT_CENTER[1])
    location_label = gr.State('Halifax, Nova Scotia')
    with gr.Row(equal_height=False, elem_id='main-layout'):
        with gr.Column(scale=7, elem_id='map-pane'):
            with gr.Group(elem_id='search-group'):
                with gr.Row(elem_id='search-fields'):
                    search_box = gr.Textbox(label='City or town', show_label=False,
                        placeholder='Search any Canadian city or town', value='Halifax',
                        scale=8, container=False, elem_id='place-field')
                    search_button = gr.Button('Search', variant='primary', scale=0,
                        min_width=100, elem_id='place-button')
            map_html = gr.HTML(elem_id='map-box', container=False)
            status = gr.Markdown(elem_id='map-status')
            with gr.Row(elem_id='quick-settings'):
                gr.Markdown('**Map settings**', elem_id='settings-label')
                radius = gr.Slider(1, 100, value=25, step=1, label='Radius (km)',
                                   elem_id='radius-control', container=False)
                references = gr.Checkbox(value=True, label='Hospital markers',
                                          elem_id='reference-toggle', container=False)
                update = gr.Button('Apply', variant='secondary', elem_id='apply-button',
                                   min_width=70)
            with gr.Accordion('Advanced · model prediction overlay', open=False,
                              elem_id='research-options'):
                predictions = gr.File(label='Import model output (GeoJSON)',
                                      file_types=['.geojson', '.json'], type='filepath')
                gr.Markdown('Displays georeferenced predictions from a separately trained model. The app does not currently run image inference.')
        with gr.Column(scale=3, elem_id='assistant-panel'):
            gr.Markdown('### Map assistant', elem_id='assistant-heading')
            chat = gr.Chatbot(label='Conversation', height=415, elem_id='assistant-chat',
                value=[{'role':'assistant', 'content':'Where would you like to look? Try “Find hospitals in Ottawa”.'}])
            with gr.Row(elem_id='chat-compose'):
                command = gr.Textbox(label='Message', show_label=False,
                    placeholder='Ask to find a place…', lines=1, container=False,
                    elem_id='chat-message', scale=6)
                send = gr.Button('Send', variant='primary', scale=0,
                                 min_width=74, elem_id='send-button')
    inputs = [radius, references, predictions]
    search_button.click(handle_search, [search_box,*inputs], [map_html,status,latitude,longitude,location_label])
    search_box.submit(handle_search, [search_box,*inputs], [map_html,status,latitude,longitude,location_label])
    update.click(handle_refresh,[latitude,longitude,location_label,*inputs],[map_html,status])
    chat_outputs = [chat,map_html,status,latitude,longitude,location_label,command]
    send.click(assistant_reply,[command,chat,*inputs],chat_outputs)
    command.submit(assistant_reply,[command,chat,*inputs],chat_outputs)
    demo.load(lambda: render_map(),outputs=[map_html,status])

if __name__ == '__main__':
    demo.launch(**({'css': CUSTOM_CSS, 'theme': gr.themes.Soft()} if int(gr.__version__.split('.')[0]) >= 6 else {}))
