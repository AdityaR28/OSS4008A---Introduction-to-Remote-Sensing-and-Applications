"""OSS4008 Phase III: interactive Ottawa building detection pilot.
Run: streamlit run app.py
"""
import io
import math
import time

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from folium.plugins import Draw
from PIL import Image
from pyproj import Transformer
from streamlit_folium import st_folium

SERVICE = ("https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/"
           "AerialImagery/GEO_Imagery_Data_Service_2023to2027/ImageServer")
MODEL_REPO = "keremberke/yolov8s-building-segmentation"
DEFAULT_BOUNDS = [-75.926374, 45.311929, -75.923456, 45.314185]
MAX_TILES = 16
RESOLUTION = 0.30  # projected metres per pixel, EPSG:3857
TILE_SIZE = 512
TO_MERC = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
TO_GEO = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)

st.set_page_config(page_title='Residential AI Mapper | OSS4008', page_icon='🏠', layout='wide')
st.title('🏠 Residential AI Mapper')
st.caption('OSS4008A · Phase III · Ottawa/Kanata pilot · Candidate building detection')

@st.cache_resource(show_spinner='Loading pretrained building model…')
def load_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    path = hf_hub_download(repo_id=MODEL_REPO, filename='best.pt')
    return YOLO(path)


def download_tile(x0, y0, x1, y1, width, height):
    params = {
        'f': 'image', 'bbox': f'{x0},{y0},{x1},{y1}',
        'bboxSR': 3857, 'imageSR': 3857,
        'size': f'{width},{height}', 'format': 'jpg',
        'pixelType': 'U8', 'interpolation': '+RSP_BilinearInterpolation',
    }
    response = requests.get(f'{SERVICE}/exportImage', params=params, timeout=60)
    response.raise_for_status()
    if not response.headers.get('content-type', '').lower().startswith('image/'):
        raise ValueError('Imagery server did not return an image: ' + response.text[:160])
    return Image.open(io.BytesIO(response.content)).convert('RGB')


def bbox_from_draw(output):
    drawings = output.get('all_drawings') or []
    for feature in reversed(drawings):
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Polygon':
            ring = geometry['coordinates'][0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            return [min(lons), min(lats), max(lons), max(lats)]
    return None


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / max(aa + bb - intersection, 1e-9)


def deduplicate(items, threshold=0.45):
    kept = []
    for item in sorted(items, key=lambda x: x['confidence'], reverse=True):
        if all(iou(item['mercator_bbox'], prev['mercator_bbox']) < threshold for prev in kept):
            kept.append(item)
    return kept


def detect(bounds, confidence, progress):
    west, south, east, north = bounds
    xmin, ymin = TO_MERC.transform(west, south)
    xmax, ymax = TO_MERC.transform(east, north)
    cols = math.ceil((xmax-xmin) / (RESOLUTION*TILE_SIZE))
    rows = math.ceil((ymax-ymin) / (RESOLUTION*TILE_SIZE))
    total = rows * cols
    if total > MAX_TILES:
        raise ValueError(f'Selection requires {total} image tiles; draw a smaller rectangle (maximum {MAX_TILES}).')
    model = load_model()
    detections = []
    for row in range(rows):
        for col in range(cols):
            x0 = xmin + col*RESOLUTION*TILE_SIZE
            x1 = min(x0 + RESOLUTION*TILE_SIZE, xmax)
            y1 = ymax - row*RESOLUTION*TILE_SIZE
            y0 = max(y1 - RESOLUTION*TILE_SIZE, ymin)
            width = max(1, round((x1-x0)/RESOLUTION))
            height = max(1, round((y1-y0)/RESOLUTION))
            img = download_tile(x0, y0, x1, y1, width, height)
            pred = model.predict(source=np.asarray(img), conf=confidence,
                                 imgsz=640, verbose=False)[0]
            for box in pred.boxes:
                px0, py0, px1, py1 = box.xyxy[0].tolist()
                gx0 = x0 + px0/width*(x1-x0)
                gx1 = x0 + px1/width*(x1-x0)
                gy1 = y1 - py0/height*(y1-y0)
                gy0 = y1 - py1/height*(y1-y0)
                # Exclude degenerate boxes; keep all predictions as candidates.
                if gx1 <= gx0 or gy1 <= gy0:
                    continue
                lon0, lat0 = TO_GEO.transform(gx0, gy0)
                lon1, lat1 = TO_GEO.transform(gx1, gy1)
                detections.append({
                    'confidence': round(float(box.conf[0]), 4),
                    'mercator_bbox': (gx0, gy0, gx1, gy1),
                    'west': lon0, 'south': lat0, 'east': lon1, 'north': lat1,
                })
            progress.progress(((row*cols+col+1)/total), text=f'Analyzed {row*cols+col+1}/{total} tiles')
    return deduplicate(detections), total, len(detections)


if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'bounds' not in st.session_state:
    st.session_state.bounds = DEFAULT_BOUNDS
if 'stats' not in st.session_state:
    st.session_state.stats = None

left, right = st.columns([2.3, 1], gap='large')
with right:
    st.subheader('Detection controls')
    st.write('Draw a **rectangle** on the map, then select **Use drawn rectangle**.')
    confidence = st.slider('Minimum confidence', 0.10, 0.90, 0.45, 0.05)
    st.info('The model detects candidate **buildings**. Residential use is not verified.')
    st.caption('Small selections work best; processing is limited to 16 tiles per run.')

with left:
    st.subheader('Interactive aerial map')
    fmap = folium.Map(location=[45.3131, -75.9249], zoom_start=17,
                      tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                      attr='Tiles © Esri and contributors')
    Draw(export=False, draw_options={
        'polyline': False, 'polygon': False, 'circle': False,
        'circlemarker': False, 'marker': False, 'rectangle': True,
    }, edit_options={'edit': True, 'remove': True}).add_to(fmap)
    current = st.session_state.bounds
    folium.Rectangle(bounds=[[current[1], current[0]], [current[3], current[2]]],
                     color='#18a999', weight=2, fill=False,
                     tooltip='Current analysis selection').add_to(fmap)
    for i, d in enumerate(st.session_state.detections, 1):
        folium.Rectangle(bounds=[[d['south'], d['west']], [d['north'], d['east']]],
                         color='#f43f5e', weight=2, fill=True, fill_opacity=0.08,
                         tooltip=f'Candidate {i} · confidence {d["confidence"]:.2f}').add_to(fmap)
    map_output = st_folium(fmap, height=540, use_container_width=True,
                           key='aerial_map', returned_objects=['all_drawings'])
    if st.button('Use drawn rectangle', type='secondary'):
        new_bounds = bbox_from_draw(map_output or {})
        if new_bounds:
            st.session_state.bounds = new_bounds
            st.session_state.detections = []
            st.session_state.stats = None
            st.rerun()
        else:
            st.warning('Use the rectangle tool on the map first.')

with right:
    west, south, east, north = st.session_state.bounds
    st.caption(f'Selection: {west:.6f}, {south:.6f} to {east:.6f}, {north:.6f}')
    if st.button('🔎 Detect Homes / Buildings', type='primary', use_container_width=True):
        progress = st.progress(0, text='Preparing analysis…')
        try:
            start = time.time()
            found, tiles, raw = detect(st.session_state.bounds, confidence, progress)
            st.session_state.detections = found
            st.session_state.stats = (tiles, raw, round(time.time()-start, 1))
            st.rerun()
        except Exception as exc:
            st.error(f'Detection could not finish: {exc}')
        finally:
            progress.empty()
    if st.session_state.stats:
        tiles, raw, seconds = st.session_state.stats
        st.metric('Candidate buildings (after overlap filtering)', len(st.session_state.detections))
        st.caption(f'{raw} raw predictions · {tiles} tiles · {seconds}s')
        st.download_button('Download detections CSV',
            pd.DataFrame([{k:v for k,v in d.items() if k!='mercator_bbox'}
                          for d in st.session_state.detections]).to_csv(index=False),
            file_name='ottawa_candidate_buildings.csv', mime='text/csv')

st.divider()
st.caption('Research prototype · Ontario imagery service (2023–2027 collection) · YOLOv8 building segmentation checkpoint. '
           'Predictions are unverified and must not be used for emergency decisions or official building inventories.')
