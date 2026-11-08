"""Offline smoke tests for the hospital map application. Run with python test_app.py."""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('hospital_app', Path(__file__).with_name('app.py'))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
assert app.validate_coordinates(44.6488, -63.5752, 8) == (44.6488, -63.5752, 8)
html, status = app.render_map(44.6488, -63.5752, 8, include_reference=False)
assert 'leaflet' in html.lower()
assert '8 km' in status
history,*_ = app.assistant_reply('help', [], 8, False, None)
assert history[-1]['role'] == 'assistant'
assert 'Canada' in history[-1]['content']
assert 'OSS 4008 · Remote Sensing | Explore' not in app.APP_SOURCE if hasattr(app, 'APP_SOURCE') else True
assert '#search-group' in app.CUSTOM_CSS
assert 'height: calc(100dvh - 255px)' in app.CUSTOM_CSS
assert 'Advanced · model prediction overlay' in Path(__file__).with_name('app.py').read_text()
print('PASS: UI constructed, map generated, coordinates validated, navigation help replied.')
