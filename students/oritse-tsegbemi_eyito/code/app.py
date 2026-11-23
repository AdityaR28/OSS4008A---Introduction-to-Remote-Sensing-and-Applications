"""Shared Ships and Ports dashboard. Launch with streamlit run code/app.py."""
import base64
import html
import json
import math
import os
from datetime import date
from pathlib import Path

import folium
from folium.plugins import Draw, MarkerCluster
import streamlit as st
from streamlit_folium import st_folium

from national_data import read_map_references, filter_ports
from project_chat import ask_assistant, ChatUnavailable, PROVINCES
from public_backend import (
    EXAMPLES, PRESETS, drawing_bounds, load_example, load_model, map_overlay,
    result_context, run_live, validate_selection)


PROJECT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Ships and ports in Canada", page_icon="⚓", layout="wide")


def setting(name, default=""):
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


@st.cache_data
def references():
    return read_map_references(PROJECT)


@st.cache_data
def example(location):
    return load_example(PROJECT, location)


@st.cache_resource
def detector():
    return load_model(PROJECT)


@st.cache_data
def preview_overlay(stem, image, metadata):
    return map_overlay({"files": {"png": image}, "metadata": metadata})


def select_example(location):
    st.session_state.result = example(location)
    st.session_state.selection = PRESETS[location].copy()
    st.session_state.map_revision += 1
    st.session_state.map_center = None
    st.session_state.drawing_signature = None


def clear_chat():
    st.session_state.messages = []
    st.session_state.pop("chat_model", None)


def apply_chat_actions(actions):
    confirmations = []
    for action in actions:
        if action["name"] == "show_example":
            select_example(action["location"])
            confirmations.append("Opened the saved " + action["location"] + " example.")
        if action["name"] == "filter_ports":
            st.session_state.province = action["province"]
            st.session_state.port_query = action["query"]
            name = next(name for name, code in PROVINCES.items() if code == action["province"])
            confirmations.append("Port reference filter set to " + name + ".")
    return confirmations


if "result" not in st.session_state:
    st.session_state.update(
        result=example("Halifax"), selection=PRESETS["Halifax"].copy(),
        map_revision=0, map_center=None, drawing_signature=None, messages=[],
        province="All", port_query="", facility="All", preset="Halifax")

if "pending_chat" in st.session_state:
    pending = st.session_state.pop("pending_chat")
    response = pending["response"]
    confirmations = apply_chat_actions(response["actions"])
    answer = "\n\n".join([response["text"], *confirmations]).strip()
    st.session_state.messages.extend([
        {"role": "user", "content": pending["prompt"]},
        {"role": "assistant", "content": answer}])
    st.session_state.messages = st.session_state.messages[-24:]
    st.session_state.chat_model = response["model"]

st.markdown("""
<style>
.stApp {background:#f2f6f7;color:#17323f}
.block-container {padding-top:1.5rem;padding-bottom:1.5rem;max-width:1600px}
h1,h2,h3 {color:#17323f}
[data-testid="stMetricValue"] {color:#126d79}
button[kind="primary"] {background:#126d79;border-color:#126d79}
[data-testid="stChatMessage"] {background:#fff;border:1px solid #dce7e9;border-radius:12px}
.project-label {font-size:.8rem;letter-spacing:.12em;color:#52717c;margin-bottom:.3rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="project-label">OSS4008A · CANADA</div>', unsafe_allow_html=True)
st.title("Ships and ports")
st.caption("Explore Canadian facilities, inspect satellite predictions, and ask about the results.")

controls, map_column = st.columns([.38, .62], gap="large")
records = references()
with controls:
    assistant_tab, analysis_tab, ports_tab, downloads_tab = st.tabs(
        ["Assistant", "Analyse", "Ports", "Downloads"])

    with ports_tab:
        st.selectbox("Province or territory", ["All"] + sorted({row["province"] for row in records}),
                     key="province", format_func=lambda code: next(
                         name for name, value in PROVINCES.items() if value == code))
        st.selectbox("Facility type", ["All"] + sorted({row["subtype"] for row in records}), key="facility")
        st.text_input("Port or marina name", key="port_query", max_chars=100)
        rows = filter_ports(records, st.session_state.province, st.session_state.facility,
                            st.session_state.port_query)
        st.caption(f"{len(rows):,} of {len(records):,} published references")
        reference_by_id = {row["reference_id"]: row for row in rows}
        port = st.selectbox(
            "Select a reference", [""] + list(reference_by_id),
            format_func=lambda key: "Choose a facility" if not key else
            reference_by_id[key]["name"] + " · " + reference_by_id[key]["province"])
        if st.button("Go to facility", disabled=not port, use_container_width=True):
            row = reference_by_id[port]
            lat, lon = float(row["latitude"]), float(row["longitude"])
            dx = .009 / math.cos(math.radians(lat))
            st.session_state.selection = [lon - dx, lat - .009, lon + dx, lat + .009]
            st.session_state.map_center = [lat, lon]
            st.session_state.map_revision += 1
            st.rerun()
        show_references = st.checkbox("Show facility references", value=True)
        st.caption("Contains information licensed under the Open Government Licence – Canada. "
                   "Blue markers identify published facilities. Source coverage varies.")

    displayed = st.session_state.result
    metadata = displayed["metadata"]
    features = displayed["geojson"]["features"]
    scores = [float(feature["properties"]["confidence"]) for feature in features]

    with analysis_tab:
        first, second = st.columns(2)
        first.metric("Predicted ships", len(features))
        second.metric("Mean confidence", f"{sum(scores)/len(scores):.0%}" if scores else "—")
        st.caption("These scores summarize the displayed run. Independent Canadian validation is pending.")
        buttons = st.columns(2)
        for location, button in zip(EXAMPLES, buttons):
            button.button(location + " example", on_click=select_example, args=(location,),
                          use_container_width=True)
        st.selectbox("Live analysis area", list(PRESETS), key="preset")
        if st.button("Use this area", use_container_width=True):
            st.session_state.selection = PRESETS[st.session_state.preset].copy()
            st.session_state.map_center = None
            st.session_state.map_revision += 1
            st.rerun()
        ee_project = setting("EE_PROJECT", "fluent-edition-509318-a6")
        ee_secret = setting("EE_SERVICE_ACCOUNT_JSON")
        live_enabled = bool(ee_secret) or os.environ.get("ALLOW_COLAB_AUTH") == "1"
        with st.form("live_analysis"):
            start = st.date_input("From", date(2025, 6, 1))
            end = st.date_input("To, exclusive", date(2025, 9, 1))
            confidence = st.slider("Minimum confidence", .05, .95, .55, .05)
            clouds = st.slider("Maximum scene cloud percentage", 0, 100, 30)
            st.caption("Draw a rectangle on the map or edit the coordinates below. "
                       "The selection may cover up to 16 km².")
            west, south, east, north = st.session_state.selection
            with st.expander("Area coordinates"):
                west = st.number_input("West", value=west, format="%.5f")
                south = st.number_input("South", value=south, format="%.5f")
                east = st.number_input("East", value=east, format="%.5f")
                north = st.number_input("North", value=north, format="%.5f")
            submitted = st.form_submit_button("Detect ships", type="primary",
                                             use_container_width=True, disabled=not live_enabled)
        if not live_enabled:
            st.info("Saved examples are ready. Live analysis will become available after the host connects Earth Engine.")
        if submitted:
            try:
                bounds = validate_selection([west, south, east, north])
                if start >= end:
                    raise ValueError("Choose an end date after the start date.")
                with st.spinner("Selecting a clear satellite scene and detecting ships…"):
                    model, device = detector()
                    result = run_live(PROJECT, model, device, bounds, start.isoformat(), end.isoformat(),
                                      confidence, clouds, ee_project, ee_secret)
                st.session_state.result = result
                st.session_state.selection = bounds
                st.session_state.map_center = None
                st.session_state.map_revision += 1
                st.rerun()
            except ValueError as error:
                st.error(str(error))
            except Exception:
                st.error("Live processing could not finish. Try a smaller area or different dates. "
                         "Your previously displayed result is still available.")

    with assistant_tab:
        st.subheader("Ask about the map")
        api_key = setting("OPENROUTER_API_KEY")
        st.caption("Uses OpenRouter's free models. Your question and the displayed result summary "
                   "are sent to the model provider.")
        if not api_key:
            st.info("AI chat is awaiting the host's OpenRouter connection.")
        with st.container(height=340):
            if not st.session_state.messages:
                st.markdown("Try asking **How many ships are shown?**, **What does confidence mean?**, "
                            "or **Show Vancouver**.")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        prompt = st.chat_input("Ask about ships, ports, or this result", disabled=not bool(api_key),
                               max_chars=2000)
        if prompt:
            context = result_context(displayed, len(rows), st.session_state.province,
                                     st.session_state.port_query, st.session_state.selection)
            try:
                with st.spinner("Checking the displayed results…"):
                    response = ask_assistant(prompt, st.session_state.messages, context, api_key)
                # Port widgets were rendered earlier in this run. Apply their new values
                # at the next run's start, before those widgets are constructed.
                st.session_state.pending_chat = {"prompt": prompt, "response": response}
                st.rerun()
            except ChatUnavailable as error:
                st.warning(str(error))
        if st.session_state.get("chat_model"):
            st.caption("Last answer · " + st.session_state.chat_model)
        st.button("Clear conversation", on_click=clear_chat)
        with st.expander("Project guide"):
            st.write("Open Analyse to choose a saved example or run detection. Use Ports to filter "
                     "the national reference layer. Draw a rectangle to select a live analysis area. "
                     "Click a red prediction to inspect its score. Downloads exports the displayed result.")

    with downloads_tab:
        st.write(displayed["label"])
        st.caption("Acquired " + metadata["acquired_utc"] + f" · confidence threshold {metadata['threshold']:.0%}")
        for extension, label, mime in [
                ("png", "Download image", "image/png"), ("csv", "Download table", "text/csv"),
                ("geojson", "Download map features", "application/geo+json"),
                ("json", "Download run details", "application/json")]:
            st.download_button(label, displayed["files"][extension],
                               file_name=displayed["stem"] + "." + extension, mime=mime,
                               use_container_width=True)
        st.image(displayed["files"]["png"], caption="Displayed prediction preview")

with map_column:
    left, right = st.columns([3, 1])
    left.subheader(displayed["label"])
    if right.button("Canada", use_container_width=True):
        st.session_state.map_center = [58, -96]
        st.session_state.map_revision += 1
        st.rerun()
    st.caption(f"{metadata['acquired_utc'][:10]} · Sentinel-2 · 10 m · "
               f"threshold {metadata['threshold']:.0%} · {len(features)} predictions")
    center = st.session_state.map_center
    mapping = folium.Map(location=center or [58, -96], zoom_start=3 if center == [58, -96] else 13,
                         tiles="OpenStreetMap", control_scale=True)
    image, overlay_bounds = preview_overlay(displayed["stem"], displayed["files"]["png"], metadata)
    folium.raster_layers.ImageOverlay(
        "data:image/png;base64," + base64.b64encode(image).decode("ascii"),
        bounds=overlay_bounds, name="Satellite prediction preview", opacity=1).add_to(mapping)
    folium.GeoJson(
        displayed["geojson"], name="Predicted ships",
        style_function=lambda feature: {"color": "#ed4c59", "weight": 2, "fillOpacity": .08},
        tooltip=folium.GeoJsonTooltip(fields=["detection_id", "confidence"],
                                     aliases=["Prediction", "Model confidence"]) if features else None,
        popup=folium.GeoJsonPopup(fields=["detection_id", "confidence", "acquired_utc"],
                                 aliases=["Prediction", "Confidence", "Acquired"]) if features else None).add_to(mapping)
    if show_references:
        cluster = MarkerCluster(name="Published port and marina references").add_to(mapping)
        for row in rows:
            popup = (html.escape(str(row["name"])) + "<br>" + html.escape(str(row["province"])) +
                     "<br>" + html.escape(str(row["subtype"])))
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])], radius=4,
                color="#2b789e", fill=True, fill_opacity=.7,
                tooltip=html.escape(str(row["name"])), popup=popup).add_to(cluster)
    west, south, east, north = st.session_state.selection
    folium.Rectangle([[south, west], [north, east]], color="#e39a30", fill=False,
                     tooltip="Requested detection area").add_to(mapping)
    Draw(export=False, draw_options={"polyline": False, "polygon": False, "circle": False,
                                     "circlemarker": False, "marker": False,
                                     "rectangle": {"shapeOptions": {"color": "#e39a30"}}},
         edit_options={"edit": False, "remove": True}).add_to(mapping)
    folium.LayerControl(collapsed=True).add_to(mapping)
    if center is None:
        mapping.fit_bounds([[south, west], [north, east]])
    response = st_folium(
        mapping, height=620, use_container_width=True,
        key="map_" + str(st.session_state.map_revision),
        returned_objects=["all_drawings", "last_clicked"])
    drawings = response.get("all_drawings")
    if drawings:
        signature = json.dumps(drawings[-1], sort_keys=True)
        if signature != st.session_state.drawing_signature:
            try:
                st.session_state.selection = drawing_bounds(drawings[-1])
                st.session_state.drawing_signature = signature
                st.rerun()
            except (ValueError, TypeError, KeyError):
                st.warning("Choose a rectangle of at most 16 km² inside the Canadian map extent.")
    clicked = response.get("last_clicked")
    if clicked:
        st.caption(f"Selected location · {clicked['lat']:.5f}, {clicked['lng']:.5f}")
    st.caption("Orange outlines the requested area. Red boxes show predictions from the displayed run. "
               "Small vessels may be unresolved at 10 m.")

st.divider()
st.button("Ships and ports", type="primary", on_click=select_example, args=("Halifax",))
