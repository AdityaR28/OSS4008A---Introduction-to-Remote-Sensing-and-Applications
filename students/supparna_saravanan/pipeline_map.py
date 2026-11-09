
import os
import json
import html
import requests
import pandas as pd

from shapely import wkt
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union

import os
from pathlib import Path
from dotenv import load_dotenv


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Load environment variables from this project's folder
load_dotenv(Path(__file__).resolve().parent / ".env")

MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

if not MAPBOX_TOKEN:
    raise ValueError(
        "MAPBOX_TOKEN not found. Set your Mapbox access token "
        "as an environment variable."
    )

OUTPUT_HTML = "pipeline_map.html"

CANADA_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/"
    "natural-earth-vector/master/geojson/"
    "ne_10m_admin_0_countries.geojson"
)

# Approximate Canadian navigation bounds
CANADA_BOUNDS = [
    [-145, 49],
    [-45, 85]
]


# --------------------------------------------------
# 1. CONVERT CLEANED DATAFRAME TO GEOJSON
# --------------------------------------------------

def dataframe_to_geojson(df):

    features = []

    for _, row in df.iterrows():

        geometry = row.get("geometry")

        # Skip records without geometry
        if geometry is None or pd.isna(geometry):
            continue

        # Convert WKT to Shapely geometry
        try:
            if isinstance(geometry, str):
                geom = wkt.loads(geometry)
            else:
                geom = geometry

            if geom is None or geom.is_empty:
                continue

        except Exception as e:
            print(f"Skipping invalid geometry: {e}")
            continue

        # Extract pipeline information
        properties = {}

        columns = [
            "pipeline_name",
            "company",
            "parent_company",
            "province",
            "location_description",
            "from_location",
            "to_location",
            "commodity",
            "pipeline_length_km",
            "capacity",
            "status",
            "start_date",
            "end_date",
            "segment_number",
            "licence_number",
            "project_number",
            "authority_type",
            "data_source",
            "source_dataset",
            "source_record_id"
        ]

        for col in columns:
            if col in df.columns:
                value = row[col]

                if pd.isna(value):
                    value = None
                elif hasattr(value, "item"):
                    value = value.item()
                elif not isinstance(value, (str, int, float, bool)):
                    value = str(value)

                properties[col] = value

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": properties
        }

        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


# --------------------------------------------------
# 2. DOWNLOAD CANADA BOUNDARY
# --------------------------------------------------

def get_canada_boundary():

    print("Downloading Canada boundary...")

    response = requests.get(
        CANADA_GEOJSON_URL,
        timeout=60
    )

    response.raise_for_status()

    world_data = response.json()

    canada_features = [
        feature
        for feature in world_data["features"]
        if feature["properties"].get("ADMIN") == "Canada"
    ]

    if not canada_features:
        raise ValueError("Could not find Canada in boundary dataset.")

    canada_geometry = unary_union([
        shape(feature["geometry"])
        for feature in canada_features
    ])

    return canada_geometry


# --------------------------------------------------
# 3. CREATE GREY MASK OUTSIDE CANADA
# --------------------------------------------------

def create_outside_mask(canada_geometry):

    # Large world polygon
    world = box(-180, -85, 180, 85)

    # Remove Canada from the world polygon.
    # Canada becomes the transparent hole.
    outside_canada = world.difference(canada_geometry)

    return {
        "type": "Feature",
        "geometry": mapping(outside_canada),
        "properties": {
            "name": "Outside Canada"
        }
    }


# --------------------------------------------------
# 4. GENERATE INTERACTIVE MAP HTML
# --------------------------------------------------

def generate_pipeline_map(
    pipelines_df,
    output_html=OUTPUT_HTML
):

    print("Preparing pipeline geometries...")

    pipeline_geojson = dataframe_to_geojson(pipelines_df)

    print(
        f"Successfully converted "
        f"{len(pipeline_geojson['features'])} pipeline features."
    )

    # Get Canada geometry
    canada_geometry = get_canada_boundary()

    canada_geojson = {
        "type": "Feature",
        "geometry": mapping(canada_geometry),
        "properties": {
            "name": "Canada"
        }
    }

    # Generate grey outside mask
    outside_mask = create_outside_mask(canada_geometry)

    # Calculate map center and bounds
    min_lon, min_lat, max_lon, max_lat = canada_geometry.bounds

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # Serialize GeoJSON
    pipeline_json = json.dumps(pipeline_geojson)
    canada_json = json.dumps(canada_geojson)
    mask_json = json.dumps(outside_mask)

    # --------------------------------------------------
    # HTML + MAPBOX GL JS
    # --------------------------------------------------

    html_content = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Canadian Pipeline Network</title>

    <script src="https://api.mapbox.com/mapbox-gl-js/v3.9.4/mapbox-gl.js"></script>

    <link href="https://api.mapbox.com/mapbox-gl-js/v3.9.4/mapbox-gl.css"
          rel="stylesheet">

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            overflow: hidden;
        }

        #map {
            width: 100%;
            height: 100vh;
        }

        /* Map information panel */

        .map-title {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 10;

            background: white;
            padding: 15px 20px;
            border-radius: 8px;

            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }

        .map-title h2 {
            font-size: 18px;
            margin-bottom: 5px;
        }

        .map-title p {
            font-size: 12px;
            color: #666;
        }

        /* Legend */

        .legend {
            position: absolute;
            bottom: 30px;
            left: 20px;
            z-index: 10;

            background: white;
            padding: 15px;
            border-radius: 8px;

            box-shadow: 0 2px 10px rgba(0,0,0,0.15);

            font-size: 13px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin-top: 8px;
        }

        .legend-line {
            width: 25px;
            height: 4px;
            background: #e04444;
            margin-right: 10px;
        }

        .legend-mask {
            width: 25px;
            height: 14px;
            background: #999;
            margin-right: 10px;
        }

        /* Pipeline popup */

        .mapboxgl-popup-content {
            padding: 15px;
            border-radius: 8px;
            min-width: 220px;
            max-width: 320px;
        }

        .popup-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .popup-row {
            font-size: 12px;
            margin: 6px 0;
            color: #444;
        }

        .popup-row strong {
            color: #111;
        }

    </style>

</head>

<body>

    <div class="map-title">

        <h2>Canadian Pipeline Network</h2>

        <p>Interactive pipeline infrastructure map</p>

    </div>

    <div class="legend">

        <strong>Map Legend</strong>

        <div class="legend-item">
            <div class="legend-line"></div>
            Pipeline Network
        </div>

        <div class="legend-item">
            <div class="legend-mask"></div>
            Outside Canada
        </div>

    </div>

    <div id="map"></div>


    <script>

        // --------------------------------------------------
        // MAPBOX INITIALIZATION
        // --------------------------------------------------

        mapboxgl.accessToken = "__MAPBOX_TOKEN__";

        const map = new mapboxgl.Map({

            container: "map",

            style: "mapbox://styles/mapbox/light-v11",

            center: [__CENTER_LON__, __CENTER_LAT__],

            zoom: 3,

            minZoom: 2,

            maxZoom: 15,

            maxBounds: __CANADA_BOUNDS__,

            renderWorldCopies: false

        });


        map.addControl(
            new mapboxgl.NavigationControl(),
            "top-right"
        );

        map.addControl(
            new mapboxgl.FullscreenControl(),
            "top-right"
        );


        // --------------------------------------------------
        // GEOJSON DATA
        // --------------------------------------------------

        const pipelineData = __PIPELINE_GEOJSON__;

        const canadaData = __CANADA_GEOJSON__;

        const outsideMask = __OUTSIDE_MASK__;


        // --------------------------------------------------
        // ADD MAP LAYERS
        // --------------------------------------------------

        map.on("load", function() {


            // 1. Canada boundary

            map.addSource("canada", {

                type: "geojson",

                data: canadaData

            });


            // 2. Grey mask outside Canada

            map.addSource("outside-mask", {

                type: "geojson",

                data: outsideMask

            });


            map.addLayer({

                id: "outside-canada-fill",

                type: "fill",

                source: "outside-mask",

                paint: {

                    "fill-color": "#999999",

                    "fill-opacity": 0.78

                }

            });


            // 3. Canada outline

            map.addLayer({

                id: "canada-outline",

                type: "line",

                source: "canada",

                paint: {

                    "line-color": "#333333",

                    "line-width": 1.5,

                    "line-opacity": 0.9

                }

            });


            // 4. Pipeline GeoJSON

            map.addSource("pipelines", {

                type: "geojson",

                data: pipelineData

            });


            // 5. Pipeline lines

            map.addLayer({

                id: "pipeline-lines",

                type: "line",

                source: "pipelines",

                layout: {

                    "line-join": "round",

                    "line-cap": "round"

                },

                paint: {

                    "line-color": "#e04444",

                    "line-width": [

                        "interpolate",

                        ["linear"],

                        ["zoom"],

                        3, 1,

                        6, 2,

                        10, 3.5

                    ],

                    "line-opacity": 0.9

                }

            });


            // --------------------------------------------------
            // PIPELINE CLICK INTERACTION
            // --------------------------------------------------

            map.on("click", "pipeline-lines", function(e) {

                const properties =
                    e.features[0].properties;

                const name =
                    properties.pipeline_name || "Unknown Pipeline";

                const company =
                    properties.company || "Not available";

                const province =
                    properties.province || "Not available";

                const commodity =
                    properties.commodity || "Not available";

                const length =
                    properties.pipeline_length_km ?? "Not available";

                const capacity =
                    properties.capacity || "Not available";

                const status =
                    properties.status || "Not available";

                const from =
                    properties.from_location || "Not available";

                const to =
                    properties.to_location || "Not available";


                // Escape text before inserting into HTML
                function safe(value) {
                    return String(value)
                        .replaceAll("&", "&amp;")
                        .replaceAll("<", "&lt;")
                        .replaceAll(">", "&gt;")
                        .replaceAll('"', "&quot;")
                        .replaceAll("'", "&#039;");
                }


                const popupHTML = `

                    <div class="popup-title">
                        ${safe(name)}
                    </div>

                    <div class="popup-row">
                        <strong>Company:</strong>
                        ${safe(company)}
                    </div>

                    <div class="popup-row">
                        <strong>Province:</strong>
                        ${safe(province)}
                    </div>

                    <div class="popup-row">
                        <strong>Commodity:</strong>
                        ${safe(commodity)}
                    </div>

                    <div class="popup-row">
                        <strong>Length:</strong>
                        ${safe(length)} km
                    </div>

                    <div class="popup-row">
                        <strong>Capacity:</strong>
                        ${safe(capacity)}
                    </div>

                    <div class="popup-row">
                        <strong>Status:</strong>
                        ${safe(status)}
                    </div>

                    <div class="popup-row">
                        <strong>From:</strong>
                        ${safe(from)}
                    </div>

                    <div class="popup-row">
                        <strong>To:</strong>
                        ${safe(to)}
                    </div>

                `;


                new mapboxgl.Popup({

                    closeButton: true,

                    maxWidth: "320px"

                })

                .setLngLat(e.lngLat)

                .setHTML(popupHTML)

                .addTo(map);

            });


            // Pointer cursor when hovering over pipelines

            map.on("mouseenter", "pipeline-lines", function() {

                map.getCanvas().style.cursor = "pointer";

            });

            map.on("mouseleave", "pipeline-lines", function() {

                map.getCanvas().style.cursor = "";

            });


            // --------------------------------------------------
            // AUTOMATICALLY ZOOM TO CANADA
            // --------------------------------------------------

            map.fitBounds(

                [
                    [__MIN_LON__, __MIN_LAT__],
                    [__MAX_LON__, __MAX_LAT__]
                ],

                {
                    padding: 40,
                    duration: 1500
                }

            );

        });

    </script>

</body>

</html>
"""

    # --------------------------------------------------
    # INSERT PYTHON DATA INTO HTML
    # --------------------------------------------------

    replacements = {

        "__MAPBOX_TOKEN__": html.escape(MAPBOX_TOKEN, quote=True),

        "__CENTER_LON__": str(center_lon),

        "__CENTER_LAT__": str(center_lat),

        "__CANADA_BOUNDS__": json.dumps(CANADA_BOUNDS),

        "__PIPELINE_GEOJSON__": pipeline_json,

        "__CANADA_GEOJSON__": canada_json,

        "__OUTSIDE_MASK__": mask_json,

        "__MIN_LON__": str(min_lon),

        "__MIN_LAT__": str(min_lat),

        "__MAX_LON__": str(max_lon),

        "__MAX_LAT__": str(max_lat)

    }

    for key, value in replacements.items():

        html_content = html_content.replace(key, value)


    # --------------------------------------------------
    # SAVE HTML
    # --------------------------------------------------

    with open(output_html, "w", encoding="utf-8") as file:

        file.write(html_content)

    print("\nMap generated successfully!")

    print(f"Output: {os.path.abspath(output_html)}")

    print(f"Pipeline features: {len(pipeline_geojson['features'])}")

    return output_html
