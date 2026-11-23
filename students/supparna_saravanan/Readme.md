# Canadian Pipeline Network — Interactive Map with Local AI Chat

A Google Colab notebook that consolidates multiple Canadian pipeline datasets into a single standardized table (`Canadian_Pipelines_Clean.csv`), clips geometries to Canada, and serves an interactive **Mapbox GL JS** web map with a **Qwen2.5** chat sidebar for natural-language map filtering and `pandas` analytics.

## Features

- **Automated Dataset Ingestion & Standardization**: Downloads pipeline files from a public GitHub repository (`students/supparna_saravanan/data` in `AdityaR28/OSS4008A---Introduction-to-Remote-Sensing-and-Applications`), reads CSV/TXT, Excel, and GeoJSON/JSON sources, normalizes column names to snake_case, maps fields via `ALIASES` into a 24-column `STANDARD_COLUMNS` schema, extracts representative latitude/longitude coordinates from WKT/Shapely geometries, deduplicates records by `(source_dataset, source_record_id)`, and preserves original non-geometry attributes in `source_fields` as JSON.
- **Canada Boundary Clipping & Masking**: Fetches Natural Earth country polygons (`ne_10m_admin_0_countries.geojson`), clips pipeline geometries to Canada with a `0.02°` (~2 km) buffer tolerance after simplification, drops stray non-line artifacts, and adds an outside-Canada mask layer (`#999999` fill) with bounded map panning (`CANADA_BOUNDS`).
- **Interactive Mapbox Web Interface**: Displays pipeline lines with zoom-interpolated widths, click popups showing key pipeline attributes (name, company, province, commodity, length in km, capacity, status, and from/to locations), a collapsible sidebar, quick-prompt chips, and a filter status bar with a **Show all** reset button.
- **Local AI Chat + Pandas Analytics**: Runs `Qwen/Qwen2.5-3B-Instruct` in `float16` on GPU (or `Qwen/Qwen2.5-1.5B-Instruct` in `float32` on CPU) using greedy decoding and JSON-prefixed generation (`{`) to translate user prompts into `filter`, `reset`, `analyze`, or `chat` actions, while delegating statistical summaries (`overview`, `longest`, and `group` breakdowns by `province`, `commodity`, `company`, or `status`) to deterministic `pandas` aggregations.

## Prerequisites & Secrets

- **Runtime**: Google Colab with a **T4 GPU** (`Runtime → Change runtime type → T4 GPU`) recommended for faster Qwen inference.
- **Dependencies installed in notebook**: `geopandas`, `shapely`, `pandas`, `requests`, `openpyxl`, `transformers`, `accelerate` (plus `torch`, pre-installed in Colab).
- **Secrets & Environment Variables** (loaded via Colab `userdata.get` first, then `os.environ`):

| Variable | Required | Purpose |
|---|---|---|
| `MAPBOX_ACCESS_TOKEN` (or `MAPBOX_TOKEN`) | Yes | Public Mapbox token (`pk.…`) embedded into the generated map HTML. |
| `HF_TOKEN` | No | Optional Hugging Face token to speed up model downloads and reduce rate limits. |

## Notebook Structure

1. **Setup**
   - `1.1 Install dependencies`: Installs required Python packages.
   - `1.2 Imports`: Imports standard library, geospatial (`geopandas`, `shapely`), data (`pandas`), HTTP server/client, and Colab modules.
   - `1.3 Load datasets from GitHub`: Downloads supported files (`.csv`, `.txt`, `.xlsx`, `.xls`, `.geojson`, `.json`) into `/content/github_pipeline_data` and populates `file_paths` (5 files downloaded in the sample run: `Pipeline_AB`, `Pipeline_Canada`, `Pipeline_Canada2`, `Pipeline_NB`, `Pipeline_Segments_BC`).
   - `1.5 Load access tokens`: Retrieves and exports `MAPBOX_ACCESS_TOKEN` and optional `HF_TOKEN`.
2. **Data cleaning**
   - `2.1 Standard schema and column aliases`: Defines `STANDARD_COLUMNS` and `ALIASES`.
   - `2.2 File-reading helpers`: Implements `normalize_column`, `read_csv_safely` (tries `utf-8`, `cp1252`, `latin1` with `on_bad_lines="skip"`), and `read_file`.
   - `2.3 Geometry helpers`: Implements `parse_wkt_geometry`, `is_valid_geometry`, `normalize_geometry_value`, `get_geometries` (`the_geom`, `geom`, `wkt`, `geometry`, or GeoDataFrame geometry), and `representative_coordinate`.
   - `2.4 Record helpers`: Implements `get_source_record_id` (`objectid`, `pinfo_id`, `licence_number`, `project_number`, `segment_number`) and `serialize_source_fields`.
   - `2.5 Main cleaning function`: `clean_pipeline_dataset()` cleans, combines, deduplicates, trims strings, and writes `/content/data/Canadian_Pipelines_Clean.csv` (`56,997` rows × `24` columns in the sample run).
3. **Map builder**
   - `3.1 Map constants`: Sets `CANADA_GEOJSON_URL`, `CANADA_BOUNDS`, and `CLIP_TOLERANCE_DEG`.
   - `3.2 Canada boundary and clipping`: Implements `get_canada_boundary`, `_keep_lines`, and `build_canada_clipper`.
   - `3.3 GeoJSON conversion and outside-Canada mask`: Implements `dataframe_to_geojson` and `create_outside_mask`.
   - `3.4 HTML / JavaScript page template`: Defines `MAP_HTML_TEMPLATE` with Mapbox layers, popups, filter logic, chat client, and suggestions.
   - `3.5 Page generator`: `generate_pipeline_map()` clips geometries (`39` features in Canada from `237` evaluated in the sample run) and injects data and bounds into `MAP_HTML_TEMPLATE`.
4. **Local AI model**
   - `4.1 Load the model`: Loads `Qwen/Qwen2.5-3B-Instruct` on GPU or `Qwen/Qwen2.5-1.5B-Instruct` on CPU.
   - `4.2 Chat function`: `run_chat()` formats messages with the tokenizer chat template, prepends `{`, and generates JSON responses.
   - `analyze_pipeline_data()`: Computes overview metrics, top `N` longest segments, or grouped segment counts and total kilometers on `pipelines_df`.
5. **Run**
   - `5.1 Clean the datasets`: Runs `clean_pipeline_dataset(file_paths)` to create `pipelines_df` and `/content/data/Canadian_Pipelines_Clean.csv`.
   - `5.2 Launch the map`: Starts a `ThreadingHTTPServer` on port `8765` (`GET /` or `/index.html`, `POST /chat`, `POST /analyze`) and exposes it via `google.colab.output.serve_kernel_port_as_window(8765)`.

## Usage

1. Select a **T4 GPU** runtime in Colab and add `MAPBOX_ACCESS_TOKEN` (and optionally `HF_TOKEN`) to **Colab Secrets** with notebook access enabled.
2. Run the setup, data cleaning, map builder, and model loading cells in order.
3. Run **5.1 Clean the datasets** to generate `pipelines_df` and `/content/data/Canadian_Pipelines_Clean.csv`.
4. Run **5.2 Launch the map** and click the generated proxy link to open the interactive map in a new browser tab.
5. Click pipeline lines to inspect details, use the chat sidebar to filter by province, commodity, company, status, length, or top longest segments (e.g., *"Show oil pipelines in Alberta"* or *"Show the 5 longest pipelines"*), or ask for dataset summaries and breakdowns.

## Troubleshooting

- **Missing or empty GitHub datasets**: Check that the GitHub API URL in section `1.3` is reachable and contains supported files (`.csv`, `.txt`, `.xlsx`, `.xls`, `.geojson`, `.json`).
- **Mapbox token error or blank map**: Verify `MAPBOX_ACCESS_TOKEN` is set in Colab Secrets with notebook access toggled on and uses a valid public `pk.` token.
- **Slow chat responses**: Confirm a GPU runtime is active (`USE_GPU` / `torch.cuda.is_available()`); otherwise the notebook falls back to `Qwen/Qwen2.5-1.5B-Instruct` on CPU.
- **Filter not applied or parse error**: Rephrase the prompt clearly (for example, *"natural gas pipelines in British Columbia"*), or click **Show all** / say *"show all"* to reset filters.
- **Disconnected runtime or updated code/data**: Re-run sections `5.1` and `5.2` (or restart and run all cells if the runtime reset); re-running `5.2` updates the served HTML without starting a duplicate server on port `8765`.