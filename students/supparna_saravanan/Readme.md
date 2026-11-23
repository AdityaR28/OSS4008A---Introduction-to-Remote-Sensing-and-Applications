# Canadian Pipeline Network — Interactive Map with Local AI Chat

A Google Colab notebook that combines several Canadian pipeline datasets into one clean table and displays them on an interactive Mapbox map. A chat sidebar lets you filter the map in plain English (for example *"show oil pipelines in Alberta"* or *"the 5 longest pipelines"*). The chat is powered by an open Qwen model running locally on the Colab GPU, so no external AI API is called.

**File:** `Canadian_Pipeline_Map_Colab.ipynb`

---

## What it does

1. **Cleans and merges** multiple pipeline datasets (CSV, TXT, Excel, GeoJSON, JSON) into a single standard schema and saves `Canadian_Pipelines_Clean.csv`.
2. **Clips** every pipeline to Canada's border and greys out everything outside Canada.
3. **Builds an interactive Mapbox page** with pipeline lines, click popups, a legend, zoom/fullscreen controls and a chat sidebar.
4. **Runs a local language model** (Qwen2.5-3B-Instruct on GPU, 1.5B on CPU) that turns chat requests into map filters.
5. **Serves everything from inside the notebook** and gives you a link to open the map in a new browser tab.

---

## Requirements

| Requirement | Details |
|---|---|
| Google account | For Colab and Google Drive |
| Colab runtime | **T4 GPU** (see below) |
| Mapbox access token | **Required.** Free at <https://account.mapbox.com>. Use a **public** token (starts with `pk.`) |
| Hugging Face token | **Optional.** Free at <https://huggingface.co/settings/tokens>. Gives authenticated, faster model downloads. The Qwen models are open and also download without it |
| Datasets | The pipeline files, stored in Google Drive |

---

## How to run

### 1. Set the runtime to T4 GPU

In Colab: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**.

Without a GPU the notebook still works, but it falls back to the smaller 1.5B model on CPU, and chat replies will be slow.

### 2. Add your tokens

Open the **Secrets** panel (key icon in the left sidebar) and add:

| Secret name | Value | Required |
|---|---|---|
| `MAPBOX_ACCESS_TOKEN` | your Mapbox public token | Yes |
| `HF_TOKEN` | your Hugging Face token | No |

Turn **Notebook access** ON for each secret.

The notebook reads the tokens from Colab Secrets first, then from existing environment variables, and exports them as the environment variables `MAPBOX_ACCESS_TOKEN` and `HF_TOKEN` (cell 1.5). The older secret name `MAPBOX_TOKEN` is also accepted.

If you would rather use environment variables directly, run this in a cell before cell 1.5:

```python
import os
os.environ["MAPBOX_ACCESS_TOKEN"] = "pk.your_token_here"
os.environ["HF_TOKEN"] = "hf_your_token_here"   # optional
```

Do not paste real tokens into a notebook you plan to share.

### 3. Put your datasets in Google Drive

By default the notebook expects this layout:

```
MyDrive/
└── Pipelines/
    ├── Pipeline_Canada.csv
    ├── Pipeline_Canada2.geojson
    ├── Pipeline_Segments_BC.csv
    ├── Pipeline_AB.csv
    └── Pipeline_NB.csv
```

Change `DATA_FOLDER` and the `file_paths` dictionary in cell **1.4** if your folder or file names differ. The dictionary keys are labels that end up in the `source_dataset` column.

### 4. Run all cells

**Runtime → Run all**, or run the sections in order. Colab will ask you to authorize Google Drive access the first time.

- The first run downloads the Qwen model (about 1–3 minutes).
- At the end of section **5.2**, click the link printed in the output to open the map in a new tab.
- Keep the Colab runtime connected while you use the map; the map and chat run from the notebook.

---

## Notebook structure

| Section | Contents |
|---|---|
| **1. Setup** | 1.1 install packages · 1.2 imports · 1.3 mount Drive · 1.4 data paths · 1.5 load tokens |
| **2. Data cleaning** | 2.1 schema and column aliases · 2.2 file readers · 2.3 geometry helpers · 2.4 record helpers · 2.5 main cleaning function |
| **3. Map builder** | 3.1 constants · 3.2 Canada boundary and clipping · 3.3 GeoJSON and outside-Canada mask · 3.4 HTML/JS template · 3.5 page generator |
| **4. Local AI model** | 4.1 load Qwen · 4.2 chat function |
| **5. Run** | 5.1 clean the datasets · 5.2 launch the map server |

---

## How it works

### Data cleaning (section 2)

For each dataset, `clean_pipeline_dataset()`:

1. Reads the file (CSV is tried as UTF-8, then cp1252, then latin1; malformed rows are skipped).
2. Normalizes column names to lowercase snake_case.
3. Maps source columns to the standard schema using the `ALIASES` table. The first matching alias wins, and fields with no match are left empty.
4. Parses geometry from GeoJSON or WKT columns and stores it as WKT. Missing or invalid geometry becomes empty rather than causing an error. It also stores a representative latitude and longitude for each record.
5. Saves all original attributes as JSON in `source_fields`, so no source information is lost.
6. Drops rows that have no identifying information at all.

Then it combines the datasets, converts numeric columns, removes duplicates (only where a source record ID exists, per dataset), trims whitespace, and writes `/content/data/Canadian_Pipelines_Clean.csv`.

**Standard columns:** `pipeline_name`, `company`, `parent_company`, `province`, `location_description`, `from_location`, `to_location`, `commodity`, `pipeline_length_km`, `capacity`, `status`, `start_date`, `end_date`, `segment_number`, `licence_number`, `project_number`, `authority_type`, `data_source`, `latitude`, `longitude`, `geometry`, `source_dataset`, `source_record_id`, `source_fields`.

### Map (section 3)

- The Canada boundary is downloaded from Natural Earth.
- Each pipeline is tested against the boundary (with a small ~2 km tolerance). Pipelines fully inside are kept, ones crossing the border are trimmed, and ones fully outside are removed.
- A world-sized polygon with Canada cut out is drawn in grey to mask everything outside Canada.
- All data is embedded into one self-contained HTML page that uses Mapbox GL JS. Clicking a pipeline shows its company, province, commodity, length, capacity, status and from/to locations.

### AI chat (sections 3.4, 4 and 5.2)

1. You type a request in the sidebar, for example *"gas pipelines in British Columbia longer than 300 km"*.
2. The page sends the recent conversation, the current filters and a system prompt to the notebook's `/chat` endpoint. The system prompt lists the provinces, commodities and statuses that actually exist in your data.
3. The local Qwen model replies with **JSON only**. The answer is pre-filled with `{` to force this format.
4. The page parses the JSON and applies the filters to the map (province, commodity, status, company, name, min/max length, or *top N longest*), then zooms to the matches and shows a short summary.
5. Saying *"show all"* or clicking **Show all** resets the map.

The model only controls map filters. It does not answer questions from its own knowledge, and if nothing matches, the map is left unchanged.

### Server

Section 5.2 starts a small Python HTTP server on port `8765`:

- `GET /` serves the map page
- `POST /chat` runs the model (one request at a time)

Colab exposes the port through `output.serve_kernel_port_as_window`. Re-running the cell just refreshes the page; the server keeps running.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for datasets | Update `DATA_FOLDER` / `file_paths` in cell 1.4 |
| Mapbox token error or blank map | Check the `MAPBOX_ACCESS_TOKEN` secret, confirm notebook access is enabled, and use a valid public `pk.` token |
| Chat is very slow | Make sure the runtime is **T4 GPU**, then restart and run all cells |
| "Couldn't turn that into a map filter" | Rephrase with a province, commodity, company or length, e.g. *"show oil pipelines in Alberta"* |
| Map link stopped working | The runtime disconnected; re-run section 5 |
| Model download is slow or rate-limited | Add an `HF_TOKEN` secret |

---

## Notes and limits

- The Mapbox token is embedded in the generated page, so use a public token and consider restricting it by URL in your Mapbox account.
- Pipeline length is used as given in the source data; units are not converted, and the column is assumed to be kilometres.
- Boundary data is simplified, so pipelines right at the border may be trimmed slightly differently than official boundaries would.
- A 3B model can occasionally misread an unusual request. The filter bar above the chat shows exactly which filters were applied.
