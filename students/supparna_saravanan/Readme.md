# Canadian Pipeline Network Mapping

A Python-based data processing and interactive mapping project for Canadian pipeline infrastructure. The project cleans and standardizes pipeline datasets, processes geographic information, and generates an interactive map to visualize pipeline networks across Canada.

The repository includes both standalone Python scripts and Google Colab notebooks, allowing the workflow to be run locally or in Google Colab.

## Project Overview

The project is organized into three main stages:

1. **Data Cleaning** – Reads, standardizes, and validates pipeline datasets, including column names, identifiers, and geographic information.
2. **Pipeline Mapping** – Processes the cleaned geographic data and generates an interactive pipeline network map.
3. **Complete Workflow** – Coordinates the cleaning and mapping stages through a single main script.

## Repository Structure

```text
.
├── colab/
│   ├── canadian_pipeline_data_cleaning.ipynb
│   ├── canadian_pipeline_network_map.ipynb
│   └── canadian_pipeline_complete_workflow.ipynb
│
├── data/
│   ├── [pipeline dataset files]
│   └── ...
│
├── clean_dataset.py
├── pipeline_map.py
├── main.py
└── README.md
```

**Note:** `Pipeline_AB.csv` is not included in the repository because of its large file size. The remaining datasets are stored in the `data/` directory.

## Main Components

### 1. `clean_dataset.py`

Handles the preparation and standardization of the source pipeline datasets.

Key functions include:

- Reading supported source data files.
- Standardizing column names using field aliases.
- Parsing and validating geographic information.
- Handling geometry and coordinate-related data.
- Preserving source identifiers and relevant attributes.
- Producing cleaned datasets for downstream mapping.

### 2. `pipeline_map.py`

Generates an interactive visualization of the Canadian pipeline network.

Key functions include:

- Loading cleaned pipeline data.
- Converting geographic data into GeoJSON.
- Retrieving and processing the Canada boundary.
- Handling geographic masking.
- Generating an interactive HTML map.
- Supporting Mapbox-based map visualization.

A valid Mapbox access token may be required, depending on the selected map configuration.

### 3. `main.py`

Acts as the main workflow orchestrator.

It connects the cleaning and mapping modules, allowing the complete process to be executed through one entry point.

The workflow is:

```text
Source Datasets
      |
      v
clean_dataset.py
      |
      v
Cleaned Pipeline Dataset
      |
      v
pipeline_map.py
      |
      v
Interactive Pipeline Map
```

## Google Colab Notebooks

The `colab/` directory contains three notebooks designed to make the workflow easier to run, understand, and modify in Google Colab.

| Notebook | Purpose |
|---|---|
| `canadian_pipeline_data_cleaning.ipynb` | Runs the dataset cleaning and standardization process. |
| `canadian_pipeline_network_map.ipynb` | Generates the interactive Canadian pipeline network map. |
| `canadian_pipeline_complete_workflow.ipynb` | Runs the complete cleaning and mapping workflow. |

Each notebook is organized into sections with explanations, code comments, configuration cells, and execution steps.

## Getting Started

### Option 1: Run Locally

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

**2. Install the required Python dependencies**

Install the packages required by the project and its scripts. Refer to the installation cells in the Colab notebooks for the dependency list.

**3. Prepare the datasets**

Place the required source datasets inside the `data/` directory.

If you intend to run the complete workflow, make sure all expected input files are available, including `Pipeline_AB.csv`.

**4. Configure the Mapbox token**

If your map configuration uses Mapbox, set your access token as an environment variable:

```python
import os

os.environ["MAPBOX_TOKEN"] = "YOUR_MAPBOX_ACCESS_TOKEN"
```

Do not commit your actual access token to GitHub.

**5. Run the workflow**

```bash
python main.py
```

The workflow will execute the data cleaning and mapping stages using the configured input and output paths.

### Option 2: Run in Google Colab

1. Open the repository's `colab/` directory.
2. Open `canadian_pipeline_complete_workflow.ipynb` in Google Colab.
3. Upload or mount the required project files and datasets.
4. Make sure the Python scripts are available in the project directory:
   - `clean_dataset.py`
   - `pipeline_map.py`
5. Ensure the required datasets are available in the `data/` directory.
6. Configure the Mapbox token if required.
7. Run the notebook cells in order.

The individual cleaning and mapping notebooks can also be used independently.

## Input Data

The project uses source datasets containing Canadian pipeline infrastructure information.

The input files are expected to be organized in the `data/` directory, following the file paths configured in the Python scripts.

**Large dataset notice:** `Pipeline_AB.csv` is excluded from this GitHub repository because of its file size.

To reproduce the complete workflow:

- Obtain `Pipeline_AB.csv` from the appropriate data source or project storage location.
- Place it in the expected location under `data/`.
- Confirm that the filename and path match the configuration in the scripts.

The workflow may not produce complete results if a required input dataset is missing.

## Outputs

Depending on the selected workflow and configuration, the project produces:

- Cleaned and standardized pipeline datasets.
- Processed geographic data.
- GeoJSON representations of pipeline features.
- An interactive HTML map of the Canadian pipeline network.

The generated files are saved according to the output paths configured in the scripts and notebooks.

## Configuration and Notes

- **File paths:** The scripts use configurable project and data paths. Check the configuration sections before running the workflow.
- **Coordinate systems:** Geographic data should use the appropriate coordinate reference system for the intended mapping operations.
- **Missing datasets:** Required input files must be supplied before running the full workflow.
- **Mapbox:** Keep access tokens private and use environment variables rather than hard-coding credentials.
- **Google Colab:** Upload the required Python scripts and datasets, or mount the appropriate Google Drive directory, before executing the notebooks.

## Reproducibility

For consistent results:

1. Use the same source datasets across runs.
2. Keep the expected folder structure intact.
3. Use the same configuration and coordinate reference system.
4. Ensure all required input files are available.
5. Configure any required external service credentials.

## License

Add the applicable project license here before distributing or reusing the code.

## Acknowledgements

This project uses Python-based geospatial data processing and interactive mapping tools to support the exploration and visualization of Canadian pipeline infrastructure.
