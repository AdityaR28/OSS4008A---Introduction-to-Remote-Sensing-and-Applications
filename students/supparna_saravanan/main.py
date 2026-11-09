from clean_dataset import clean_pipeline_dataset
from pipeline_map import generate_pipeline_map


file_paths = {
    "Pipeline_Canada": "data/Pipeline_Canada.csv",
    "Pipeline_Canada2": "data/Pipeline_Canada2.geojson",
    "Pipeline_Segments_BC": "data/Pipeline_Segments_BC.csv",
    "Pipeline_AB": "data/Pipeline_AB.csv",
    "Pipelines_NB": "data/Pipeline_NB.csv"
}


# Clean dataset
pipelines_df = clean_pipeline_dataset(file_paths)

print(pipelines_df.shape)
print(pipelines_df.head())


# Generate interactive Mapbox map
html_file = generate_pipeline_map(
    pipelines_df,
    output_html="pipeline_map.html"
)

print(f"Map saved to: {html_file}")