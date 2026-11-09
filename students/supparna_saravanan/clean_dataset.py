import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# STANDARD DATASET STRUCTURE
# --------------------------------------------------

STANDARD_COLUMNS = [
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
    "latitude",
    "longitude",
    "geometry",
    "source_dataset",
    "source_record_id",
    "source_fields",
]


# --------------------------------------------------
# SOURCE COLUMN ALIASES
# --------------------------------------------------

ALIASES = {
    "pipeline_name": [
        "pipeline name", "name", "name_en", "name_fr"
    ],
    "company": [
        "cer-regulated company", "company_name", "proponent", "company"
    ],
    "parent_company": [
        "parent company"
    ],
    "province": [
        "province", "province_name"
    ],
    "location_description": [
        "location (description)", "location", "location_description"
    ],
    "from_location": [
        "from_location", "from location"
    ],
    "to_location": [
        "to_location", "to location"
    ],
    "commodity": [
        "transported commodity", "commodity", "substance 1", "line_type_desc"
    ],
    "pipeline_length_km": [
        "cer-regulated pipeline length (km)", "pipeline_length_km"
    ],
    "capacity": [
        "pipeline capacity", "capacity"
    ],
    "status": [
        "status", "segment_status"
    ],
    "start_date": [
        "commenced operations", "start_date", "approval_date", "approval date"
    ],
    "end_date": [
        "end_date", "retirement_date"
    ],
    "segment_number": [
        "segment_number"
    ],
    "licence_number": [
        "licence_number", "license_number"
    ],
    "project_number": [
        "project_number"
    ],
    "authority_type": [
        "authority_type"
    ],
    "data_source": [
        "data_source"
    ],
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalize_column(name):
    """Convert column names into consistent lowercase snake_case."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def read_csv_safely(path):
    """Read comma-separated CSV files using pandas' fast C parser."""
    last_error = None

    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                path,
                sep=",",
                encoding=encoding,
                engine="c",
                low_memory=False,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"Could not read {path}. Last encoding error: {last_error}"
    )


def read_file(path):
    """Read CSV, TXT, Excel, GeoJSON, and JSON files."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".geojson", ".json"):
        return gpd.read_file(path)

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    if suffix in (".csv", ".txt"):
        return read_csv_safely(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def parse_wkt_geometry(value):
    """Parse a WKT string, returning None when it is missing or invalid."""
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        return wkt.loads(value)
    except Exception:
        return None


def is_valid_geometry(geom):
    """
    True only for a non-missing, non-empty Shapely geometry.

    This explicitly rejects NaN values (which pandas represents as floats),
    strings, and other non-geometry objects.
    """
    return (
        geom is not None
        and hasattr(geom, "geom_type")
        and hasattr(geom, "is_empty")
        and not geom.is_empty
    )


def normalize_geometry_value(value):
    """Return a Shapely geometry or None, regardless of input cell type."""
    if is_valid_geometry(value):
        return value

    if isinstance(value, str):
        parsed = parse_wkt_geometry(value)
        return parsed if is_valid_geometry(parsed) else None

    return None


def get_geometries(df):
    """Find GeoJSON geometry or a WKT geometry column and normalize values."""
    geometry_values = None

    if isinstance(df, gpd.GeoDataFrame):
        geometry_values = df.geometry

    else:
        for column in ("the_geom", "geom", "wkt", "geometry"):
            if column in df.columns:
                geometry_values = df[column]
                break

    if geometry_values is None:
        return pd.Series(None, index=df.index, dtype=object)

    # Always normalize each cell. This safely handles NaN floats, None,
    # WKT strings, and already-parsed Shapely geometries.
    return geometry_values.apply(normalize_geometry_value)


def get_source_record_id(df):
    """Find the best available source record identifier."""
    id_fields = [
        "objectid",
        "pinfo_id",
        "licence_number",
        "project_number",
        "segment_number",
    ]

    result = pd.Series(None, index=df.index, dtype=object)

    for column in id_fields:
        if column in df.columns:
            result = result.fillna(df[column])

    return result


def serialize_source_fields(df):
    """
    Preserve original non-geometry attributes as JSON.
    Geometry is excluded to avoid duplicating large WKT values.
    """
    geometry_columns = {"geometry", "the_geom", "geom", "wkt"}
    source_columns = [
        column for column in df.columns
        if column not in geometry_columns
    ]

    records = df[source_columns].to_dict(orient="records")

    return [
        json.dumps(record, ensure_ascii=False, default=str)
        for record in records
    ]


# --------------------------------------------------
# MAIN CLEANING FUNCTION
# --------------------------------------------------

def clean_pipeline_dataset(
    file_paths,
    output_csv="Canadian_Pipelines_Clean.csv",
):
    """
    Combine pipeline datasets into one standardized CSV.

    Parameters
    ----------
    file_paths : dict
        Dictionary mapping dataset names to file paths.
    output_csv : str or pathlib.Path
        Output filename. Saved inside this script's data folder unless
        an absolute path is supplied.

    Returns
    -------
    pandas.DataFrame
        Combined cleaned pipeline dataset.
    """

    output_path = Path(output_csv)

    # Save relative output paths in the project's data folder.
    if not output_path.is_absolute():
        output_path = DATA_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []

    for dataset_name, file_path in file_paths.items():
        print(f"\nReading {dataset_name}...")

        df = read_file(file_path)

        # Normalize source column names.
        df.columns = [normalize_column(column) for column in df.columns]

        print(f"Loaded {len(df):,} records.")

        cleaned = pd.DataFrame(index=df.index)

        # Map source columns to standardized columns.
        for standard_field, aliases in ALIASES.items():
            matching_column = next(
                (
                    normalize_column(alias)
                    for alias in aliases
                    if normalize_column(alias) in df.columns
                ),
                None,
            )

            if matching_column is not None:
                cleaned[standard_field] = df[matching_column]
            else:
                cleaned[standard_field] = None

        cleaned["source_dataset"] = dataset_name
        cleaned["source_record_id"] = get_source_record_id(df)

        # Parse and normalize geometries once.
        geometries = get_geometries(df)

        # Save geometry as WKT for CSV compatibility.
        cleaned["geometry"] = geometries.apply(
            lambda geom: geom.wkt if is_valid_geometry(geom) else None
        )

        # Extract representative-point coordinates safely.
        def representative_coordinate(geom, coordinate):
            if not is_valid_geometry(geom):
                return None

            point = geom.representative_point()
            return point.y if coordinate == "latitude" else point.x

        cleaned["latitude"] = geometries.apply(
            lambda geom: representative_coordinate(geom, "latitude")
        )
        cleaned["longitude"] = geometries.apply(
            lambda geom: representative_coordinate(geom, "longitude")
        )

        # Preserve original non-geometry attributes.
        cleaned["source_fields"] = serialize_source_fields(df)

        # Ensure all output columns exist and are in the same order.
        for column in STANDARD_COLUMNS:
            if column not in cleaned.columns:
                cleaned[column] = None

        cleaned = cleaned[STANDARD_COLUMNS]

        # Remove rows with no useful identifying information.
        useful_columns = [
            "pipeline_name",
            "company",
            "geometry",
            "location_description",
            "from_location",
            "licence_number",
            "project_number",
        ]

        cleaned = cleaned.dropna(how="all", subset=useful_columns)

        all_records.append(cleaned)

        print(
            f"Processed {len(cleaned):,} records from {dataset_name}."
        )

    if not all_records:
        raise ValueError("No datasets were supplied in file_paths.")

    # Combine datasets.
    print("\nCombining datasets...")
    combined = pd.concat(all_records, ignore_index=True)

    # Convert numeric fields without assuming unknown source length units.
    for column in ("pipeline_length_km", "latitude", "longitude"):
        combined[column] = pd.to_numeric(
            combined[column], errors="coerce"
        )

    # Remove duplicate records only when a source ID is available.
    has_id = combined["source_record_id"].notna()

    identified = combined.loc[has_id].drop_duplicates(
        subset=["source_dataset", "source_record_id"],
        keep="first",
    )
    unidentified = combined.loc[~has_id]

    combined = pd.concat(
        [identified, unidentified],
        ignore_index=True,
    )

    # Trim whitespace in text columns.
    for column in combined.select_dtypes(include="object").columns:
        combined[column] = combined[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Export CSV.
    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n----------------------------------")
    print("PIPELINE DATA CLEANING COMPLETE")
    print("----------------------------------")
    print(f"Total records: {len(combined):,}")
    print(f"Output file: {output_path.resolve()}")
    print("----------------------------------")

    return combined
