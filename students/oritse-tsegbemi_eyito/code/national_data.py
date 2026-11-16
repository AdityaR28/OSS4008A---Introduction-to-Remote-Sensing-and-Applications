"""National reference acquisition in Google Colab; never ship ground truth."""
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ODI_PAGE = 'https://www150.statcan.gc.ca/n1/pub/34-26-0003/342600032023001-eng.htm'
ODI_URL = 'https://www150.statcan.gc.ca/pub/34-26-0003/2023001/zip/ODI_v2_ports_marinas.zip'
LICENCE = 'Open Government Licence - Canada'


def read_map_references(project):
    """Combine the ODI snapshot with separately attributed map references."""
    import csv
    path = Path(project) / 'data' / 'canada_ports_reference.csv'
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        references = list(csv.DictReader(handle))
    supplement = path.with_name('canada_facility_supplements.csv')
    if supplement.exists():
        with supplement.open(encoding='utf-8', newline='') as handle:
            references.extend(csv.DictReader(handle))
    return references


def acquire_ports(project):
    """Retain provider metadata and geometries; derive map points in EPSG:4326."""
    import geopandas as gpd
    import pandas as pd
    import requests
    project = Path(project)
    raw = project/'data/raw/odi_ports_v2'
    raw.mkdir(parents=True, exist_ok=True)
    response = requests.get(ODI_URL, timeout=120)
    response.raise_for_status()
    payload = response.content
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # Extract data/metadata only, with no archive-provided executable code.
        for member in archive.infolist():
            target = (raw/member.filename).resolve()
            if not target.is_relative_to(raw.resolve()):
                raise ValueError('Unsafe source archive path.')
            if not member.is_dir() and target.suffix.lower() in {'.gpkg', '.csv', '.pdf'}:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    packages = list(raw.rglob('*.gpkg'))
    if len(packages) != 1:
        raise ValueError('Expected one ports/marinas GeoPackage; inspect the source layout.')
    frame = gpd.read_file(packages[0])
    if frame.crs is None:
        raise ValueError('Source reference system is missing.')
    original_columns = list(frame.columns)
    def column(*names):
        lookup = {c.lower(): c for c in frame.columns}
        for name in names:
            if name in lookup:
                return lookup[name]
        raise ValueError('Review ODI schema; missing one of '+repr(names))
    # Source field aliases are explicit. Schema changes fail visibly.
    name = column('name', 'name_en', 'infrastructure_name')
    province = column('province_territory', 'prov_terr', 'province', 'prname', 'pruid')
    subtype = column('sub_type', 'infrastructure_subtype', 'infra_type', 'infra_subtype', 'type', 'subtype')
    source_id = column('unique_id', 'id', 'uid')
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    # Representative points are display anchors, not surveyed port locations.
    points = frame.to_crs(3347).geometry.representative_point().to_crs(4326)
    normal = pd.DataFrame({
        'reference_id': frame[source_id].astype(str), 'name': frame[name].fillna('Unnamed reference').astype(str),
        'province': frame[province].fillna('Unknown').astype(str),
        'subtype': frame[subtype].fillna('Unknown').astype(str),
        'longitude': points.x, 'latitude': points.y,
        'source_url': ODI_PAGE, 'record_type': 'published infrastructure reference',
        'validation_status': 'provider record; not ship/port detection ground truth'})
    for field in ['provider','source_id','source_class','ownership','jurisdiction','un_locode']:
        if field in frame.columns:
            normal[field]=frame[field].fillna('Unknown').astype(str)
    if normal.reference_id.duplicated().any():
        raise ValueError('Duplicate ODI identifiers require review.')
    normal.to_csv(project/'data/canada_ports_reference.csv', index=False)
    features = [{'type':'Feature', 'geometry':{'type':'Point','coordinates':[r['longitude'],r['latitude']]},
                 'properties':{k:v for k,v in r.items() if k not in {'longitude','latitude'}}}
                for r in normal.to_dict('records')]
    (project/'data/canada_ports_reference.geojson').write_text(
        json.dumps({'type':'FeatureCollection','features':features}, indent=2), encoding='utf-8')
    report = {'source_url': ODI_PAGE, 'download_url': ODI_URL, 'licence': LICENCE,
              'attribution':'Contains information licensed under the Open Government Licence - Canada.',
              'retrieved_utc':datetime.now(timezone.utc).isoformat(),
              'archive_sha256':hashlib.sha256(payload).hexdigest(), 'source_version':'ODI v2',
              'source_collection_period':'October 2023 to June 2024', 'exact_record_acquisition_dates':None,
              'records':len(normal), 'province_counts':normal.province.value_counts().to_dict(),
              'provinces_territories_without_records':sorted(set(['AB','BC','MB','NB','NL','NS','NT','NU','ON','PE','QC','SK','YT'])-set(normal.province)),
              'subtype_counts':normal.subtype.value_counts().to_dict(), 'source_columns':original_columns,
              'metadata_files':[str(p.relative_to(project)) for p in raw.rglob('*') if p.suffix in {'.csv','.pdf'}],
              'limitations':['Selected infrastructure; geographic completeness varies by source and licence.',
                             'Ports and marinas include different facility types; filter explicitly.',
                             'Derived representative map points are not port boundaries or ship labels.',
                             'No assumption of current operating status or exhaustive Canadian coverage.']}
    (project/'data/canada_ports_source.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return normal, report


def filter_ports(records, province='All', subtype='All', query=''):
    """Pure reference filter shared by Colab and a future host UI."""
    query = query.strip().casefold()
    return [r for r in records if (province == 'All' or r['province'] == province)
            and (subtype == 'All' or r['subtype'] == subtype)
            and (not query or query in str(r['name']).casefold())]
