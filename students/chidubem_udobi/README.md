# Canadian dataset sources

| Source | Use | Caution |
|---|---|---|
| Sentinel-2 SR Harmonized (`COPERNICUS/S2_SR_HARMONIZED`) | Optical context and NDVI | 10 m pixels cannot reliably identify individual homes |
| Agriculture and Agri-Food Canada Annual Crop Inventory (`AAFC/ACI`) | Canadian forest and land-cover classes | 30 m classes; validate legend and year |
| NRCan Automatically Extracted Buildings | Candidate building polygons | Verify local coverage, vintage, errors, residential status |
| City of Ottawa aerial photography | Candidate AI training imagery | Verify download access, licence and acquisition year |

Sources:
- https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- https://developers.google.com/earth-engine/datasets/catalog/AAFC_ACI
- https://open.canada.ca/data/en/dataset/7a5cda52-c7df-427f-9ced-26f19a8a64d6
- https://ottawa.ca/en/planning-development-and-construction/developing-property/geomatic-services/geospatial-analytics-technology-and-solutions/aerial-photography-base-mapping-and-lidar

Checklist:
- [ ] Confirm imagery dates and scene count
- [ ] Confirm AAFC year and forest class codes
- [ ] Confirm building polygon coverage
- [ ] Verify residential-vs-nonresidential limitation
- [ ] Verify orthophoto access and permitted uses

Do not commit large satellite scenes or credentials.
