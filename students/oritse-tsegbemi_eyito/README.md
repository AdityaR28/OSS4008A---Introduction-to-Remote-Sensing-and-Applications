# Ships and Ports in Canada

**Student:** Oritse-Tsegbemi Eyito  
**Course:** OSS4008A - Introduction to Remote Sensing and Applications  
**Study area:** Halifax Harbour, Nova Scotia  
**Environment:** Python in Google Colab, Google Earth Engine, geemap

## Objective

Develop a tool for mapping ships and ports in Canada. The initial notebook is
an imagery-viewer baseline for one harbour: it selects a dated Sentinel-2 image,
displays raw and quality-screened RGB layers, reports coverage, and exports a PNG.
Ship detection, port boundaries, model training, and shared-dashboard integration
have not been implemented.

## Folder structure

```text
students/oritse-tsegbemi_eyito/
├── README.md
├── code/
│   └── 01_halifax_harbour_baseline.ipynb
├── data/
│   └── README.md
└── results/
    └── README.md
```

Generated CSV and GeoJSON records belong in `data/`. PNG previews and run metadata
belong in `results/`. Generated outputs have not been included in this source
package; they are created when the notebook runs.

## Run in Colab

1. Open `code/01_halifax_harbour_baseline.ipynb` in Google Colab.
2. Use a standard CPU runtime and run the cells in order.
3. Authenticate with an account authorized for the configured Earth Engine project.
   Change `PROJECT_ID` if another account/project is being used.
4. Select a candidate scene and click **Load image**. Compare the raw and
   quality-screened layers and inspect the coverage report.
5. Rerun section 6 after changing scenes to save matching metadata.
6. Run section 7 to generate the PNG. Set `DOWNLOAD_PNG = True` to request a download.
7. Optionally run the outputs ZIP cell after section 7, extract it, and upload
   generated files to the matching `data/` and `results/` folders.
8. Save the notebook to `students/oritse-tsegbemi_eyito/code/` on GitHub using
   **File > Save a copy in GitHub**, or download it and upload through GitHub.

The standalone output workspace is
`/content/ships_and_ports/students/oritse-tsegbemi_eyito/`.
If working in a cloned repository, set `PROJECT_DIR` to
`/content/OSS4008A---Introduction-to-Remote-Sensing-and-Applications/students/oritse-tsegbemi_eyito/`.
Commit only files inside the assigned student folder. Update the weekly log below
as work is completed.

## Data

Sentinel-2 surface reflectance, collection `COPERNICUS/S2_SR_HARMONIZED`, accessed
through Earth Engine. The default search covers June 1 to September 1, 2025
(end exclusive) around Halifax Harbour. RGB bands have 10 m pixels; the SCL
quality layer has 20 m pixels. Data terms and access details are in `data/README.md`.

## Results and limitations

The initial Colab run found 16 candidate granules. The July 26, 2025 scene had
100% RGB coverage and 99.49% retained coverage after quality screening.
A later June 21, 2025 scene produced a valid 1600 x 1600 PNG preview.
Coverage is a data-quality measure, not detection accuracy.

Small vessels may be unresolved at 10 m. Scene cloud percentage describes the
whole scene and does not guarantee a clear harbour. PNG previews use the map's
colour settings and do not preserve geographic coordinates or reflectance values.

Known issue: OpenStreetMap background tiles returned HTTP 403 while satellite
imagery loaded. Browser delivery of downloads has not been confirmed. The source
notebook has no saved outputs. The previous imagery/PNG logic was tested in Colab;
the repository-layout output paths in this package passed local syntax and
consistency checks and need a fresh Colab run.

## Milestones

| Review | Course milestone | Current status |
| --- | --- | --- |
| November 9 | UI connection | Imagery viewer works; shared UI connection pending |
| November 23 | AI and testing | Training data, labels, model and validation pending |
| December 7 | Final delivery | Integration, case studies and documentation pending |

## Weekly progress log

### Week of November 2, 2026 - Initial baseline

- **Completed:** Colab imagery viewer for Halifax Harbour with scene selection,
  RGB scaling, SCL quality screening, and coverage estimates.
- **Verified:** Earth Engine authentication, candidate-image retrieval, map
  display, metadata generation, ZIP integrity, and PNG generation in Colab.
- **Organization:** Prepared the course `code/`, `data/`, and `results/` layout
  with this progress log in the student README. Repository upload is pending.
- **Problems:** OpenStreetMap background tile errors; browser downloads unconfirmed;
  imagery resolution needs evaluation for vessel detection.
- **Next:** Run the reorganized notebook, add generated outputs, investigate
  suitable Canadian training imagery, and confirm port-mapping/shared UI scope.

## References

- [Course repository instructions](https://github.com/AdityaR28/OSS4008A---Introduction-to-Remote-Sensing-and-Applications/blob/main/README.md)
- [Sentinel-2 dataset, quality classes and data terms](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Earth Engine authentication](https://developers.google.com/earth-engine/guides/auth)
- [geemap Colab workflow](https://geemap.org/notebooks/00_geemap_colab/)
- Course Labs 1 and 2: Python and Earth Engine workflow reference.
