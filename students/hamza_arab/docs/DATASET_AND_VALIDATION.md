# Dataset and validation notes

## Reference sites

The current map uses Statistics Canada's Open Database of Healthcare Facilities (ODHF v2.0), filtered to facilities classified as hospitals. The Colab preparation cell returned 678 records in the latest run. These are useful locations to investigate, but they are not verified building-roof annotations and may include incomplete or imprecise geographic information.

Earlier Halifax experiments also used OpenStreetMap/Overpass data. That remains exploratory material; it is no longer the primary hospital-reference dataset for the Canada map.

## Imagery and labels

The actual detector will need high-resolution overhead imagery with usage rights suitable for model training and the course submission. Sentinel-2 can show regional context but is generally too coarse for reliable hospital-building detection.

Hospital structures need manually reviewed bounding boxes. The project has a LabelMe-to-YOLO conversion notebook, but no verified training-label dataset has been completed. Nearby buildings should not be labelled hospitals simply because they fall within a hospital property or facility polygon. Nonhospital examples such as schools, office buildings and large complexes will also be needed.

## Model validation

Build train, validation and test subsets by independent hospital site or geography, not by randomly separating adjacent crops of the same site. Once a model has been trained, record the source images, annotations, number of sites, precision, recall, mAP and examples of false detections or missed hospitals. Do not report accuracy until those tests have actually been run.

## Outputs

`03_Data` holds small reference formats and will later hold training-data manifests and annotation records. `04_Results` is reserved for actual test metrics, example prediction maps and figures. These folders are intentional, even though they currently contain little data.
