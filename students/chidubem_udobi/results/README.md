# Phase III — AI Detection Test Results

Two Ottawa/Kanata residential areas were tested using the Residential AI Mapper.

| Test | Candidate building detections | Image tiles | Processing time | Confidence |
|---|---:|---:|---:|---:|
| Test 1 | 34 | 9 | 62.8 seconds | 0.45 |
| Test 2 | 60 | 12 | 15.1 seconds | 0.45 |

## Test procedure

1. Opened the Streamlit Residential AI Mapper.
2. Selected a residential area using the map's rectangle drawing tool.
3. Confirmed the selection using **Use drawn rectangle**.
4. Set minimum detection confidence to 0.45.
5. Clicked **Detect Homes / Buildings**.
6. Reviewed the geographic detection overlays and exported the results as CSV.

## Interpretation

Both tests successfully produced mapped candidate building detections and downloadable geographic results.

These counts represent model predictions, not verified residential properties. No ground-truth accuracy assessment has yet been completed.

## Data sources

- Ontario GEO Imagery Data Service 2023–2027
- Pretrained YOLOv8s building-segmentation model: `keremberke/yolov8s-building-segmentation`

## Limitations

The current system detects candidate buildings but does not verify residential use or classify wildland–urban interface risk.
