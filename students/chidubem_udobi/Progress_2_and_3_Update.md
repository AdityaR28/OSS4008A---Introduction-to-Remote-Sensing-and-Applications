# OSS4008A — Progress 2 and 3 Update

**Project:** Where Are Homes in Canada? — AI-Based Residential Mapping and Wildland–Urban Interface Detection

**Study area:** Ottawa/Kanata, Ontario, Canada

**Date:** November 23, 2026

## 1. Progress since the first meeting

The project has progressed from an initial geographic selection and remote sensing prototype to a functioning web-based AI building-detection application.

During Phase II, I developed a Google Colab prototype using Canadian aerial imagery and a pretrained YOLO building-detection model.

During Phase III, I developed a Streamlit interface that allows users to select a geographic area, run AI detection and visualize candidate buildings on an interactive map.

The interface is connected to the AI inference backend, and two successful tests have been completed.

## 2. User interface status

**Status: Developed and operational for the Ottawa/Kanata pilot study.**

The application includes:

- Interactive aerial map with zoom and navigation
- Rectangle drawing tool for geographic selection
- Use drawn rectangle button to confirm the selected area
- Adjustable minimum detection confidence
- Detect Homes / Buildings button to run AI inference
- Geographic detection overlays showing candidate buildings
- Detection count and processing information
- Download detections CSV button

The application runs in Python using Google Colab and Streamlit. A temporary Cloudflare tunnel provides browser access to the running interface.

## 3. AI integration and data sources

The application uses a pretrained YOLOv8s building-segmentation model hosted on Hugging Face.

Model: https://huggingface.co/keremberke/yolov8s-building-segmentation

Canadian imagery is retrieved from Ontario's GEO Imagery Data Service 2023–2027.

Imagery service: https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/AerialImagery/GEO_Imagery_Data_Service_2023to2027/ImageServer

The model processes imagery tiles for the selected geographic area and returns candidate building predictions.

## 4. Testing results

| Test | Candidate detections | Image tiles | Processing time | Minimum confidence |
|---|---:|---:|---:|---:|
| Ottawa/Kanata Test 1 | 34 | 9 | 62.8 seconds | 0.45 |
| Ottawa/Kanata Test 2 | 60 | 12 | 15.1 seconds | 0.45 |

Both tests successfully displayed candidate building detections as red bounding boxes on the interactive map.

The application also generated downloadable CSV results containing geographic detection coordinates.

Screenshots and CSV evidence are stored in the `results` folder.

## 5. Challenges and solutions

**Google Colab runtime disconnections:** The launcher was revised to support reloading the application files from Google Drive.

**Streamlit display issues:** Colab's built-in proxy experienced WebSocket errors. A temporary Cloudflare tunnel was used to make the application accessible in a browser.

**Inference performance:** The application limits the number of imagery tiles processed per request to help control runtime.

## 6. Current limitations

The current AI model detects candidate buildings, not verified residential homes.

The model uses pretrained weights; fine-tuning on a manually annotated Canadian dataset has not yet been completed.

The pilot results have not been compared against manually annotated ground-truth data, so a local precision or recall score is not yet available.

Wildland–urban interface risk classification is not yet implemented.

## 7. Next steps

- Conduct a small manual review of predicted building locations.
- Create annotated Canadian imagery for model validation and fine-tuning.
- Evaluate detection performance using appropriate metrics.
- Integrate vegetation and land-cover information for WUI analysis.
- Complete the final project report, user documentation and presentation materials.

## Preliminary Visual Validation — November 23, 2026

A qualitative visual inspection was conducted using the second Ottawa/Kanata test, which produced 60 candidate building detections at a confidence threshold of 0.45.

The upper and lower portions of the selected residential area were examined to assess whether predicted bounding boxes corresponded to visible rooftops.

### Observations

- Many predicted bounding boxes overlapped recognizable building rooftops, demonstrating that the model can identify candidate structures in Canadian aerial imagery.
- Several visible rooftops were not detected, indicating potential false negatives.
- Some detections covered only portions of roofs rather than complete building footprints.
- Some adjacent or overlapping detections may represent duplicate predictions.
- A small number of predictions were visually ambiguous and require closer examination.

### Interpretation

The results demonstrate functional AI-assisted building identification, but the detections are not yet sufficiently validated for an overall accuracy claim.

This preliminary assessment is qualitative and does not establish precision, recall, F1-score or mean average precision.

### Recommended Improvements

1. Create manually annotated ground-truth building labels for a representative sample of Canadian aerial imagery.
2. Compare model predictions against those labels to calculate precision and recall.
3. Improve duplicate filtering and bounding-box localization.
4. Fine-tune the pretrained model using Canadian residential imagery.
5. Integrate vegetation and land-cover information for future wildland–urban interface analysis.
## 8. Conclusion

The Phase III milestone has been demonstrated through a functioning web interface with integrated AI inference, geographic selection, mapped building detections and downloadable results.

The Ottawa/Kanata pilot provides a working foundation for further model validation and WUI analysis.
