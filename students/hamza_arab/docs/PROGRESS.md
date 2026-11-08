# Project progress — November 8, 2026

## Where I am

I have a working web-map prototype for my Hospital Infrastructure in Canada project. It runs from a Google Colab notebook and can search for Canadian cities and towns. The app filters Statistics Canada's ODHF v2.0 healthcare database to hospitals and displays those known locations on the map. The latest dataset run returned 678 hospital reference records nationwide.

I also tested Google Earth Engine in Colab and explored Halifax imagery and hospital/building reference data. These earlier notebooks helped establish the imagery and geographic-data workflow.

The web interface has been revised several times based on testing. It now has a combined location search, map panel, compact study-radius settings and a conversation panel for basic navigation commands. Previous iterations were launched and checked in a browser; the latest single-screen layout still needs a final visual check.

## What the map can and cannot do

The hospital markers are database records, not findings from satellite images. The radius selects which reference records appear around a searched location. Drawing polygons on the map is possible, but those drawings currently do not trigger image collection or detection. The assistant responds to a small set of navigation requests; it is not yet a general conversational AI.

The application has a GeoJSON import for viewing separately generated predictions. There are no valid model predictions to import yet because the detector has not been trained.

## Work prepared for the next phase

- Notebook 03 explores candidate buildings and hospital footprints, which need to be checked before becoming labels.
- Notebook 05 converts verified LabelMe annotations into a training format.
- Notebook 06 contains a training/evaluation workflow, but has not produced a trained model or accuracy scores.
- Folders are in place for training-data records and future evaluation outputs.

## Next technical milestone

Acquire properly licensed high-resolution overhead images of confirmed hospital buildings and comparable nonhospital sites. Verify annotations, create geographically separated training/validation/test sets, train a baseline detector, and measure its performance. Connect actual predictions to user-selected areas in the map after the model is tested.

## Questions to confirm with the TA

1. Does the shared class interface have to use a specific framework, or can this Gradio app be integrated?
2. Can I train on suitably licensed municipal/provincial aerial imagery alongside Google Earth Engine data?
3. Is LabelMe mandatory, and how should annotations and imagery licensing be submitted?
4. Is a chatbot required, or is the imagery detector and interactive map the assessed AI component?
5. Which GitHub files and progress deliverables are expected for this week's checkpoint?

## Meeting readiness

Ready to demonstrate: the Canada-wide reference map, location search, hospital markers and map settings; explain the data source and planned detector. Not ready to claim: image-based hospital detection, accuracy metrics, or end-to-end analysis of a selected polygon.
