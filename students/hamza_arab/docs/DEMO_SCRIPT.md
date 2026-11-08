# Notes for the November 9 meeting

## What I would show

1. Open the web map launched from `04_canada_web_demo.ipynb` in Colab.
2. Search for Kitchener or Ottawa and show how the map moves to that location.
3. Point out the hospital markers and change the study radius to show how the reference search changes.
4. Explain that the 678 national hospital records came from filtering Statistics Canada's ODHF v2.0 dataset. The number describes reference records, not successful AI detections.
5. Briefly describe the intended next stage: high-resolution imagery, reviewed annotations, model training/evaluation, and connecting real predictions to the map.

## A short way to explain the project

"My goal is to build a tool that can locate hospitals in overhead imagery and display its findings on a Canada-wide map. So far I've built the map and connected it to a Canadian hospital reference dataset. I can search for cities and show known hospital locations, which will help with collecting and checking training examples. The model isn't trained yet; my next step is to obtain suitable high-resolution imagery, create verified labels and test a first detector."

## Questions I should be prepared for

**Is it detecting hospitals automatically?** Not yet. Today's markers are known facilities from Statistics Canada.

**Is the whole country included?** The search and reference dataset cover Canada. Image-based detections have not been run across the country.

**What does the map assistant do?** Right now it handles simple place-navigation commands; it isn't a general chatbot.

**Why not train on Sentinel-2 directly?** At approximately 10 m per visible-band pixel, individual building detail is usually insufficient for this classification task. Higher-resolution overhead imagery is needed.

**What will I ask the TA?** Whether the Gradio interface fits the class-wide app, which imagery/licences are acceptable, whether LabelMe is mandatory, and what needs to be in the upcoming GitHub progress submission.
