# ShapeScore
Automatically score *C. elegans* survival from still images based on posture using FasterRCNN and SAM 2.

![alt text](https://github.com/tomh014/shape-score/blob/main/summary%20figure%20small.png)

**Installation**
ShapeScore can optionally be used from a virtual environment with, for example, conda. 
Create a new environment with:
```
conda create -n shapescore python=3.11
```

Follow the instructions on https://pytorch.org/ to install the required version of pytorch. ShapeScore supports both GPU and CPU.
From the virtual environment, install the required packages with the requirements.txt file using:
```
python -m pip install -r requirements.txt
```

Download the following three models and add to the "model" folder in the main ShapeScore directory:

Worm detector weights

https://huggingface.co/hodde014/worm-detector/blob/main/fasterrcnn_depl.pth

SAM 2 tiny

https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

Survival classifier weights

https://huggingface.co/hodde014/worm-posture-survival-classifier/blob/main/combined_5-21.combined_model_5-20-25.deadalive.pth

**Usage**
Examples of usage can be found in the jupyter notebooks detection.ipynb and visualization.ipynb. ShapeScore requires this specific directory/project folder structure to function. run_detection.py can also be ran from the main directory to produce worm scores from images. 

Worm detector training data/annotations:
https://huggingface.co/hodde014/worm-detector/blob/main/data.zip
Survival classifier training data:
https://huggingface.co/hodde014/worm-posture-survival-classifier/blob/main/combined_5-21.zip

