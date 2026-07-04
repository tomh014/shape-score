# ShapeScore
Automatically score *C. elegans* survival from still images based on posture using FasterRCNN and SAM 2.

![alt text](https://github.com/tomh014/shape-score/blob/main/summary%20figure%20small.png)

**Installation**  
ShapeScore can optionally be used from a virtual environment with, for example, conda. 
Create and activate a new environment with:
```
conda create -n shapescore python=3.11
conda activate shapescore
```
Install the required packages to your environemnt with:
```
python -m pip install -r requirements.txt  
```


Alternatively, create the environment in one line with
```
conda env create --shapescore --file=shapescore.yml
```


Download the following three models and add to the "model" folder in the main ShapeScore directory:

Worm detector weights:  
https://huggingface.co/hodde014/worm-detector/blob/main/fasterrcnn_depl.pth

SAM 2 tiny:  
https://drive.google.com/file/d/10o3Rnf5IYubdGVmNlZSv_GveIVyoehHV/view?usp=sharing

Survival classifier weights:  
https://huggingface.co/hodde014/worm-posture-survival-classifier/blob/main/combined_5-21.combined_model_5-20-25.deadalive.pth


**Usage**  
This repository can be cloned locally and code can be ran from the main directory. Users should follow detection.ipynb for a more detailed walkthrough of the software. visualization.ipynb and train_classifier.ipynb can be followed for visual inspection of automated detection/segmentation and training a custom classifier.

**Project setup**  
A typical project for assessment of survival will be set up as follows:
```
shapescore1.x/
├── project/
│   ├── [project]
│      ├── input_raw
│         ├── *.jpg
│      ├── plate_layouts
│         ├── [plate 1].csv
│         ├── ...
│         ├── [plate n].csv
│      ├── fields.csv
```

The "fields.csv" should describe the user's desired naming convention for well-level images in input_raw with identifiers seperated by underscores, or [field 1]\_[field2]\_...[field n].jpg. For example, one may use timepoint_plate_well_opticalconfig.jpg as used in the example projects. The only required fields are plate and well. fields.csv has two columns for the field index and the name of the field.

plate_layouts contains a .csv with two required columns (index and well) along with all fields except plate. Each .csv in the plate_layouts folder should be named according to a plate represented in the filenames of images in the input_raw folder. Indicies begin at well A01 and proceed down the column and begin again at the top of the next column (e.g. 0-7 is A01-H01, 8-15 is A02-H02). Provided example plate layouts can be used and modified. 

**Additional data**  
Training data:  
Worm detector training data/annotations:  
https://huggingface.co/hodde014/worm-detector/blob/main/data.zip  

Survival classifier training data:  
https://huggingface.co/hodde014/worm-posture-survival-classifier/blob/main/combined_5-21.zip  

