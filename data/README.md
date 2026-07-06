# YOLO Dataset Format

YOLOv8 expects images and labels in a fixed directory structure with a `data.yaml` file that describes the dataset.

## Directory structure

```
data/
├── images/
│   ├── train/          # training images
│   └── val/            # validation images
└── labels/
    ├── train/          # one .txt label file per training image
    └── val/            # one .txt label file per validation image
```

For every image in `images/train/` or `images/val/`, there must be a label file with the same base name in `labels/train/` or `labels/val/`. A missing label file is treated as an empty annotation.

## Label file format

Each `.txt` file contains one line per object:

```
class_id x_center y_center width height
```

All five values are normalized to the range `[0, 1]` relative to the image width and height:

- `class_id`: integer index starting from 0.
- `x_center`: horizontal center of the bounding box.
- `y_center`: vertical center of the bounding box.
- `width`: bounding box width.
- `height`: bounding box height.

Example label file for one car and two people:

```
2 0.5 0.6 0.3 0.4
0 0.2 0.3 0.1 0.2
0 0.7 0.8 0.12 0.25
```

## data.yaml file

Create a `data.yaml` file in the dataset root:

```yaml
path: /content/your_dataset   # absolute path to the dataset folder
train: images/train
val: images/val
nc: 3
names:
  - person
  - bicycle
  - car
```

- `path`: absolute path to the dataset directory.
- `train` / `val`: relative paths from `path` to the image folders.
- `nc`: number of classes.
- `names`: list of class names in order of class id.

## Downloading a dataset from Roboflow

Roboflow Universe hosts thousands of YOLO-formatted datasets. Use the Roboflow Python package to download one:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
dataset = project.download("yolov8")
```

The downloaded folder will contain a ready-to-use `data.yaml` file.

## Default dataset

The default configuration uses `coco128.yaml`. This is a small 128-image subset of the COCO dataset with 80 classes. Ultralytics automatically downloads it on first use, so no manual setup is required.
