# IDM Project Report - Real-Time Object Detection with YOLOv8

## 1. Introduction

_Fill in: state the project objective, explain why YOLOv8 was chosen, and describe the real-time object detection problem it solves._

## 2. Related Work

_Fill in: give a brief history of the YOLO family from v1 to v8, and explain the difference between single-stage detectors like YOLO and two-stage detectors like R-CNN._

## 3. Methodology

### 3.1 Model

_Fill in: summarize the YOLOv8n architecture, including the backbone, neck, and detection head. Note why YOLOv8n is suitable for fast or resource-constrained inference._

### 3.2 Dataset

_Fill in: describe the dataset used (e.g., coco128 or a custom Roboflow dataset), the number of classes, the train/validation split, and any notable class distribution._

### 3.3 Training Setup

_Fill in: list the training parameters (epochs=15, batch=16, imgsz=640, lr0=0.001, freeze=10, optimizer=AdamW) and reference the values from the notebook's training cell._

### 3.4 Evaluation Metrics

_Fill in: define mAP50, mAP50-95, precision, and recall, and explain why each metric matters for object detection._

## 4. Results

### 4.1 Quantitative

_Fill in: paste the metrics table from the Colab evaluation cell output (mAP50, mAP50-95, precision, recall, speed)._

### 4.2 Qualitative

_Fill in: insert sample detection images from the Ultralytics training run (`runs/detect/train_fresh/`) and the annotated inference outputs from the notebook demo cell._

## 5. Analysis

_Fill in: discuss where the model fails. Common failure modes include small objects, occlusion, poor lighting, class confusion, and crowded scenes. Suggest improvements such as more data, larger model size, longer training, or better augmentation._

## 6. Conclusion

_Fill in: summarize the findings, restate whether the project met its real-time detection goal, and propose future work such as training on a larger dataset, using a larger model variant, or deploying to an edge device._

## 7. References

_Fill in: list references including the Ultralytics YOLOv8 documentation, the COCO dataset paper, and the original YOLO paper._
