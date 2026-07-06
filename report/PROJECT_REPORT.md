# IDM Project Report — Real-Time Object Detection using YOLOv8

**Course:** CSET225 | Intelligent Design Models | Summer Term  
**Student:** Sanskar Singh  

---

## 1. Introduction

This project implements a real-time object detection system using YOLOv8 (You Only Look Once, version 8). The model identifies what objects are present in an image and where they are located, drawing a bounding box around each one. YOLOv8 is a single-stage detector — it predicts class labels and bounding box coordinates in one forward pass through the network, making it fast enough for real-time applications.

The project loads a pretrained YOLOv8 nano model, fine-tunes it on the COCO128 dataset, evaluates it using mAP metrics, and runs inference on sample images to demonstrate detection results.

---

## 2. Related Work

Object detection has two main families:

- **Two-stage detectors** (R-CNN family): First propose regions that might contain objects, then classify each region. More accurate but slower.
- **Single-stage detectors** (YOLO family): Predict bounding boxes and class labels simultaneously in one pass. Faster, suitable for real-time use.

YOLO evolved through versions v1 (2016) through v8 (2023). Each version improved speed and accuracy. YOLOv8 replaced anchor boxes with anchor-free detection, uses a C2f module (cross-stage partial connections), and supports task-aligned assignment for training.

---

## 3. Methodology

### 3.1 Model — YOLOv8n (Nano)

YOLOv8n is the smallest variant of YOLOv8:

| Property | Value |
|---|---|
| Parameters | 3.2 million |
| File size | 6.2 MB |
| GFLOPs | 8.9 |
| Layers | 130 (73 after fusion) |
| Architecture | Backbone (Conv + C2f) → Neck (FPN + PAN) → Head (Detect) |

The nano variant is chosen for fast inference. Larger variants (s/m/l/x) offer higher accuracy but are slower.

### 3.2 Dataset — COCO128

| Property | Value |
|---|---|
| Images | 128 |
| Classes | 80 (COCO categories) |
| Train/Val split | Same 128 images used for both |
| Format | YOLO format: one `.txt` label file per image |
| Auto-download | Ultralytics downloads `coco128.zip` automatically |

COCO128 is a small subset of the COCO dataset, used for testing and demonstration. The 80 classes include person, car, bus, dog, cat, chair, etc.

### 3.3 Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| `epochs` | 15 | Enough for 128 images; more risks overfitting |
| `imgsz` | 640 | Standard YOLOv8 input resolution |
| `batch` | 16 | 16 images per gradient update (8 batches per epoch) |
| `device` | 0 | GPU (Tesla T4) for fast training |
| `lr0` | 0.001 | Low learning rate preserves pretrained features |
| `freeze` | 10 | Freeze backbone layers 0-9, train head only |
| `patience` | 20 | Early stopping if no improvement |

**Why lr0=0.001?** The default 0.01 is designed for large datasets (118K images). With only 128 images, a high learning rate destroys pretrained features — the model forgets what it already knew. 0.001 takes careful steps.

**Why freeze=10?** The backbone (first 10 layers) detects low-level features (edges, textures, shapes) that are already well-learned from pretraining. Training them on 128 images would overfit. Freezing them and training only the detection head is standard transfer learning for small datasets.

### 3.4 Evaluation Metrics

| Metric | Definition |
|---|---|
| **mAP50** | Mean Average Precision at IoU threshold 0.50. A detection is correct if its box overlaps the ground truth by ≥50%. Averaged across all 80 classes. |
| **mAP50-95** | Same as mAP50 but averaged across IoU thresholds from 0.50 to 0.95 in steps of 0.05. Stricter — the standard COCO benchmark metric. |
| **Precision** | Of all detections the model made, what fraction were correct. High precision = fewer false alarms. |
| **Recall** | Of all real objects in the images, what fraction did the model find. High recall = fewer missed objects. |

**IoU (Intersection over Union):** The overlap area between predicted and ground-truth bounding boxes, divided by their union area. IoU=1.0 means perfect overlap.

---

## 4. Results

### 4.1 Quantitative Metrics

| Metric | Value |
|---|---|
| mAP50 | **0.7209** |
| mAP50-95 | **0.5409** |
| Precision | **0.7277** |
| Recall | **0.6463** |

**Hardware:** NVIDIA Tesla T4 GPU (Google Colab)  
**Training time:** ~60 seconds (15 epochs with frozen backbone)  
**Inference speed:** 2.1ms per image (post-training validation)

### 4.2 Per-Class Performance (Selected)

| Class | Images | Instances | mAP50 |
|---|---|---|---|
| giraffe | 4 | 9 | 0.995 |
| suitcase | 2 | 4 | 0.895 |
| frisbee | 5 | 5 | 0.795 |
| tie | 6 | 7 | 0.735 |
| umbrella | 4 | 18 | 0.746 |
| person | 61 | 254 | 0.191 |

Most classes with sufficient instances show strong detection. Classes with very few instances (1-2 images) show lower scores due to insufficient training data.

### 4.3 Qualitative Results

The fine-tuned model successfully detects objects in sample images:

- **bus.jpg**: Detects persons, bus, stop sign with bounding boxes
- **zidane.jpg**: Detects persons in the image

Annotated images with bounding boxes are displayed inline in the notebook.

---

## 5. Analysis

### What Worked Well
- The freeze=10 + lr0=0.001 configuration preserved pretrained features while adapting the detection head
- mAP50 of 0.72 on 128 images is strong for the dataset size
- Training completed in under 60 seconds on a T4 GPU

### Failure Modes
- Classes with only 1-2 training instances (e.g., banana, sandwich) show 0 mAP — insufficient data
- Small objects (e.g., distant traffic lights) are occasionally missed
- The model may confuse visually similar classes (e.g., suitcase vs. handbag)

### Improvement Strategies
1. **More data**: Train on the full COCO dataset (118K images) or a domain-specific dataset
2. **Larger model**: Use yolov8s or yolov8m for higher capacity
3. **More epochs**: 50-100 epochs with the full dataset
4. **Custom dataset**: Fine-tune on a specific domain (medical, traffic, retail) for real-world utility
5. **Data augmentation**: Stronger augmentation (mosaic, mixup) to artificially expand the 128 images

---

## 6. Conclusion

The project successfully demonstrates real-time object detection using YOLOv8. A pretrained nano model was fine-tuned on COCO128 with transfer learning best practices (low learning rate, frozen backbone), achieving mAP50=0.72. The pipeline runs end-to-end on Google Colab with a T4 GPU in under 2 minutes.

Future work: train on a larger or domain-specific dataset, deploy to an edge device (Jetson Nano, Raspberry Pi), or export to ONNX/TensorRT for optimized inference.

---

## 7. Code Walkthrough (Line-by-Line)

### Step 1: Install and Check GPU

```python
# Install the Ultralytics package — it contains the YOLOv8 model code
!pip install -q ultralytics

# Import PyTorch — the deep learning framework YOLOv8 runs on
import torch

# Check if a GPU is available for faster training
print('CUDA available:', torch.cuda.is_available())

# Print the GPU name so we know what hardware we are using
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

### Step 2: Load Pretrained Model and Run Baseline

```python
# Path is a helper class for joining file paths safely
from pathlib import Path

# YOLO is the main class from Ultralytics — used to load, train, and run the model
from ultralytics import YOLO

# Image is from PIL (Python Imaging Library) — used to open and display images
from PIL import Image

# Load the pretrained YOLOv8 nano model (downloads automatically if not present)
# 'yolov8n.pt' = the nano weights file (3.2 million parameters, 6.2 MB)
model = YOLO('yolov8n.pt')

# Run object detection on a sample bus image from the internet
# conf=0.25 means: only show detections with 25% or higher confidence
# save=True means: save the annotated image (with bounding boxes) to disk
results = model.predict('https://ultralytics.com/images/bus.jpg', conf=0.25, save=True)

# results[0].save_dir is the folder where the annotated image was saved
# Path(...) wraps it so we can use / to join the folder path with the filename
# display() shows the annotated image with bounding boxes inside the notebook
display(Image.open(Path(results[0].save_dir) / 'bus.jpg'))
```

### Step 3: Fine-Tune on COCO128

```python
# Reload a FRESH pretrained model before training
# This is important: the model from Step 2 was already used for prediction
# We need a clean copy of the pretrained weights to start training from
model = YOLO('yolov8n.pt')

# Start fine-tuning (training) the model on the COCO128 dataset
results = model.train(
    data='coco128.yaml',   # Dataset config file — tells the model where images and labels are
    epochs=15,             # Number of training passes through all 128 images
    imgsz=640,             # Resize all images to 640x640 pixels before feeding to the model
    batch=16,              # Process 16 images at a time (one batch) before updating weights
    device=0,              # Use GPU number 0 (the Tesla T4) for training — much faster than CPU
    lr0=0.001,             # Learning rate = 0.001 — small steps so we don't destroy pretrained features
    freeze=10,             # Freeze first 10 layers (backbone) — only train the detection head
    patience=20,           # Stop early if no improvement for 20 epochs (saves time)
    name='train_fresh'     # Name of the folder where training results will be saved
)
```

### Step 4: Evaluate

```python
# glob searches for files matching a pattern — ** means search all subfolders
import glob

# Find the best.pt file (the model weights from the best epoch during training)
# recursive=True means search inside every subfolder
matches = glob.glob('**/best.pt', recursive=True)

# Use the first match found, or fall back to a default path
weights_path = matches[0] if matches else 'runs/detect/train_fresh/weights/best.pt'

# Print which weights file we are using (so we know the model loaded correctly)
print(f"Loading weights: {weights_path}")

# Load the fine-tuned model using the best weights from training
best = YOLO(weights_path)

# Run validation: the model predicts on val images and we compare to ground truth labels
# This computes mAP, precision, and recall across all 80 object classes
metrics = best.val(data='coco128.yaml')

# Print the four key metrics, formatted to 4 decimal places
print(f"mAP50:      {metrics.box.map50:.4f}")    # Mean Average Precision at IoU 0.50
print(f"mAP50-95:   {metrics.box.map:.4f}")      # Mean Average Precision at IoU 0.50-0.95
print(f"Precision:  {metrics.box.mp:.4f}")       # Overall precision (correct / all detections)
print(f"Recall:     {metrics.box.mr:.4f}")       # Overall recall (found / all real objects)
```

### Step 5: Demo Inference on Sample Images

```python
# cv2 is OpenCV — used for image processing (color conversion here)
import cv2

# os is used for file operations (checking modification time to find newest file)
import os

# glob searches for files — we use it to find the trained model weights
import glob

# YOLO is the model class, Image is for displaying images
from ultralytics import YOLO
from PIL import Image

# Find all best.pt files and sort by modification time (newest first)
# This ensures we use the weights from the most recent training run
matches = sorted(glob.glob('**/best.pt', recursive=True), key=os.path.getmtime, reverse=True)

# Use the newest best.pt file
weights_path = matches[0] if matches else 'runs/detect/train_fresh/weights/best.pt'

# Load the fine-tuned model
model_ft = YOLO(weights_path)

# Print which weights file we are using
print(f"Using fine-tuned weights: {weights_path}")

# Loop over two sample images — bus.jpg (street scene) and zidane.jpg (people)
for url in ['https://ultralytics.com/images/bus.jpg',
            'https://ultralytics.com/images/zidane.jpg']:

    # Extract just the filename from the URL (e.g., 'bus.jpg')
    name = url.split('/')[-1]

    # Run detection on this image — [0] gets the first (only) result
    # conf=0.25 means only show detections with 25%+ confidence
    res = model_ft.predict(url, conf=0.25)[0]

    # res.plot() returns the image with bounding boxes drawn (in BGR color format)
    # cv2.cvtColor converts BGR to RGB — needed because OpenCV uses BGR but display expects RGB
    annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)

    # Display the annotated image with bounding boxes in the notebook
    display(Image.fromarray(annotated))

    # Extract the class names of all detected objects
    # res.boxes contains all detected bounding boxes
    # b.cls is the class ID (integer), res.names[int(b.cls)] converts it to a name string
    names = [res.names[int(b.cls)] for b in res.boxes]

    # Print how many objects were detected and their class names
    print(f"  {name}: {len(res.boxes)} detections — {names}")
```

### Step 6: Save Trained Weights

```python
# shutil is used for copying files
import shutil

# os and glob are used for finding files
import os, glob

# Path is for handling file paths
from pathlib import Path

# Find all best.pt files, but exclude /content/best.pt itself (avoid self-copy)
matches = sorted(
    [m for m in glob.glob('**/best.pt', recursive=True) if os.path.abspath(m) != '/content/best.pt'],
    key=os.path.getmtime,
    reverse=True  # newest first
)

# Use the newest match, or fall back to default path
weights_path = matches[0] if matches else 'runs/detect/train_fresh/weights/best.pt'

# Destination path — /content/ is the main folder in Colab, easy to find in the file browser
dest = '/content/best.pt'

# Copy the weights file to /content/ (only if source and destination are different)
if os.path.abspath(weights_path) != os.path.abspath(dest):
    shutil.copy2(weights_path, dest)  # copy2 preserves file metadata (timestamps etc.)

# Calculate file size in MB (bytes / 1024 / 1024)
size_mb = Path(dest).stat().st_size / 1024 / 1024

# Print the save location and file size so the user knows where to find it
print(f"Weights saved to: {dest}")
print(f"File size: {size_mb:.1f} MB")
print(f"\nDownload via Colab file browser:")
print(f"  Left panel > Files > best.pt > right-click > Download")
```

---

## 8. Key Terms Glossary

| Term | Simple meaning |
|---|---|
| **YOLOv8** | "You Only Look Once" — detects all objects in one pass through the network. Fast. |
| **Fine-tuning** | Taking a pretrained model and training it a bit more on your data |
| **Pretrained** | Already trained on a huge dataset by someone else — you start from their knowledge |
| **COCO128** | 128 images with 80 labeled object classes. Tiny version of the COCO dataset. |
| **Bounding box** | Rectangle drawn around a detected object (x, y, width, height) |
| **IoU** | Intersection over Union — how much your predicted box overlaps the real box. 1.0 = perfect. |
| **mAP50** | Main accuracy score. Averages precision across all 80 classes at IoU≥0.50. 0.72 = decent. |
| **mAP50-95** | Same as mAP50 but averaged across stricter IoU thresholds (0.50 to 0.95). The standard COCO metric. |
| **Precision** | Of all boxes the model drew, how many were real objects (not false alarms) |
| **Recall** | Of all real objects in the image, how many the model actually found |
| **freeze=10** | Lock first 10 network layers so they keep pretrained features. Only retrain the head. |
| **lr0=0.001** | Learning rate — size of each weight update step. Small = careful = preserves pretrained knowledge |
| **Epoch** | One full pass through all training images. 15 epochs = model sees each image 15 times. |
| **conf=0.25** | Confidence threshold — only show detections where model is ≥25% sure. Filters weak guesses. |
| **Backbone** | First layers of the network — detect edges, textures, shapes (low-level features) |
| **Head** | Last layers of the network — combine features into object predictions (class + box) |
| **AdamW** | The optimizer used — decides how to update weights based on gradients. Adaptive learning rate. |
| **AMP** | Automatic Mixed Precision — uses float16 where safe for 2x faster training with no accuracy loss |

---

## 9. Viva Q&A Quick Reference

**Q: What is YOLOv8?**  
A single-stage object detector — it predicts class labels and bounding boxes in one forward pass. Two-stage detectors like R-CNN first propose regions, then classify. YOLO does both simultaneously, making it much faster.

**Q: What does "fine-tuning" mean?**  
Taking a model already trained on a large dataset and training it further on a smaller target dataset. The model keeps its learned features and adapts to the new data.

**Q: Why freeze=10?**  
The backbone layers detect low-level features already good from pretraining. With only 128 images, training them would overfit. Freezing them and training only the head is standard for small datasets.

**Q: Why lr0=0.001 instead of default 0.01?**  
The default 0.01 is for large datasets. With 128 images, high learning rate amplifies noise and destroys pretrained weights. 0.001 is 10x gentler.

**Q: What is mAP50?**  
Mean Average Precision at IoU 0.50. Measures detection accuracy averaged across all 80 classes. A detection is correct if its box overlaps ground truth by ≥50%.

**Q: What is IoU?**  
Intersection over Union — overlap area between predicted and ground-truth boxes divided by their union area. IoU=1.0 means perfect overlap.

**Q: Precision vs Recall?**  
Precision = correct detections / all detections made (fewer false alarms). Recall = found objects / all real objects (fewer misses). There is a tradeoff — lowering confidence threshold increases recall but decreases precision.

**Q: What is COCO128?**  
A 128-image subset of COCO, used for quick testing. Same 80 classes but 1000x fewer images. Not for production — just for validating the pipeline.

**Q: Why is mAP 0.72 and not higher?**  
128 images across 80 classes = ~1.6 images per class. More data would push mAP to 0.85+. The score is strong for the dataset size because we froze the backbone and used a low learning rate.

**Q: Why not just use the pretrained model?**  
Fine-tuning adapts the model to a specific domain. With a custom dataset (medical, traffic, retail), the fine-tuned model would outperform the pretrained one on domain-specific images.

---

## 10. References

1. Ultralytics. "YOLOv8 Documentation." https://docs.ultralytics.com
2. Lin, T. et al. "Microsoft COCO: Common Objects in Context." (2014). https://arxiv.org/abs/1405.0312
3. Redmon, J. et al. "You Only Look Once: Unified, Real-Time Object Detection." (2016). https://arxiv.org/abs/1506.02640
4. Jocher, G. et al. "Ultralytics YOLOv8." (2023). https://github.com/ultralytics/ultralytics
