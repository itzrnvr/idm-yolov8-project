# IDM Project Summary

> Open this page during your viva. Read it top to bottom. Everything you need is here.

---

## What This Project Does

Detects objects (people, cars, buses, dogs, etc.) in images by drawing boxes around them with labels. Uses YOLOv8 — a neural network that looks at an image once and instantly predicts **what** objects are there and **where** they are (bounding box coordinates). 80 possible object types.

---

## How Long It Takes to Run

| Step | What | Time |
|------|------|------|
| Step 1 | Install ultralytics + CUDA check | ~15s |
| Step 2 | Download yolov8n.pt + baseline inference | ~15s |
| Step 3 | Download COCO128 + train 15 epochs (frozen backbone) | ~60s |
| Step 4 | Load best.pt + run validation | ~10s |
| Step 5 | Inference on 2 sample images + display | ~10s |
| Step 6 | Copy weights to /content/ | ~2s |

**Total: ~2 minutes on T4 GPU.** Step 3 is the longest (training). With `freeze=10` only the detection head trains, so each epoch is fast (~2-3s). The 60 seconds includes downloading the dataset + model + 15 epochs + auto-validation.

Watch the progress bar in the top right — green checkmark on the last cell means done.

---

## The 6 Steps, Plain English

### Step 1 — Setup
Install Ultralytics (the YOLOv8 package). Check that a GPU is connected. GPU = fast, CPU = slow.

### Step 2 — Baseline Test
Load a model that's already trained on 118,000 images by the YOLO creators. Run it on a bus photo. It detects 4 persons, 1 bus, 1 stop sign. This proves the pipeline works before we touch anything.

### Step 3 — Fine-Tune (the actual project work)
Take that pretrained model and train it further on COCO128 (128 images). Why? To adapt it. Key decisions:
- **lr0=0.001** — small learning steps so we don't erase what the model already knows
- **freeze=10** — lock the first 10 layers (they detect edges/textures/shapes — already good). Only retrain the final layers (the ones that decide "this is a person" vs "this is a car")
- **15 epochs** — show all 128 images to the model 15 times

### Step 4 — Evaluate
Run the trained model on the validation images and measure:
- **mAP50 = 0.72** — 72% accuracy at loose overlap (IoU ≥ 50%). Good for 128 images.
- **Precision = 0.73** — 73% of detections the model makes are correct
- **Recall = 0.65** — it finds 65% of all real objects in the images

### Step 5 — Demo
Run the fine-tuned model on two test photos (bus.jpg, zidane.jpg). Draws boxes around detected objects. Displays results inline.

### Step 6 — Save Weights
Copies the trained model file (best.pt, 6.2 MB) to /content/ so you can download it.

---

## Key Terms for Viva

| Term | Simple meaning |
|------|----------------|
| **YOLOv8** | "You Only Look Once" — detects all objects in one pass through the network. Fast. |
| **Fine-tuning** | Taking a pretrained model and training it a bit more on your data |
| **Pretrained** | Already trained on a huge dataset by someone else — you start from their knowledge |
| **COCO128** | 128 images with 80 labeled object classes. Tiny version of the COCO dataset. |
| **Bounding box** | Rectangle drawn around a detected object (x, y, width, height) |
| **IoU** | Intersection over Union — how much your predicted box overlaps the real box. 1.0 = perfect. |
| **mAP50** | Main accuracy score. Averages precision across all 80 classes at IoU≥0.50. 0.72 = decent. |
| **Precision** | Of all boxes the model drew, how many were real objects (not false alarms) |
| **Recall** | Of all real objects in the image, how many the model actually found |
| **freeze=10** | Lock first 10 network layers so they keep pretrained features. Only retrain the head. |
| **lr0=0.001** | Learning rate — size of each weight update step. Small = careful = preserves pretrained knowledge |
| **Epoch** | One full pass through all training images. 15 epochs = model sees each image 15 times. |
| **conf=0.25** | Confidence threshold — only show detections where model is ≥25% sure. Filters weak guesses. |

---

## If Asked "Why Not Just Use the Pretrained Model?"

> "Fine-tuning adapts the model to a specific domain. The pretrained model is trained on general COCO images. If I had a custom dataset — say, medical scans or traffic cameras — fine-tuning would specialize the model for that domain. COCO128 is a proof-of-concept showing the pipeline works; with a real dataset the fine-tuned model would outperform the pretrained one on domain-specific images."

## If Asked "Why Is mAP Only 0.72?"

> "128 training images across 80 classes is extremely small — about 1.6 images per class. A model trained on the full COCO dataset (118,000 images) achieves ~0.50 mAP50 on the same benchmark. Our 0.72 is actually higher because we froze the backbone and used a low learning rate, preserving pretrained features while adapting the head. More data would push this to 0.85+."
