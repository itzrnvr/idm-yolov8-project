# IDM Project - Real-Time Object Detection with YOLOv8

**CSET225 | Intelligent Design Models | Summer Term | Sanskar Singh**

## Overview

This project fine-tunes a YOLOv8 object detector on the COCO128 dataset and evaluates it with mAP metrics. YOLOv8 detects objects in a single forward pass across the 80 standard COCO classes (person, car, dog, etc.), making it fast enough for real-time applications.

## Project structure

```
IDM_YOLOv8_Project/
├── README.md                         # This guide
├── notebooks/
│   └── IDM_YOLOv8_Colab.ipynb        # Colab notebook: setup, train, evaluate, demo
├── scripts/
│   └── train_colab.py               # Headless training script for Colab CLI
├── data/
│   └── README.md                     # YOLO dataset format and Roboflow download instructions
└── report/
    └── REPORT.md                     # Report template for results and analysis
```

- `README.md`: project overview, Colab run path, and troubleshooting.
- `notebooks/IDM_YOLOv8_Colab.ipynb`: executable notebook that installs dependencies, loads a pretrained model, fine-tunes on COCO128, evaluates with mAP metrics, and runs sample inference.
- `data/README.md`: how to format YOLO datasets and download data from Roboflow.
- `report/REPORT.md`: scaffold for the final project report with section prompts.

## Quickstart - Google Colab

1. Open `notebooks/IDM_YOLOv8_Colab.ipynb` in Google Colab (upload the file, or open from GitHub if the repo is linked).
2. Choose **Runtime > Change runtime type** and select **T4 GPU**.
3. Run all cells from top to bottom with **Runtime > Run all**.
4. The notebook trains the model, prints evaluation metrics (mAP50, mAP50-95, precision, recall), and displays annotated inference results on sample images.

## Quickstart - Colab CLI (headless, no browser)

The [Colab CLI](https://github.com/googlecolab/google-colab-cli) runs the full training pipeline from the terminal on a remote T4 GPU — no browser tab required. A Windows-supported fork is available.

### One-time setup

Install and authenticate:
    pip install google-colab-cli
    colab auth

### Run the project

1. Check for an active session:
       colab sessions
2. Execute the headless training script on a T4 session:
       colab exec -s <session-name> -f scripts/train_colab.py
3. Download the trained weights and metrics:
       colab download -s <session-name> /content/runs/detect/train/weights/best.pt ./weights/best.pt
       colab download -s <session-name> /content/metrics.json ./runs/metrics.json
4. Release the VM when done:
       colab stop -s <session-name>

### Smoke test (1 epoch, ~2 min)

Edit `scripts/train_colab.py` and set `EPOCHS = 1`, then re-run step 2. Restore to 15 for the full run.

## Configuration

The training script (`scripts/train_colab.py`) and the notebook use these parameters:

```yaml
model: yolov8n.pt            # base pretrained weights (yolov8n/s/m/l/x)
data: coco128.yaml           # ultralytics data yaml name OR absolute path
epochs: 15
imgsz: 640
batch: 16
device: 0                    # 0 = first CUDA GPU; "cpu" for CPU
lr0: 0.001                   # low learning rate for small datasets
freeze: 10                   # freeze backbone layers, train head only
```

- `model`: YOLOv8 size. `yolov8n` is fastest and smallest; `yolov8s`, `yolov8m`, `yolov8l`, and `yolov8x` are progressively larger and more accurate.
- `data`: dataset definition. The default `coco128.yaml` auto-downloads. To use a custom Roboflow dataset, replace it with the path to the downloaded `data.yaml` file.
- `epochs`: number of training passes through the dataset.
- `batch`: images per training batch.
- `imgsz`: input resolution (square resize).
- `device`: GPU index (`0`) or `"cpu"`. Always use T4 GPU on Colab.
- `lr0`: learning rate. 0.001 (not default 0.01) prevents destroying pretrained features on small datasets.
- `freeze`: number of backbone layers to freeze. 10 = train only the detection head, preserving pretrained feature extractors.

## Expected outputs

- Fine-tuned weights: `runs/detect/train_fresh/weights/best.pt`
- Evaluation metrics printed by the notebook:
  - `mAP50` - mean Average Precision at IoU 0.50
  - `mAP50-95` - mean Average Precision averaged across IoU 0.50 to 0.95
  - `P` (precision) and `R` (recall)
- Sample inference results: annotated images displayed inline in the notebook
- Training plots: saved under `runs/detect/train_fresh/` by Ultralytics (results.png, confusion_matrix.png, etc.)

## Colab notes

- Free Colab provides a T4 GPU with limited session duration (around 12 hours).
- Weights must be downloaded before the session ends. The notebook's last cell copies `best.pt` to `/content/best.pt` for easy access via the Colab file browser.
- CPU training is not recommended; always select T4 GPU runtime.

## Troubleshooting

- **Session disconnects**: all files in the Colab runtime are lost. Re-run the training cells or re-upload a saved `best.pt` checkpoint.
- **CUDA not available**: confirm the runtime is set to **T4 GPU** under **Runtime > Change runtime type**.
- **Low mAP (near 0)**: ensure you reload a fresh `YOLO('yolov8n.pt')` before training. Reusing an already-trained model object with the default lr0=0.01 will destroy pretrained features. Use lr0=0.001 and freeze=10.
