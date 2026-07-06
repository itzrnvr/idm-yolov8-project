"""IDM Project - Headless YOLOv8 training script for Colab CLI execution.

Runs the full pipeline on a Colab VM without any interactive (browser) cells:
  install -> CUDA check -> baseline inference -> fine-tune -> evaluate -> save artifacts

Execute on a Colab T4 via the Colab CLI:
    colab new -s idm-yolo --gpu T4
    colab exec -s idm-yolo -f scripts/train_colab.py
    colab download -s idm-yolo /content/runs/detect/train/weights/best.pt ./weights/best.pt
    colab download -s idm-yolo /content/metrics.json ./runs/metrics.json
    colab stop -s idm-yolo

Course: CSET225 | Intelligent Design Models | Summer Term
Author: Sanskar Singh
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
# Epochs: 25 for the full run. Override via argv, e.g. colab run ... train_colab.py 1
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
BASE_MODEL = "yolov8n.pt"          # n / s / m / l / x  (smallest -> largest)
DATA = "coco128.yaml"              # auto-downloads; swap for a custom data.yaml
IMGSZ = 640
BATCH = 16
DEVICE = 0                         # first CUDA GPU


def step(n, total, msg):
    print(f"\n[STEP {n}/{total}] {msg}", flush=True)
    print("-" * 60, flush=True)


def main():
    total_steps = 6
    t0 = time.time()

    # ── Step 1: Install dependencies ───────────────────────────────────────
    step(1, total_steps, "Installing ultralytics...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics"]
    )

    # ── Step 2: Verify CUDA ────────────────────────────────────────────────
    step(2, total_steps, "Verifying CUDA...")
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Training will be very slow on CPU.")
        print("Ensure the Colab session was provisioned with --gpu T4.")

    # ── Step 3: Load pretrained model + baseline inference ─────────────────
    step(3, total_steps, "Loading pretrained YOLOv8 and running baseline inference...")
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    baseline = model.predict(
        "https://ultralytics.com/images/bus.jpg", conf=0.25, save=True
    )
    print(f"Baseline inference saved to: {baseline[0].save_dir}")

    # ── Step 4: Fine-tune on COCO128 ───────────────────────────────────────
    step(4, total_steps, f"Fine-tuning {BASE_MODEL} on {DATA} for {EPOCHS} epochs...")
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        lr0=0.001,
        freeze=10,
        patience=20,
        name="train",
    )
    weights_path = Path("runs/detect/train/weights/best.pt")
    best = YOLO(str(weights_path))
    metrics = best.val(data=DATA)

    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    print()
    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"  mAP50-95 : {map5095:.4f}")
    print(f"  mAP50    : {map50:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print("=" * 60)

    # ── Step 6: Save artifacts for download ────────────────────────────────
    step(6, total_steps, "Saving artifacts...")
    metrics_json = {
        "model": BASE_MODEL,
        "data": DATA,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "mAP50-95": round(map5095, 4),
        "mAP50": round(map50, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "training_time_seconds": round(time.time() - t0, 1),
    }
    metrics_path = Path("/content/metrics.json")
    metrics_path.write_text(json.dumps(metrics_json, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Best weights at:  {weights_path.resolve()}")
    print()
    print("Download locally with:")
    print(f"  colab download -s <session> {weights_path} ./weights/best.pt")
    print(f"  colab download -s <session> {metrics_path} ./runs/metrics.json")
    print()
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
