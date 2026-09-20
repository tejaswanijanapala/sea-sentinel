# Sea Sentinel: AI Model Retraining & Parallel Inference Guide

This guide documents the procedures for completely retraining the YOLO and Attention U-Net models from scratch, and explains the independent parallel inference architecture.

## Dataset Preparation
Original datasets are preserved in `ml_pipeline/datasets/processed/`.
For retraining, datasets are securely copied to `ml_pipeline/datasets/retraining/`.
- **YOLO format**: Images and normalized `.txt` bounding boxes.
- **U-Net format**: Images and binary `.png` mask images.

## Dataset Validation
Before training, run the dataset audit scripts:
```bash
python ml_pipeline/scripts/audit_yolo_dataset.py
python ml_pipeline/scripts/audit_unet_dataset.py
```
This generates `YOLO_DATASET_AUDIT.md` and `UNET_DATASET_AUDIT.md`.

## Training Commands
To retrain YOLO from scratch (randomly initialized `yolo11n.yaml`):
```bash
python ml_pipeline/training/train_yolo.py --data datasets/retraining/yolo/dataset.yaml --epochs 100 --batch 16
```
To retrain Attention U-Net from scratch (BCE+Dice loss):
```bash
python ml_pipeline/training/train_unet.py --data-dir datasets/retraining/unet --epochs 50 --batch 8
```

## Model Locations
- **Retrained Models**: Saved automatically to `ml_pipeline/models/retrained/yolo` and `ml_pipeline/models/retrained/attention_unet`.
- **Production Models**: Remain untouched in `models/yolo` and `models/unet`.

## Model Switching
Production models are not overwritten automatically. To deploy new models, update the environment variables or configuration:
```env
YOLO_MODEL_PATH=models/retrained/yolo/best.pt
UNET_MODEL_PATH=models/retrained/attention_unet/best.pt
```

## Parallel Inference Architecture
The Sea Sentinel inference pipeline now executes YOLO and U-Net completely independently.
- **Execution**: Both models receive the preprocessed sonar image simultaneously using `concurrent.futures.ThreadPoolExecutor`.
- **Independence**: YOLO does NOT wait for U-Net. U-Net does NOT wait for YOLO.
- **Fault Tolerance**: If YOLO fails, U-Net still executes and vice versa.

## Fusion Logic
After both models complete execution, the `FusionEngine` merges the results:
- **BOTH**: Candidate detected by both YOLO (bounding box) and U-Net (segmentation mask) with high IoU/Centroid proximity.
- **YOLO_ONLY**: Object found by YOLO but missed by U-Net.
- **UNET_ONLY**: Object segmented by U-Net but missed by YOLO.
No raw data is discarded; all findings are passed to the risk assessment layer.

## API Output
The API `/api/v2/parallel-inference/analyze` provides explicit visibility into the independent model paths:
```json
{
  "yolo": { "status": "completed", "detections": [...] },
  "attention_unet": { "status": "completed", "segmentation": {...} },
  "fusion": { "status": "completed", "objects": [...] },
  "anomaly": {...},
  "risk": {...}
}
```

## Troubleshooting & Deployment
- **Edge Deployment**: Only the final `.pt` models should be deployed to NVIDIA Jetsons. Do NOT copy the `datasets/` folder to edge devices.
- **Cloud/Render**: Training scripts are strictly separated from `backend/app.py`. The backend will NEVER initiate a training loop on server startup.
