import os
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml_pipeline/datasets/processed/yolo_dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    
    print("Executing Phase 5 & 6: YOLO Retraining")
    
    # Initialize completely from scratch (no weights loaded)
    # yolo11n.yaml contains architecture
    model = YOLO("yolo11n.yaml")
    
    project_dir = Path("ml_pipeline/models/retrained/yolo")
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        project=str(project_dir),
        name="run",
        exist_ok=True,
        device="cpu",
        seed=42
    )
    
    # Write metadata
    metadata = {
        "model_type": "YOLO",
        "architecture": "yolo11n",
        "training_from_scratch": True,
        "dataset_version": "retraining_v1",
        "training_seed": 42,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "training_date": datetime.now().isoformat()
    }
    
    with open(project_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"YOLO Retraining complete. Model saved to {project_dir}/run/weights/best.pt")

if __name__ == "__main__":
    main()
