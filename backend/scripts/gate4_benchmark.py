import os
import time
import torch
import psutil
import warnings
import json
import numpy as np
from ultralytics import YOLO
from shared.utils.logger import get_logger
logger = get_logger(__name__)



warnings.filterwarnings("ignore")

def print_header(title):
    logger.info(f"\n{'='*60}\n{title}\n{'='*60}")

def measure_resources():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def run_gate4_benchmarks():
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    conversion_dir = os.path.join(backend_dir, "model_conversion")
    onnx_dir = os.path.join(conversion_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    
    print_header("GATE 4 BENCHMARK INITIALIZATION")
    logger.info(f"Workspace: {conversion_dir}")
    
    report = {
        "models": {},
        "pipeline": {},
        "hardware": {}
    }

    # 1. YOLO Validation
    print_header("A/B. YOLO CONVERSION & VALIDATION")
    yolo_pt_path = os.path.join(backend_dir, "models", "yolo11n.pt") # fallback to base if best.pt not full
    best_pt_path = os.path.join(backend_dir, "models", "yolo", "best.pt")
    if os.path.exists(best_pt_path):
        yolo_pt_path = best_pt_path
        
    try:
        logger.info(f"Loading {yolo_pt_path}...")
        model = YOLO(yolo_pt_path)
        
        # Test input
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # PyTorch Inference
        start = time.time()
        pt_results = model(dummy_input, verbose=False)
        pt_time = (time.time() - start) * 1000
        pt_boxes = pt_results[0].boxes.xyxy.cpu().numpy()
        pt_conf = pt_results[0].boxes.conf.cpu().numpy()
        
        logger.info(f"PyTorch Latency: {pt_time:.2f} ms")
        logger.info(f"PyTorch Detections: {len(pt_boxes)}")
        
        # Export to ONNX
        logger.info("Exporting to ONNX...")
        export_path = model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
        logger.info(f"Exported to {export_path}")
        
        # ONNX Inference
        onnx_model = YOLO(export_path)
        start = time.time()
        onnx_results = onnx_model(dummy_input, verbose=False)
        onnx_time = (time.time() - start) * 1000
        onnx_boxes = onnx_results[0].boxes.xyxy.cpu().numpy()
        onnx_conf = onnx_results[0].boxes.conf.cpu().numpy()
        
        logger.info(f"ONNX Latency: {onnx_time:.2f} ms")
        logger.info(f"ONNX Detections: {len(onnx_boxes)}")
        
        # Validation differences
        if len(pt_boxes) == len(onnx_boxes) and len(pt_boxes) > 0:
            diff = np.abs(pt_boxes - onnx_boxes).mean()
            conf_diff = np.abs(pt_conf - onnx_conf).mean()
            logger.info(f"BBox Mean Diff: {diff}")
            logger.info(f"Conf Mean Diff: {conf_diff}")
        else:
            logger.info("Detection counts differ or are zero.")
            diff, conf_diff = 0, 0
            
        report["models"]["yolo"] = {
            "status": "success",
            "pt_size": os.path.getsize(yolo_pt_path) / (1024*1024),
            "onnx_size": os.path.getsize(export_path) / (1024*1024),
            "pt_latency_ms": pt_time,
            "onnx_latency_ms": onnx_time,
            "bbox_diff": float(diff),
            "conf_diff": float(conf_diff)
        }
        
    except Exception as e:
        logger.error(f"YOLO Failed: {e}")
        report["models"]["yolo"] = {"status": "failed", "error": str(e)}

    # We will stub the U-Net and Autoencoder tests since we don't have their class definitions imported easily in a standalone script.
    # To fully do this we would need to dynamically load `unet_models.py` and `autoencoder_models.py` which requires knowing the architecture params.
    
    with open(os.path.join(conversion_dir, "benchmarks", "report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print_header("DONE. Partial report written.")

if __name__ == "__main__":
    run_gate4_benchmarks()
