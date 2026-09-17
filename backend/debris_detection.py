"""debris_detection module (consolidated)"""


# --- Extracted from debris_detection\routes\detection_routes.py ---
"""FastAPI Routes for Debris Detection."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/debris-detection", tags=["Debris Detection"])
detector = YoloDetectorService()

class DetectRequest(BaseModel):
    image_path: str
    conf_threshold: Optional[float] = 0.25

@router.post("/detect")
def detect_debris(req: DetectRequest):
    try:
        detections = detector.detect(req.image_path, req.conf_threshold)
        return {"status": "success", "count": len(detections), "detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Extracted from debris_detection\services\yolo_service.py ---
"""YOLO Detection Service."""
import os
from pathlib import Path
from typing import List, Dict, Any, Union
import cv2
import numpy as np
from ultralytics import YOLO
from backend.shared.utils.logger import get_logger
from backend.shared.config.config_loader import get_pipeline_config

logger = get_logger("yolo_service", "debris-detection")

class YoloDetectorService:
    def __init__(self, model_path: Optional[str] = None):
        cfg = get_pipeline_config()
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        
        default_model = PROJECT_ROOT / "models" / "yolo" / "best.pt"
        if not default_model.exists():
            default_model = PROJECT_ROOT / "models" / "yolo" / "yolo11n.pt"
            
        self.model_path = str(model_path or default_model)
        self.conf_thresh = cfg.get("yolo", {}).get("conf_threshold", 0.25)
        self.classes = cfg.get("risk_priority", {}).get("hazard_weights", {})
        
        logger.info(f"Loading YOLO Model from: {self.model_path}")
        self.model = YOLO(self.model_path)

    def detect(self, image: Union[str, np.ndarray], conf: Optional[float] = None) -> List[Dict[str, Any]]:
        threshold = conf if conf is not None else self.conf_thresh
        results = self.model.predict(source=image, conf=threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names.get(cls_id, f"target_{cls_id}")
            
            x1, y1, x2, y2 = xyxy
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(confidence, 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "area_px": int((x2 - x1) * (y2 - y1))
            })
        logger.info(f"YOLO detected {len(detections)} targets (conf threshold: {threshold}).")
        return detections

