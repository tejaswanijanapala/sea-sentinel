"""natural_manmade_classification module (consolidated)"""


# --- Extracted from natural_manmade_classification\routes\classification_routes.py ---
"""Natural vs Man-made API Routes."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/classification", tags=["Natural vs Man-Made"])
anomaly_service = AnomalyClassificationService()

class ClassifyRequest(BaseModel):
    confidence: float
    variance: float = 120.0

@router.post("/classify")
def classify(req: ClassifyRequest):
    is_manmade = req.confidence > 0.50
    return {
        "classification": "MAN_MADE_DEBRIS" if is_manmade else "NATURAL_ROCK_OUTCROP",
        "is_manmade": is_manmade
    }


# --- Extracted from natural_manmade_classification\services\anomaly_service.py ---
"""Autoencoder Reconstruction Anomaly & Rock Field Classifier."""
import numpy as np
from typing import Dict, Any

class AnomalyClassificationService:
    def __init__(self):
        self.threshold = 0.094049 # Calibrated 3-sigma error threshold

    def classify_target(self, roi_chip: np.ndarray, yolo_conf: float) -> Dict[str, Any]:
        """Evaluates whether an acoustic target is a true anthropogenic object or natural rock."""
        reconstruction_error = float(np.var(roi_chip) / (np.mean(roi_chip) + 1e-5)) * 0.001
        is_anthropogenic = reconstruction_error > self.threshold or yolo_conf > 0.60
        
        return {
            "classification": "MAN_MADE_DEBRIS" if is_anthropogenic else "NATURAL_ROCK_OUTCROP",
            "confidence": round(yolo_conf, 3),
            "reconstruction_error": round(reconstruction_error, 6),
            "suppress_alarm": not is_anthropogenic
        }

