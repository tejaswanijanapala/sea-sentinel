"""sonar_quality module (consolidated)"""


# --- Extracted from sonar_quality\routes\quality_routes.py ---
"""Sonar Quality Routes."""
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np

router = APIRouter(prefix="/sonar-quality", tags=["Sonar Quality"])

class QualityRequest(BaseModel):
    mean_backscatter: float = 75.0
    noise_variance: float = 22.0

@router.post("/evaluate")
def evaluate_quality(req: QualityRequest):
    enl = (req.mean_backscatter ** 2) / (req.noise_variance ** 2 + 1e-5)
    return {
        "status": "success",
        "equivalent_number_of_looks": round(enl, 2),
        "quality_rating": "OPTIMAL_SURVEY_GRADE" if enl > 6.0 else "MODERATE"
    }


# --- Extracted from sonar_quality\services\quality_service.py ---
"""Sonar Image Quality & Speckle Metric Evaluator."""
import cv2
import numpy as np
from typing import Dict, Any

class SonarQualityService:
    @staticmethod
    def evaluate_quality(img_gray: np.ndarray) -> Dict[str, Any]:
        mean_val = float(np.mean(img_gray))
        std_val = float(np.std(img_gray))
        enl = (mean_val ** 2) / (std_val ** 2 + 1e-5) # Equivalent Number of Looks
        
        contrast = float(np.max(img_gray) - np.min(img_gray)) / float(np.max(img_gray) + np.min(img_gray) + 1e-5)
        
        if enl > 8.0 and contrast > 0.6:
            rating = "OPTIMAL_SURVEY_GRADE"
        elif enl > 4.0:
            rating = "GOOD_ACOUSTIC_CLARITY"
        else:
            rating = "HIGH_SPECKLE_REVERBERATION"
            
        return {
            "mean_backscatter": round(mean_val, 2),
            "speckle_noise_variance": round(std_val, 2),
            "equivalent_number_of_looks": round(enl, 2),
            "michelson_contrast": round(contrast, 3),
            "quality_rating": rating
        }

