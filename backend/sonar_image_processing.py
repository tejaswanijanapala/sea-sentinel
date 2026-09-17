"""sonar_image_processing module (consolidated)"""


# --- Extracted from sonar_image_processing\routes\processing_routes.py ---
"""Sonar Image Processing API Routes."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import cv2

router = APIRouter(prefix="/sonar-processing", tags=["Sonar Image Processing"])

class ProcessRequest(BaseModel):
    image_path: str
    clip_limit: float = 2.5

@router.post("/enhance")
def enhance_image(req: ProcessRequest):
    try:
        img = cv2.imread(req.image_path)
        if img is None:
            raise HTTPException(status_code=404, detail="Image not found")
        enhanced = SonarFilterService.enhance(img, req.clip_limit)
        return {"status": "success", "shape": list(enhanced.shape)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Extracted from sonar_image_processing\services\sonar_filter_service.py ---
"""Sonar Speckle Filtering and Radiometric CLAHE Service."""
import cv2
import numpy as np
from typing import Tuple
from backend.shared.utils.logger import get_logger

logger = get_logger("sonar_filter", "sonar-image-processing")

class SonarFilterService:
    @staticmethod
    def lee_filter(img: np.ndarray, win_size: int = 5, cu: float = 0.22) -> np.ndarray:
        img_f = img.astype(np.float32)
        mean_kernel = np.ones((win_size, win_size), dtype=np.float32) / (win_size ** 2)
        
        local_mean = cv2.filter2D(img_f, -1, mean_kernel)
        local_sq_mean = cv2.filter2D(img_f ** 2, -1, mean_kernel)
        local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)
        
        ci = np.sqrt(local_var) / (local_mean + 1e-5)
        weight = np.clip(1.0 - (cu ** 2) / (ci ** 2 + 1e-5), 0.0, 1.0)
        filtered = local_mean + weight * (img_f - local_mean)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    @classmethod
    def enhance(cls, img: np.ndarray, clip_limit: float = 2.5, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        denoised = cls.lee_filter(gray, win_size=5, cu=0.22)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        enhanced = clahe.apply(denoised)
        logger.debug(f"Enhanced sonar image: {img.shape} => {enhanced.shape}")
        return enhanced

