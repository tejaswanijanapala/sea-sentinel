"""debris_density module (consolidated)"""


# --- Extracted from debris_density\routes\density_routes.py ---
"""Debris Density Routes."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(prefix="/debris-density", tags=["Debris Density"])

class DensityRequest(BaseModel):
    targets: List[Dict[str, Any]]
    swath_area_sq_m: float = 5000.0

@router.post("/estimate")
def estimate_density(req: DensityRequest):
    res = DebrisDensityService.calculate_density(req.targets, req.swath_area_sq_m)
    return {"status": "success", "density": res}


# --- Extracted from debris_density\services\density_service.py ---
"""Debris Density Heatmap & Spatial Clustering Service."""
from typing import List, Dict, Any

class DebrisDensityService:
    @staticmethod
    def calculate_density(targets: List[Dict[str, Any]], swath_area_sq_m: float = 5000.0) -> Dict[str, Any]:
        count = len(targets)
        density_per_1000m2 = round((count / max(1.0, swath_area_sq_m)) * 1000.0, 2)
        
        if density_per_1000m2 > 5.0:
            category = "HIGH_CONCENTRATION_ACCUMULATION_ZONE"
        elif density_per_1000m2 > 1.5:
            category = "MODERATE_DEBRIS_FIELD"
        else:
            category = "SPARSE_ISOLATED_TARGETS"
            
        return {
            "target_count": count,
            "swath_area_sq_m": swath_area_sq_m,
            "density_per_1000m2": density_per_1000m2,
            "accumulation_category": category
        }

