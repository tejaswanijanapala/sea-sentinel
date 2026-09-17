"""debris_risk_scoring module (consolidated)"""


# --- Extracted from debris_risk_scoring\routes\risk_routes.py ---
"""Debris Risk Scoring API Routes."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/debris-risk", tags=["Debris Risk Scoring"])
risk_service = DebrisRiskScoringService()

class RiskRequest(BaseModel):
    class_name: str
    confidence: float
    area_px: int

@router.post("/score")
def score_debris(req: RiskRequest):
    res = risk_service.calculate_risk(req.class_name, req.confidence, req.area_px)
    return {"status": "success", "risk": res}


# --- Extracted from debris_risk_scoring\services\risk_engine_service.py ---
"""Marine Debris Hazard and Risk Priority Engine."""
from typing import Dict, Any
from backend.shared.config.config_loader import get_pipeline_config

class DebrisRiskScoringService:
    def __init__(self):
        cfg = get_pipeline_config()
        self.hazard_weights = cfg.get("risk_priority", {}).get("hazard_weights", {
            "fishing_net": 90,
            "shipwreck_fragment": 85,
            "engine_debris": 80,
            "pipeline_or_cable": 70,
            "plastic_debris": 60,
            "default": 50
        })

    def calculate_risk(self, class_name: str, confidence: float, area_px: int) -> Dict[str, Any]:
        weight = self.hazard_weights.get(class_name, self.hazard_weights.get("default", 50))
        area_factor = min(1.0, area_px / 10000.0)
        raw_score = 0.50 * weight + 0.30 * (confidence * 100.0) + 0.20 * (area_factor * 100.0)
        score = round(min(100.0, max(0.0, raw_score)), 1)
        
        if score >= 75:
            level = "CRITICAL"
        elif score >= 50:
            level = "HIGH"
        elif score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"
            
        return {
            "risk_score": score,
            "risk_level": level,
            "hazard_weight": weight,
            "recommended_action": "ROV Retrieval" if level in ["CRITICAL", "HIGH"] else "Monitor"
        }

