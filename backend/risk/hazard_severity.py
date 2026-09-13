"""
Sea Sentinel: Layer A — Hazard Severity Model
Quantifies the inherent hazard magnitude of the marine debris object.
"""

from typing import Dict, Any
from .models import NormalizedRiskParameters, LayerASeverityOutput
from .config import HAZARD_SEVERITY_WEIGHTS


class HazardSeverityModel:
    """
    Computes Hazard Severity Score (HSS, 0-100) based on inherent physical,
    morphological, and acoustic properties of the target.
    """

    def __init__(self, custom_weights: Dict[str, float] = None):
        self.weights = custom_weights or HAZARD_SEVERITY_WEIGHTS

    def compute_severity(self, norm: NormalizedRiskParameters) -> LayerASeverityOutput:
        w = self.weights

        c_type = norm.debris_type_score * w.get("debris_type", 0.20)
        c_size = norm.size_volume_score * w.get("size_volume", 0.15)
        c_wc = norm.water_column_score * w.get("water_column_position", 0.10)
        c_ent = norm.entanglement_potential_score * w.get("entanglement_potential", 0.20)
        c_pers = norm.persistence_score * w.get("persistence", 0.15)
        c_vis = norm.visibility_detectability_score * w.get("visibility_detectability", 0.20)

        hss = c_type + c_size + c_wc + c_ent + c_pers + c_vis
        hss_clamped = float(max(0.0, min(100.0, hss)))

        contributions = {
            "debris_type": round(c_type, 2),
            "size_volume": round(c_size, 2),
            "water_column_position": round(c_wc, 2),
            "entanglement_potential": round(c_ent, 2),
            "persistence": round(c_pers, 2),
            "visibility_detectability": round(c_vis, 2)
        }

        return LayerASeverityOutput(
            hazard_severity_score=round(hss_clamped, 2),
            contributions=contributions
        )
