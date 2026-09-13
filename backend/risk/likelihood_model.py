"""
Sea Sentinel: Layer B — Exposure / Likelihood Model
Quantifies the operational and environmental interaction probability of the marine debris.
Incorporates non-linear interaction terms (e.g. High Mobility x Proximity to Shipping Routes).
"""

from typing import Dict, Any
from .models import NormalizedRiskParameters, LayerBLikelihoodOutput
from .config import LIKELIHOOD_WEIGHTS


class LikelihoodExposureModel:
    """
    Computes Likelihood Score (LS, 0-100) and normalized L (0-1) for base risk equation.
    Models non-linear compound exposures.
    """

    def __init__(self, custom_weights: Dict[str, float] = None):
        self.weights = custom_weights or LIKELIHOOD_WEIGHTS

    def compute_likelihood(self, norm: NormalizedRiskParameters) -> LayerBLikelihoodOutput:
        w = self.weights

        c_nav = norm.navigation_distance_score * w.get("navigation_exposure", 0.30)
        c_depth = norm.depth_exposure_score * w.get("depth_interaction", 0.15)
        c_size = norm.size_volume_score * w.get("size_encounter_footprint", 0.15)
        c_wc = norm.water_column_score * w.get("water_column_position", 0.10)
        c_mob = norm.mobility_score * w.get("mobility", 0.10)
        c_dens = norm.debris_density_score * w.get("debris_density", 0.10)
        c_rec = norm.recurrence_score * w.get("recurrence", 0.05)
        c_det = (100.0 - norm.visibility_detectability_score * 0.4) * w.get("visibility_detectability", 0.05)

        base_ls = c_nav + c_depth + c_size + c_wc + c_mob + c_dens + c_rec + c_det

        # Non-Linear Interaction Terms:
        interaction_bonus = 0.0
        # High Mobility + Close Navigation Route creates compound encounter probability
        if norm.mobility_score >= 60.0 and norm.navigation_distance_score >= 60.0:
            interaction_bonus += ((norm.mobility_score - 60.0) / 40.0) * ((norm.navigation_distance_score - 60.0) / 40.0) * 10.0

        # Water Column near-surface bonus for shallow depth
        if norm.water_column_score >= 80.0 and norm.depth_exposure_score >= 75.0:
            interaction_bonus += 6.0

        # Large size in primary navigation corridor bonus
        if norm.size_volume_score >= 75.0 and norm.navigation_distance_score >= 70.0:
            interaction_bonus += 5.0

        final_ls = float(max(10.0, min(100.0, base_ls + interaction_bonus)))
        l_norm = final_ls / 100.0

        contributions = {
            "navigation_exposure": round(c_nav, 2),
            "depth_interaction": round(c_depth, 2),
            "size_encounter_footprint": round(c_size, 2),
            "water_column_position": round(c_wc, 2),
            "mobility": round(c_mob, 2),
            "debris_density": round(c_dens, 2),
            "recurrence": round(c_rec, 2),
            "detectability": round(c_det, 2),
            "interaction_term": round(interaction_bonus, 2)
        }

        return LayerBLikelihoodOutput(
            likelihood_score=round(final_ls, 2),
            likelihood_normalized=round(l_norm, 4),
            interaction_bonus=round(interaction_bonus, 2),
            contributions=contributions
        )
