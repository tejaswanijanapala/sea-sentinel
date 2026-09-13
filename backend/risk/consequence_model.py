"""
Sea Sentinel: Layer C — Consequence Model
Quantifies the multi-domain impact severity if an encounter/interaction occurs.
Separated into Navigation, Ecological, Operational/Economic, and Human-Safety dimensions.
"""

from typing import Dict, Any, Optional
from .models import NormalizedRiskParameters, LayerCConsequenceOutput
from .config import CONSEQUENCE_CONTEXT_PROFILES


class ConsequenceModel:
    """
    Computes Consequence Score (CS, 0-100) and normalized C (0-1) across 4 dimensions:
    - Navigation Consequence
    - Ecological Consequence
    - Operational/Economic Consequence
    - Human-Safety Consequence
    """

    @staticmethod
    def compute_navigation_consequence(norm: NormalizedRiskParameters) -> float:
        """
        Navigation Impact: Collision severity, propeller fouling, rudder jamming, fairway blockage.
        Driven by Size, Water Column Position, Entanglement Potential, and Potential Consequence.
        """
        # Water column position heavily dictates hull/propeller strike potential
        wc_factor = norm.water_column_score / 100.0  # floating/near surface = high
        size_factor = norm.size_volume_score / 100.0

        c_nav = (
            (norm.potential_consequence_score * 0.35) +
            (norm.size_volume_score * 0.25) +
            (norm.water_column_score * 0.20) +
            (norm.entanglement_potential_score * 0.20)
        )
        # Propeller entanglement boost for nets/ropes
        if norm.entanglement_potential_score >= 80.0:
            c_nav = min(100.0, c_nav * 1.15)
        # Heavy floating container collision boost
        if norm.size_volume_score >= 80.0 and norm.water_column_score >= 80.0:
            c_nav = min(100.0, c_nav * 1.20)

        return float(max(5.0, min(100.0, c_nav)))

    @staticmethod
    def compute_ecological_consequence(norm: NormalizedRiskParameters) -> float:
        """
        Ecological Impact: Ghost fishing mortality, coral smothering, microplastic breakdown, toxic leaching.
        Driven by Entanglement Potential, Ecological Sensitivity, Persistence, and Debris Type.
        """
        c_eco = (
            (norm.entanglement_potential_score * 0.35) +
            (norm.ecological_sensitivity_score * 0.30) +
            (norm.persistence_score * 0.20) +
            (norm.debris_type_score * 0.15)
        )
        # Ghost gear in sensitive habitat compounding multiplier
        if norm.entanglement_potential_score >= 80.0 and norm.ecological_sensitivity_score >= 70.0:
            c_eco = min(100.0, c_eco * 1.25)

        return float(max(5.0, min(100.0, c_eco)))

    @staticmethod
    def compute_operational_economic_consequence(norm: NormalizedRiskParameters) -> float:
        """
        Economic/Operational Impact: Trawl net damage, pipeline/cable disruption, salvage cost, port delay.
        Driven by Size, Debris Density, Potential Consequence, and Debris Type.
        """
        c_ops = (
            (norm.potential_consequence_score * 0.35) +
            (norm.size_volume_score * 0.30) +
            (norm.debris_density_score * 0.20) +
            (norm.recurrence_score * 0.15)
        )
        return float(max(5.0, min(100.0, c_ops)))

    @staticmethod
    def compute_human_safety_consequence(norm: NormalizedRiskParameters) -> float:
        """
        Human Safety Impact: Small craft capsize, diver entanglement, unexploded ordnance detonation.
        Driven by Debris Type, Water Column Position, Size, and Potential Consequence.
        """
        c_safe = (
            (norm.debris_type_score * 0.40) +
            (norm.water_column_score * 0.30) +
            (norm.potential_consequence_score * 0.30)
        )
        # Munitions / UXO critical override
        if norm.debris_type_score >= 95.0:
            c_safe = max(95.0, c_safe)

        return float(max(5.0, min(100.0, c_safe)))

    def compute_consequence(
        self,
        norm: NormalizedRiskParameters,
        context_name: str = "STANDARD_BALANCED"
    ) -> LayerCConsequenceOutput:
        c_nav = self.compute_navigation_consequence(norm)
        c_eco = self.compute_ecological_consequence(norm)
        c_ops = self.compute_operational_economic_consequence(norm)
        c_safe = self.compute_human_safety_consequence(norm)

        profile = CONSEQUENCE_CONTEXT_PROFILES.get(context_name, CONSEQUENCE_CONTEXT_PROFILES["STANDARD_BALANCED"])
        w_nav = profile.get("navigation", 0.35)
        w_eco = profile.get("ecological", 0.35)
        w_ops = profile.get("operational_economic", 0.15)
        w_safe = profile.get("human_safety", 0.15)

        composite_cs = (c_nav * w_nav) + (c_eco * w_eco) + (c_ops * w_ops) + (c_safe * w_safe)
        composite_cs = float(max(5.0, min(100.0, composite_cs)))
        c_norm = composite_cs / 100.0

        return LayerCConsequenceOutput(
            consequence_score=round(composite_cs, 2),
            consequence_normalized=round(c_norm, 4),
            navigation_consequence=round(c_nav, 2),
            ecological_consequence=round(c_eco, 2),
            operational_economic_consequence=round(c_ops, 2),
            human_safety_consequence=round(c_safe, 2),
            context_profile_used=context_name
        )
