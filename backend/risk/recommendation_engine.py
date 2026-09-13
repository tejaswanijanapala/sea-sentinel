"""
Sea Sentinel: Risk-Control Recommendation Engine
Generates actionable maritime safety and environmental risk control measures
with explicit causal justifications based on IMO FSA Step 3 principles.
"""

from typing import List, Tuple
from .models import (
    RawRiskParameters,
    NormalizedRiskParameters,
    LayerASeverityOutput,
    LayerBLikelihoodOutput,
    LayerCConsequenceOutput,
    LayerDUncertaintyOutput
)


class RiskRecommendationEngine:
    """
    Synthesizes risk factors into targeted, operational risk-control recommendations:
    - Navigational Warning (NOTMAR / Coastal Navtex)
    - High-Resolution AUV / ROV Visual Verification
    - Surface Drift Monitoring & Interception
    - Immediate Ecological Containment & Ghost Gear Retrieval
    - Routine Survey Archiving & Passive Monitoring
    """

    def generate_recommendations(
        self,
        raw: RawRiskParameters,
        norm: NormalizedRiskParameters,
        layer_a: LayerASeverityOutput,
        layer_b: LayerBLikelihoodOutput,
        layer_c: LayerCConsequenceOutput,
        layer_d: LayerDUncertaintyOutput,
        final_risk_score: float,
        risk_level: str
    ) -> Tuple[List[str], str]:
        actions = []
        reasons = []

        # 1. Critical Navigation Hazard (< 100m from route, near-surface/floating or high size)
        if norm.navigation_distance_score >= 75.0 and (norm.water_column_score >= 60.0 or norm.size_volume_score >= 70.0):
            actions.append("Issue Coastal Navigational Warning (NOTMAR / Navtex) advising transit caution in local sector.")
            reasons.append("Severe collision or propulsion fouling hazard in immediate proximity to commercial shipping lane.")

        # 2. Entanglement / Ghost Net Ecological Threat
        if norm.entanglement_potential_score >= 80.0:
            actions.append("Deploy ROV/AUV equipped with cutting/retrieval manipulator for ghost gear extraction.")
            reasons.append(f"High entanglement morphology ({raw.debris_type}) poses continuous threat of ghost fishing and benthic degradation.")

        # 3. High Mobility / Drifting Debris
        if norm.mobility_score >= 70.0:
            actions.append("Initiate active drift tracking and schedule hydrodynamic surveillance intercept within 12–24h.")
            reasons.append(f"High mobility trajectory ({raw.mobility_class}) threatens to traverse active navigation fairways or marine habitats.")

        # 4. Spatial / AI Uncertainty Trigger
        if layer_d.position_verification_required or layer_d.risk_confidence_score < 50.0:
            actions.append("Execute targeted high-frequency micro-grid sonar pass for positional verification and acoustic confirmation.")
            reasons.append(f"Positional uncertainty buffer (±{layer_d.uncertainty_buffer_radius_m:.1f}m) overlaps critical boundary or AI certainty is low.")

        # 5. Ecological Habitat Proximity
        if norm.ecological_sensitivity_score >= 75.0:
            actions.append("Notify local Marine Protected Area authority and evaluate non-invasive extraction protocols.")
            reasons.append(f"Proximity to sensitive benthic habitat ({raw.habitat_type or 'Protected Zone'}) warrants urgent ecological mitigation.")

        # 6. Low / Moderate Seabed Residual
        if not actions:
            if risk_level in ["LOW", "MODERATE"]:
                actions.append("Log target in Sea Sentinel Global Ocean Database; maintain periodic acoustic survey monitoring.")
                reasons.append("Seabed object exhibits low immediate navigational and ecological interaction probability.")
            else:
                actions.append("Schedule secondary inspection during next scheduled hydrographic mission pass.")
                reasons.append("Moderate risk profile warrants validation during routine operations.")

        reasoning_text = " ".join(reasons)
        return actions, reasoning_text
