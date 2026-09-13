"""
Sea Sentinel: Master IMO-Aligned Marine Debris Hazard Risk Assessment Engine
Orchestrates the 4-layer risk analysis pipeline, 5x5 matrix mapping,
multi-context breakdown, dynamic drift forecasting, and decision support recommendations.

Compliance & Methodology Note:
Sea Sentinel is an IMO-aligned, project-specific marine debris hazard risk assessment framework
based on the principles of IMO Formal Safety Assessment (FSA) (MSC/Circ.1023 / MEPC/Circ.392).
All parameter weights, thresholds, and scoring equations are Sea Sentinel project-specific modeling choices.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .models import (
    RawRiskParameters,
    NormalizedRiskParameters,
    RiskAssessmentResult,
    RiskMatrixPosition
)
from .config import (
    RISK_MODEL_VERSION,
    FRAMEWORK_DESIGNATION,
    RISK_THRESHOLDS,
    MULTI_CONTEXT_RISK_WEIGHTS
)
from .normalizer import RiskParameterNormalizer
from .hazard_severity import HazardSeverityModel
from .likelihood_model import LikelihoodExposureModel
from .consequence_model import ConsequenceModel
from .uncertainty_model import UncertaintyConfidenceModel
from .drift_predictor import DriftPredictor
from .recommendation_engine import RiskRecommendationEngine


class IMORiskEngine:
    """
    Authoritative Marine Debris Hazard Risk Assessment Engine for Sea Sentinel.
    Computes rigorous, explainable, and multi-dimensional risk intelligence.
    """

    def __init__(self):
        self.normalizer = RiskParameterNormalizer()
        self.severity_model = HazardSeverityModel()
        self.likelihood_model = LikelihoodExposureModel()
        self.consequence_model = ConsequenceModel()
        self.uncertainty_model = UncertaintyConfidenceModel()
        self.drift_predictor = DriftPredictor()
        self.recommendation_engine = RiskRecommendationEngine()

    @staticmethod
    def map_to_5x5_matrix(likelihood_score: float, consequence_score: float) -> RiskMatrixPosition:
        """
        Maps continuous scores (0-100) to standard 5x5 Maritime Risk Matrix:
        Likelihood (1=Rare, 2=Unlikely, 3=Possible, 4=Likely, 5=Almost Certain)
        Consequence (1=Negligible, 2=Minor, 3=Moderate, 4=Major, 5=Severe)
        """
        # Likelihood Rank
        if likelihood_score < 20.0:
            l_rank, l_lbl = 1, "Rare"
        elif likelihood_score < 40.0:
            l_rank, l_lbl = 2, "Unlikely"
        elif likelihood_score < 60.0:
            l_rank, l_lbl = 3, "Possible"
        elif likelihood_score < 80.0:
            l_rank, l_lbl = 4, "Likely"
        else:
            l_rank, l_lbl = 5, "Almost Certain"

        # Consequence Rank
        if consequence_score < 20.0:
            c_rank, c_lbl = 1, "Negligible"
        elif consequence_score < 40.0:
            c_rank, c_lbl = 2, "Minor"
        elif consequence_score < 60.0:
            c_rank, c_lbl = 3, "Moderate"
        elif consequence_score < 80.0:
            c_rank, c_lbl = 4, "Major"
        else:
            c_rank, c_lbl = 5, "Severe"

        product = l_rank * c_rank
        if product >= 16 or (l_rank >= 4 and c_rank >= 4):
            cat = "CRITICAL"
        elif product >= 9 or (c_rank >= 4):
            cat = "HIGH"
        elif product >= 4:
            cat = "MODERATE"
        else:
            cat = "LOW"

        return RiskMatrixPosition(
            likelihood_rank=l_rank,
            likelihood_label=l_lbl,
            consequence_rank=c_rank,
            consequence_label=c_lbl,
            matrix_cell=f"L{l_rank}-C{c_rank}",
            matrix_risk_category=cat
        )

    @staticmethod
    def extract_top_contributing_factors(
        raw: RawRiskParameters,
        norm: NormalizedRiskParameters,
        layer_b: Any,
        layer_c: Any
    ) -> List[str]:
        """Identifies and articulates the top 3-5 drivers of the computed risk."""
        drivers = []

        # 1. Navigation Route Proximity
        if norm.navigation_distance_score >= 70.0:
            dist_str = f"{raw.distance_to_route_m:.0f}m" if raw.distance_to_route_m is not None else "Immediate"
            density_str = f" ({raw.route_traffic_density or 'High'} traffic corridor)"
            drivers.append(f"Proximity to commercial shipping lane: {dist_str} distance{density_str}.")

        # 2. Entanglement Morphology
        if norm.entanglement_potential_score >= 75.0:
            drivers.append(f"High entanglement morphology ({raw.debris_type.replace('_', ' ')}): Propeller jamming and ghost fishing hazard.")

        # 3. Water Column & Surface Flotation
        if norm.water_column_score >= 80.0:
            drivers.append("Near-surface / floating water column position: Direct vessel collision and propulsion contact profile.")

        # 4. Physical Dimensions / Area
        if norm.size_volume_score >= 70.0:
            sz_desc = f"{raw.length_m}m × {raw.width_m}m" if raw.length_m and raw.width_m else "Large extent"
            drivers.append(f"Substantial physical dimensions ({sz_desc}): Elevated acoustic and structural impact footprint.")

        # 5. Dynamic Mobility / Drifting
        if norm.mobility_score >= 65.0:
            drivers.append(f"Dynamic mobility ({raw.mobility_class}): Active hydrodynamic displacement and expanding exposure zone.")

        # 6. Sensitive Habitat Proximity
        if norm.ecological_sensitivity_score >= 70.0:
            hab_str = raw.habitat_type.replace("_", " ") if raw.habitat_type else "Protected Ecological Area"
            dist_str = f"{raw.distance_to_sensitive_habitat_m:.0f}m" if raw.distance_to_sensitive_habitat_m is not None else "<500m"
            drivers.append(f"Proximity to sensitive marine habitat ({hab_str}): Within {dist_str} of sensitive benthic zone.")

        # 7. Recurrence Persistence
        if norm.recurrence_score >= 60.0:
            drivers.append(f"Chronic recurrence ({raw.detection_count} independent survey detections): Persistent unaddressed seabed obstacle.")

        # Fallback if few triggered
        if len(drivers) < 2:
            drivers.append(f"Acoustic target classification: {raw.debris_type.replace('_', ' ').title()}.")
            drivers.append(f"Acoustic backscatter detectability: {norm.visibility_detectability_score:.0f}/100.")

        return drivers[:5]

    @staticmethod
    def compute_risk_priority_score(
        final_risk: float,
        consequence: float,
        hazard_severity: float,
        confidence: float,
        recurrence_score: float,
        mobility_score: float,
        verification_required: bool
    ) -> float:
        """
        Answers: 'Which debris should be investigated first?'
        Combines Final Risk (45%), Consequence Severity (25%), Inherent Hazard Severity (15%), Confidence/Evidence (10%), and Recurrence/Mobility (5%).
        """
        priority = (
            (final_risk * 0.45) +
            (consequence * 0.25) +
            (hazard_severity * 0.15) +
            (confidence * 0.10) +
            (max(recurrence_score, mobility_score) * 0.05)
        )
        return float(max(10.0, min(100.0, round(priority, 1))))

    def assess_hazard(
        self,
        raw: RawRiskParameters,
        debris_id: str = "TGT_001",
        survey_id: Optional[str] = None,
        context_name: str = "STANDARD_BALANCED"
    ) -> RiskAssessmentResult:
        """
        Executes end-to-end multi-layer IMO-aligned risk assessment.
        """
        # Step 1: Normalize all 15 raw metrics
        norm = self.normalizer.normalize_all(raw)

        # Step 2: Layer A — Inherent Hazard Severity
        layer_a = self.severity_model.compute_severity(norm)

        # Step 3: Layer B — Likelihood / Exposure
        layer_b = self.likelihood_model.compute_likelihood(norm)

        # Step 4: Layer C — Multi-Domain Consequence
        layer_c = self.consequence_model.compute_consequence(norm, context_name=context_name)

        # Step 5: Layer D — Uncertainty & Spatial Buffer Intersections
        layer_d = self.uncertainty_model.evaluate_uncertainty(raw, norm)

        # Step 6: Base Risk Equation: Risk = Likelihood (0-1) * Consequence (0-1) * 100
        l_val = layer_b.likelihood_normalized
        c_val = layer_c.consequence_normalized
        base_risk = (l_val * c_val) * 100.0

        # Contextual Calibration (Incorporates Inherent Hazard Severity HSS into final risk rating)
        # Final Risk = Base Risk (55%) + Hazard Severity (45%)
        final_risk = (base_risk * 0.55) + (layer_a.hazard_severity_score * 0.45)
        final_risk = float(max(0.0, min(100.0, round(final_risk, 1))))

        # Determine Categorical Risk Level
        if final_risk >= RISK_THRESHOLDS["CRITICAL"]["min"]:
            risk_level = "CRITICAL"
        elif final_risk >= RISK_THRESHOLDS["HIGH"]["min"]:
            risk_level = "HIGH"
        elif final_risk >= RISK_THRESHOLDS["MODERATE"]["min"]:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Step 7: Multi-Context Risk Dimensions (NR, ER, OR, HSR)
        # Combines domain-specific consequence with exposure profile
        nav_risk = float(max(10.0, min(100.0, round(layer_c.navigation_consequence * 0.65 + layer_b.likelihood_score * 0.35, 1))))
        eco_risk = float(max(10.0, min(100.0, round(layer_c.ecological_consequence * 0.65 + layer_b.likelihood_score * 0.35, 1))))
        ops_risk = float(max(10.0, min(100.0, round(layer_c.operational_economic_consequence * 0.65 + layer_b.likelihood_score * 0.35, 1))))
        safe_risk = float(max(10.0, min(100.0, round(layer_c.human_safety_consequence * 0.65 + layer_b.likelihood_score * 0.35, 1))))

        # Step 8: 5x5 Maritime Risk Matrix Position
        matrix_pos = self.map_to_5x5_matrix(layer_b.likelihood_score, layer_c.consequence_score)

        # Step 9: Dynamic Drift Projections
        drift_projections = self.drift_predictor.forecast_drift(raw)
        pred_future_risk_24h = None
        if drift_projections:
            # Check 24h projection exposure
            proj_24 = next((p for p in drift_projections if p.horizon_hours == 24), None)
            if proj_24:
                fut_l = proj_24.future_navigation_exposure / 100.0
                pred_future_risk_24h = round(float(max(final_risk, (fut_l * c_val) * 100.0)), 1)

        # Step 10: Risk Priority Score
        priority_score = self.compute_risk_priority_score(
            final_risk=final_risk,
            consequence=layer_c.consequence_score,
            hazard_severity=layer_a.hazard_severity_score,
            confidence=layer_d.risk_confidence_score,
            recurrence_score=norm.recurrence_score,
            mobility_score=norm.mobility_score,
            verification_required=layer_d.position_verification_required
        )

        # Step 11: Top Contributing Drivers
        top_drivers = self.extract_top_contributing_factors(raw, norm, layer_b, layer_c)

        # Step 12: Actionable Risk Control Recommendations
        actions, reasoning = self.recommendation_engine.generate_recommendations(
            raw=raw,
            norm=norm,
            layer_a=layer_a,
            layer_b=layer_b,
            layer_c=layer_c,
            layer_d=layer_d,
            final_risk_score=final_risk,
            risk_level=risk_level
        )

        return RiskAssessmentResult(
            debris_id=debris_id,
            target_id=raw.debris_type,
            survey_id=survey_id or "SURVEY_SESSION",
            calculated_at=datetime.utcnow().isoformat() + "Z",
            risk_model_version=RISK_MODEL_VERSION,
            framework_designation=FRAMEWORK_DESIGNATION,
            hazard_severity_score=layer_a.hazard_severity_score,
            likelihood_score=layer_b.likelihood_score,
            consequence_score=layer_c.consequence_score,
            risk_confidence=layer_d.risk_confidence_score,
            base_risk_score=round(base_risk, 1),
            final_risk_score=final_risk,
            risk_priority_score=priority_score,
            risk_level=risk_level,
            navigation_risk=nav_risk,
            ecological_risk=eco_risk,
            operational_economic_risk=ops_risk,
            human_safety_risk=safe_risk,
            risk_matrix=matrix_pos,
            data_completeness_percent=layer_d.data_completeness_percent,
            confidence_level=layer_d.confidence_level,
            uncertainty_flags=layer_d.uncertainty_flags,
            position_verification_required=layer_d.position_verification_required,
            mobility_status=raw.mobility_class or "STATIONARY_SEABED",
            current_risk_score=final_risk,
            predicted_future_risk_24h=pred_future_risk_24h,
            drift_projections=drift_projections,
            top_contributing_factors=top_drivers,
            recommended_actions=actions,
            recommendation_reasoning=reasoning,
            raw_parameters=raw.dict(),
            normalized_parameters=norm.dict()
        )
