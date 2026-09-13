"""
Sea Sentinel: Layer D — Uncertainty & Confidence Model
Quantifies evidence reliability, spatial error boundaries, data completeness,
and corridor intersection flags (POSITION VERIFICATION REQUIRED).
"""

from typing import Dict, Any, List, Optional, Tuple
from .models import RawRiskParameters, NormalizedRiskParameters, LayerDUncertaintyOutput


class UncertaintyConfidenceModel:
    """
    Evaluates evidence quality and spatial uncertainty:
    1. Computes Risk Confidence (RC, 0-100) combining AI confidence, sensor quality, and data completeness.
    2. Measures Data Completeness (x/15 parameters).
    3. Performs spatial buffer intersection against critical marine zones.
    4. Issues 'POSITION VERIFICATION REQUIRED' flag when uncertainty region overlaps a critical zone.
    """

    ALL_15_PARAM_KEYS = [
        "debris_type",
        "size_volume",
        "water_depth",
        "distance_to_navigation_route",
        "water_column_position",
        "entanglement_potential",
        "distance_to_sensitive_habitat",
        "material_persistence",
        "mobility_class",
        "local_debris_density",
        "detection_recurrence",
        "ai_detection_confidence",
        "position_uncertainty",
        "visibility_detectability",
        "potential_consequence"
    ]

    def evaluate_data_completeness(self, raw: RawRiskParameters) -> Tuple[float, int, List[str]]:
        """Checks which of the 15 project parameters are available."""
        missing = []
        available_count = 0

        # 1. Type
        if raw.debris_type and raw.debris_type != "unknown":
            available_count += 1
        else:
            missing.append("debris_type")

        # 2. Size
        if raw.length_m is not None or raw.area_sq_m is not None or raw.volume_cu_m is not None:
            available_count += 1
        else:
            missing.append("size_volume")

        # 3. Depth
        if raw.water_depth_m is not None or raw.object_depth_m is not None:
            available_count += 1
        else:
            missing.append("water_depth")

        # 4. Route distance
        if raw.distance_to_route_m is not None:
            available_count += 1
        else:
            missing.append("distance_to_navigation_route")

        # 5. Water column
        if raw.water_column_position is not None:
            available_count += 1
        else:
            missing.append("water_column_position")

        # 6. Entanglement
        available_count += 1  # Inferred from type or override

        # 7. Habitat distance
        if raw.distance_to_sensitive_habitat_m is not None or raw.habitat_type is not None:
            available_count += 1
        else:
            missing.append("distance_to_sensitive_habitat")

        # 8. Persistence
        available_count += 1  # Inferred from material/type

        # 9. Mobility
        if raw.mobility_class is not None:
            available_count += 1
        else:
            missing.append("mobility_class")

        # 10. Density
        if raw.local_debris_density_sq_km is not None:
            available_count += 1
        else:
            missing.append("local_debris_density")

        # 11. Recurrence
        if raw.detection_count is not None:
            available_count += 1
        else:
            missing.append("detection_recurrence")

        # 12. Detection confidence
        if raw.ai_detection_confidence is not None:
            available_count += 1
        else:
            missing.append("ai_detection_confidence")

        # 13. Position uncertainty
        if raw.position_uncertainty_m is not None:
            available_count += 1
        else:
            missing.append("position_uncertainty")

        # 14. Visibility
        if raw.acoustic_contrast_ratio is not None:
            available_count += 1
        else:
            missing.append("visibility_detectability")

        # 15. Potential consequence
        available_count += 1  # Inferred from morphological profile

        pct = (available_count / 15.0) * 100.0
        return round(pct, 1), available_count, missing

    def evaluate_uncertainty(
        self,
        raw: RawRiskParameters,
        norm: NormalizedRiskParameters
    ) -> LayerDUncertaintyOutput:
        completeness_pct, avail_cnt, missing_params = self.evaluate_data_completeness(raw)

        # 1. Risk Confidence Decomposition:
        # AI Conf (35%) + Sensor Quality (25%) + Position Accuracy (20%) + Data Completeness (20%)
        ai_part = float(raw.ai_detection_confidence or 0.85) * 100.0 * 0.35
        sensor_part = float(raw.sensor_quality_score or 0.90) * 100.0 * 0.25
        pos_part = norm.position_uncertainty_score * 0.20
        comp_part = completeness_pct * 0.20

        risk_conf = ai_part + sensor_part + pos_part + comp_part
        risk_conf_clamped = float(max(10.0, min(100.0, risk_conf)))

        # Categorical Level
        if risk_conf_clamped >= 75.0:
            conf_level = "HIGH"
        elif risk_conf_clamped >= 45.0:
            conf_level = "MODERATE"
        else:
            conf_level = "LOW"

        # 2. Spatial Uncertainty Buffer Intersection Engine
        unc_radius = max(0.5, float(raw.position_uncertainty_m or 5.0))
        nav_dist = float(raw.distance_to_route_m) if raw.distance_to_route_m is not None else 9999.0
        eco_dist = float(raw.distance_to_sensitive_habitat_m) if raw.distance_to_sensitive_habitat_m is not None else 9999.0

        # Buffer overlaps corridor if center distance <= uncertainty radius + nominal buffer (25m)
        intersects_route = (nav_dist <= (unc_radius + 25.0))
        intersects_habitat = (eco_dist <= (unc_radius + 50.0))
        intersects_infra = False

        flags = []
        pos_verification_req = False

        if intersects_route:
            flags.append(f"CRITICAL: Spatial uncertainty buffer (±{unc_radius:.1f}m) intersects commercial shipping corridor (distance {nav_dist:.1f}m).")
            pos_verification_req = True

        if intersects_habitat:
            flags.append(f"ECOLOGICAL WARNING: Uncertainty boundary (±{unc_radius:.1f}m) overlaps sensitive marine habitat boundary (distance {eco_dist:.1f}m).")
            pos_verification_req = True

        if unc_radius >= 30.0:
            flags.append(f"HIGH POSITIONAL UNCERTAINTY: Positional error ±{unc_radius:.1f}m exceeds hydrographic precision threshold.")
            pos_verification_req = True

        if raw.ai_detection_confidence < 0.60:
            flags.append(f"LOW AI CONFIDENCE: Detection model certainty {raw.ai_detection_confidence*100:.1f}% warrants field verification.")

        if completeness_pct < 65.0:
            flags.append(f"INCOMPLETE METADATA: Assessment based on {avail_cnt}/15 parameters ({completeness_pct:.0f}% completeness).")

        return LayerDUncertaintyOutput(
            risk_confidence_score=round(risk_conf_clamped, 1),
            confidence_level=conf_level,
            data_completeness_percent=completeness_pct,
            available_parameters_count=avail_cnt,
            missing_parameters=missing_params,
            uncertainty_buffer_radius_m=round(unc_radius, 2),
            intersects_navigation_route=intersects_route,
            intersects_sensitive_habitat=intersects_habitat,
            intersects_subsea_infrastructure=intersects_infra,
            position_verification_required=pos_verification_req,
            uncertainty_flags=flags
        )
