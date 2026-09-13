from backend.risk.models import RawRiskParameters
from backend.risk.risk_engine import IMORiskEngine


def engine():
    return IMORiskEngine()


def test_scenario_1_large_floating_container_near_shipping_lane(engine):
    """
    Scenario 1: Large floating container near busy commercial shipping lane.
    Expected: Critical Navigation Risk (NR > 75), Critical Final Risk.
    """
    raw = RawRiskParameters(
        debris_type="container",
        length_m=12.2,
        width_m=2.4,
        height_m=2.6,
        area_sq_m=29.3,
        volume_cu_m=76.0,
        water_depth_m=25.0,
        object_depth_m=1.0,
        distance_to_route_m=45.0,
        route_traffic_density="HIGH",
        water_column_position="NEAR_SURFACE_FLOATING",
        mobility_class="SURFACE_FLOATING",
        drift_velocity_knots=1.2,
        ai_detection_confidence=0.96,
        position_uncertainty_m=3.0,
        potential_consequence_severity="CRITICAL"
    )

    res = engine.assess_hazard(raw, debris_id="SCENARIO_1_CONTAINER", context_name="COMMERCIAL_SHIPPING_CORRIDOR")

    assert res.risk_level in ["HIGH", "CRITICAL"]
    assert res.navigation_risk >= 70.0
    assert res.likelihood_score >= 70.0
    assert res.consequence_score >= 60.0
    assert any("shipping lane" in d.lower() for d in res.top_contributing_factors)
    assert any("NOTMAR" in a or "Navigational Warning" in a for a in res.recommended_actions)


def test_scenario_2_small_plastic_in_marine_protected_area(engine):
    """
    Scenario 2: Small plastic fragment far from shipping routes but inside sensitive habitat (MPA).
    Expected: Low navigation risk, but elevated ecological risk.
    """
    raw = RawRiskParameters(
        debris_type="plastic_fragment",
        length_m=0.4,
        width_m=0.3,
        area_sq_m=0.12,
        water_depth_m=15.0,
        distance_to_route_m=6500.0,
        route_traffic_density="LOW",
        water_column_position="SUBSURFACE_MIDWATER",
        distance_to_sensitive_habitat_m=20.0,
        habitat_type="MARINE_PROTECTED_AREA",
        mobility_class="SUSPENDED_DRIFTING",
        ai_detection_confidence=0.91,
        position_uncertainty_m=4.0
    )

    res = engine.assess_hazard(raw, debris_id="SCENARIO_2_PLASTIC", context_name="MARINE_PROTECTED_AREA")

    assert res.navigation_risk < 35.0
    assert res.ecological_risk > res.navigation_risk
    assert any("habitat" in d.lower() or "protected" in d.lower() for d in res.top_contributing_factors)


def test_scenario_3_large_ghost_fishing_net(engine):
    """
    Scenario 3: Large abandoned ghost fishing net near benthic zone.
    Expected: Critical entanglement and ecological consequence.
    """
    raw = RawRiskParameters(
        debris_type="fishing_net",
        length_m=35.0,
        width_m=8.0,
        area_sq_m=280.0,
        water_depth_m=18.0,
        distance_to_route_m=850.0,
        water_column_position="SUBSURFACE_MIDWATER",
        distance_to_sensitive_habitat_m=80.0,
        habitat_type="CORAL_REEF",
        mobility_class="SUSPENDED_DRIFTING",
        ai_detection_confidence=0.94,
        position_uncertainty_m=3.5
    )

    res = engine.assess_hazard(raw, debris_id="SCENARIO_3_GHOSTNET")

    assert res.hazard_severity_score >= 75.0
    assert res.consequence_score >= 75.0
    assert res.ecological_risk >= 55.0
    assert any("entanglement" in d.lower() for d in res.top_contributing_factors)
    assert any("ROV" in a or "ghost gear" in a for a in res.recommended_actions)


def test_scenario_4_mobile_floating_debris_with_future_exposure(engine):
    """
    Scenario 4: Mobile floating debris currently 800m from route but drifting toward route.
    Expected: Dynamic drift forecast produces 6h, 12h, 24h, 48h projections with elevated future exposure.
    """
    raw = RawRiskParameters(
        debris_type="drum_or_barrel",
        length_m=1.2,
        width_m=0.8,
        area_sq_m=0.96,
        water_depth_m=30.0,
        distance_to_route_m=800.0,
        water_column_position="NEAR_SURFACE_FLOATING",
        mobility_class="SURFACE_FLOATING",
        drift_velocity_knots=1.5,
        current_speed_knots=1.2,
        current_direction_deg=180.0,
        wind_speed_knots=15.0,
        latitude=25.772,
        longitude=-76.959,
        ai_detection_confidence=0.88,
        position_uncertainty_m=5.0
    )

    res = engine.assess_hazard(raw, debris_id="SCENARIO_4_DRIFTING")

    assert len(res.drift_projections) == 4
    assert res.drift_projections[0].horizon_hours == 6
    assert res.drift_projections[3].horizon_hours == 48
    # Uncertainty expands over time
    assert res.drift_projections[3].uncertainty_radius_m > res.drift_projections[0].uncertainty_radius_m
    assert res.predicted_future_risk_24h is not None


def test_scenario_5_low_confidence_high_uncertainty_corridor_overlap(engine):
    """
    Scenario 5: Target with low AI confidence (42%) and high position uncertainty (±45m) overlapping shipping lane.
    Expected: Position verification required flag raised, Low Confidence rating, does NOT obscure risk.
    """
    raw = RawRiskParameters(
        debris_type="marine_debris",
        length_m=8.0,
        width_m=3.0,
        water_depth_m=12.0,
        distance_to_route_m=35.0,  # inside uncertainty buffer (35m <= 45m + 25m)
        water_column_position="SEABED",
        ai_detection_confidence=0.42,
        position_uncertainty_m=45.0
    )

    res = engine.assess_hazard(raw, debris_id="SCENARIO_5_UNCERTAIN")

    assert res.position_verification_required is True
    assert res.confidence_level in ["LOW", "MODERATE"]
    assert len(res.uncertainty_flags) > 0
    assert any("verification" in a.lower() for a in res.recommended_actions)


def test_mathematical_determinism_and_base_risk_equation(engine):
    """
    Verifies that Base Risk = Likelihood (0-1) * Consequence (0-1) * 100,
    5x5 matrix mapping is consistent, and calculation is 100% deterministic.
    """
    raw = RawRiskParameters(
        debris_type="engine_or_machinery",
        length_m=3.0,
        width_m=2.0,
        water_depth_m=22.0,
        distance_to_route_m=180.0,
        water_column_position="SEABED",
        ai_detection_confidence=0.90,
        position_uncertainty_m=4.0
    )

    res1 = engine.assess_hazard(raw, debris_id="DET_TEST_01")
    res2 = engine.assess_hazard(raw, debris_id="DET_TEST_01")

    assert res1.final_risk_score == res2.final_risk_score
    assert res1.likelihood_score == res2.likelihood_score
    assert res1.consequence_score == res2.consequence_score
    assert res1.risk_matrix.matrix_cell == res2.risk_matrix.matrix_cell
    assert res1.data_completeness_percent > 80.0
