"""
Sea Sentinel: IMO-Aligned Marine Debris Hazard Risk Assessment Engine
Data Models & Schemas

Framework Note:
Sea Sentinel is an IMO-aligned, project-specific marine debris hazard risk assessment framework
based on the principles of IMO Formal Safety Assessment (FSA) (MSC/Circ.1023 / MEPC/Circ.392).
All parameter weights, thresholds, and scoring equations are Sea Sentinel project-specific modeling choices.
"""

from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class RawRiskParameters(BaseModel):
    """The 15 exact project parameters in their native units."""
    # 1. Debris Type
    debris_type: str = Field(default="marine_debris", description="Debris classification (e.g., fishing_net, container, engine, cable, plastic_fragment)")
    # 2. Size / Volume
    length_m: Optional[float] = Field(default=None, description="Length in meters")
    width_m: Optional[float] = Field(default=None, description="Width in meters")
    height_m: Optional[float] = Field(default=None, description="Height in meters")
    area_sq_m: Optional[float] = Field(default=None, description="Surface footprint in m²")
    volume_cu_m: Optional[float] = Field(default=None, description="Estimated volume in m³")
    # 3. Depth
    water_depth_m: Optional[float] = Field(default=None, description="Seabed water depth at detection site in meters")
    object_depth_m: Optional[float] = Field(default=None, description="Depth of the object itself in meters from sea surface")
    # 4. Distance to Navigation Route
    distance_to_route_m: Optional[float] = Field(default=None, description="Distance to nearest commercial shipping lane / fairway in meters")
    route_traffic_density: Optional[str] = Field(default="MODERATE", description="Traffic density: HIGH, MODERATE, LOW, NONE")
    # 5. Water-Column Position
    water_column_position: Literal["SEABED", "SUBSURFACE_MIDWATER", "NEAR_SURFACE_FLOATING"] = Field(
        default="SEABED", description="Vertical position in water column"
    )
    # 6. Entanglement Potential
    entanglement_potential_override: Optional[float] = Field(default=None, description="Manual override 0-100 if specified by expert")
    # 7. Ecological Sensitivity
    distance_to_sensitive_habitat_m: Optional[float] = Field(default=None, description="Distance to nearest coral reef, MPA, seagrass bed in meters")
    habitat_type: Optional[str] = Field(default=None, description="Type of nearest habitat (e.g., CORAL_REEF, MPA, SEAGRASS, NONE)")
    # 8. Persistence
    material_composition: Optional[str] = Field(default=None, description="e.g., SYNTHETIC_POLYMER, FERROUS_METAL, NON_FERROUS, TIMBER, MIXED")
    # 9. Mobility
    mobility_class: Optional[str] = Field(default="STATIONARY_SEABED", description="STATIONARY_SEABED, SLOW_BEDLOAD, SUSPENDED_DRIFTING, SURFACE_FLOATING")
    drift_velocity_knots: Optional[float] = Field(default=0.0, description="Estimated drift velocity in knots")
    current_speed_knots: Optional[float] = Field(default=0.5, description="Local ocean current velocity in knots")
    current_direction_deg: Optional[float] = Field(default=90.0, description="Local current bearing in degrees (0-360)")
    wind_speed_knots: Optional[float] = Field(default=10.0, description="Local surface wind speed in knots")
    wind_direction_deg: Optional[float] = Field(default=90.0, description="Local wind direction in degrees")
    # 10. Debris Density
    local_debris_density_sq_km: Optional[float] = Field(default=None, description="Density of verified debris objects per km² in 1km radius")
    # 11. Recurrence
    detection_count: int = Field(default=1, description="Number of independent surveys where this object was detected")
    first_detected_at: Optional[str] = Field(default=None, description="ISO timestamp of earliest detection")
    latest_detected_at: Optional[str] = Field(default=None, description="ISO timestamp of most recent detection")
    # 12. Detection Confidence
    ai_detection_confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence output from AI detection model")
    sensor_quality_score: float = Field(default=0.90, ge=0.0, le=1.0, description="Acoustic SNR and image quality metric")
    classification_confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Taxonomy classification certainty")
    # 13. Position Uncertainty
    position_uncertainty_m: float = Field(default=5.0, ge=0.0, description="Spatial uncertainty radius (1-sigma) in meters")
    latitude: Optional[float] = Field(default=None, description="WGS84 Latitude")
    longitude: Optional[float] = Field(default=None, description="WGS84 Longitude")
    # 14. Visibility / Detectability
    acoustic_contrast_ratio: Optional[float] = Field(default=0.80, description="Acoustic target-to-background contrast ratio")
    has_acoustic_shadow: bool = Field(default=True, description="Whether an acoustic shadow confirms 3D bathymetric relief")
    # 15. Potential Consequence
    potential_consequence_severity: Optional[str] = Field(default=None, description="Pre-assigned severity rating if known (CRITICAL, HIGH, MODERATE, LOW)")


class NormalizedRiskParameters(BaseModel):
    """All 15 parameters converted into standard continuous 0-100 scale."""
    debris_type_score: float = Field(default=50.0, ge=0.0, le=100.0)
    size_volume_score: float = Field(default=50.0, ge=0.0, le=100.0)
    depth_exposure_score: float = Field(default=50.0, ge=0.0, le=100.0)
    navigation_distance_score: float = Field(default=50.0, ge=0.0, le=100.0)
    water_column_score: float = Field(default=50.0, ge=0.0, le=100.0)
    entanglement_potential_score: float = Field(default=50.0, ge=0.0, le=100.0)
    ecological_sensitivity_score: float = Field(default=50.0, ge=0.0, le=100.0)
    persistence_score: float = Field(default=50.0, ge=0.0, le=100.0)
    mobility_score: float = Field(default=50.0, ge=0.0, le=100.0)
    debris_density_score: float = Field(default=50.0, ge=0.0, le=100.0)
    recurrence_score: float = Field(default=20.0, ge=0.0, le=100.0)
    detection_confidence_score: float = Field(default=85.0, ge=0.0, le=100.0)
    position_uncertainty_score: float = Field(default=80.0, ge=0.0, le=100.0)
    visibility_detectability_score: float = Field(default=75.0, ge=0.0, le=100.0)
    potential_consequence_score: float = Field(default=60.0, ge=0.0, le=100.0)


class LayerASeverityOutput(BaseModel):
    hazard_severity_score: float = Field(ge=0.0, le=100.0, description="Layer A Inherent Hazard Severity Score (0-100)")
    contributions: Dict[str, float] = Field(default_factory=dict)


class LayerBLikelihoodOutput(BaseModel):
    likelihood_score: float = Field(ge=0.0, le=100.0, description="Layer B Likelihood/Exposure Score (0-100)")
    likelihood_normalized: float = Field(ge=0.0, le=1.0, description="Likelihood value L in 0-1 scale for base risk equation")
    interaction_bonus: float = Field(default=0.0, description="Non-linear interaction term (e.g. high mobility x close route)")
    contributions: Dict[str, float] = Field(default_factory=dict)


class LayerCConsequenceOutput(BaseModel):
    consequence_score: float = Field(ge=0.0, le=100.0, description="Layer C Overall Consequence Score (0-100)")
    consequence_normalized: float = Field(ge=0.0, le=1.0, description="Consequence value C in 0-1 scale for base risk equation")
    navigation_consequence: float = Field(ge=0.0, le=100.0)
    ecological_consequence: float = Field(ge=0.0, le=100.0)
    operational_economic_consequence: float = Field(ge=0.0, le=100.0)
    human_safety_consequence: float = Field(ge=0.0, le=100.0)
    context_profile_used: str = Field(default="STANDARD_BALANCED")


class LayerDUncertaintyOutput(BaseModel):
    risk_confidence_score: float = Field(ge=0.0, le=100.0, description="Overall reliability of the risk assessment")
    confidence_level: Literal["HIGH", "MODERATE", "LOW"] = Field(default="HIGH")
    data_completeness_percent: float = Field(ge=0.0, le=100.0, description="Percentage of the 15 parameters provided")
    available_parameters_count: int = Field(ge=0, le=15)
    missing_parameters: List[str] = Field(default_factory=list)
    uncertainty_buffer_radius_m: float = Field(ge=0.0)
    intersects_navigation_route: bool = Field(default=False)
    intersects_sensitive_habitat: bool = Field(default=False)
    intersects_subsea_infrastructure: bool = Field(default=False)
    position_verification_required: bool = Field(default=False)
    uncertainty_flags: List[str] = Field(default_factory=list)


class DriftProjection(BaseModel):
    horizon_hours: int = Field(description="Forecast horizon (6, 12, 24, 48 hours)")
    projected_latitude: float
    projected_longitude: float
    uncertainty_radius_m: float
    future_navigation_exposure: float = Field(ge=0.0, le=100.0)
    future_ecological_exposure: float = Field(ge=0.0, le=100.0)
    intersects_route: bool = Field(default=False)
    intersects_habitat: bool = Field(default=False)


class RiskMatrixPosition(BaseModel):
    likelihood_rank: int = Field(ge=1, le=5, description="1=Rare, 2=Unlikely, 3=Possible, 4=Likely, 5=Almost Certain")
    likelihood_label: str
    consequence_rank: int = Field(ge=1, le=5, description="1=Negligible, 2=Minor, 3=Moderate, 4=Major, 5=Severe")
    consequence_label: str
    matrix_cell: str = Field(description="e.g. 'L4-C5'")
    matrix_risk_category: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]


class RiskAssessmentResult(BaseModel):
    """Complete IMO-Aligned Marine Debris Hazard Assessment Output."""
    debris_id: str
    target_id: Optional[str] = None
    survey_id: Optional[str] = None
    calculated_at: str
    risk_model_version: str = "SS-RISK-v1.0"
    framework_designation: str = "IMO-aligned project-specific formal safety assessment framework"

    # Core 4-Layer Scores
    hazard_severity_score: float = Field(ge=0.0, le=100.0)
    likelihood_score: float = Field(ge=0.0, le=100.0)
    consequence_score: float = Field(ge=0.0, le=100.0)
    risk_confidence: float = Field(ge=0.0, le=100.0)

    # Core Mathematical Equations
    base_risk_score: float = Field(ge=0.0, le=100.0, description="Base Risk = Likelihood (0-1) * Consequence (0-1) * 100")
    final_risk_score: float = Field(ge=0.0, le=100.0, description="Overall risk calibrated by contextual drivers")
    risk_priority_score: float = Field(ge=0.0, le=100.0, description="Multi-factor priority ranking for remediation")
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]

    # 4 Multi-Context Risk Dimensions
    navigation_risk: float = Field(ge=0.0, le=100.0)
    ecological_risk: float = Field(ge=0.0, le=100.0)
    operational_economic_risk: float = Field(ge=0.0, le=100.0)
    human_safety_risk: float = Field(ge=0.0, le=100.0)

    # 5x5 Matrix Representation
    risk_matrix: RiskMatrixPosition

    # Evidence, Completeness & Uncertainty
    data_completeness_percent: float = Field(ge=0.0, le=100.0)
    confidence_level: Literal["HIGH", "MODERATE", "LOW"]
    uncertainty_flags: List[str] = Field(default_factory=list)
    position_verification_required: bool = False

    # Dynamic Mobility / Drift Projections (6h, 12h, 24h, 48h)
    mobility_status: str
    current_risk_score: float
    predicted_future_risk_24h: Optional[float] = None
    drift_projections: List[DriftProjection] = Field(default_factory=list)

    # Explainability & Decision Support
    top_contributing_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    recommendation_reasoning: str = ""

    # Raw and Normalized Parameter Records for Auditing
    raw_parameters: Dict[str, Any] = Field(default_factory=dict)
    normalized_parameters: Dict[str, float] = Field(default_factory=dict)
