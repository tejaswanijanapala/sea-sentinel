"""
Sea Sentinel: IMO-Aligned Risk Assessment REST API Endpoints
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
from backend.risk.models import RawRiskParameters, RiskAssessmentResult
from backend.risk.risk_engine import IMORiskEngine
from backend.risk.audit_service import RiskAuditService
from backend.risk.config import (
    RISK_MODEL_VERSION,
    FRAMEWORK_DESIGNATION,
    RISK_THRESHOLDS,
    HAZARD_SEVERITY_WEIGHTS,
    LIKELIHOOD_WEIGHTS,
    CONSEQUENCE_CONTEXT_PROFILES
)
from backend.database.local_db import LocalDatabase

router = APIRouter(prefix="/api/risk", tags=["IMO-Aligned Risk Assessment Engine"])

_db = LocalDatabase()
_engine = IMORiskEngine()
_audit_service = RiskAuditService(_db.get_connection)


@router.post("/calculate", response_model=RiskAssessmentResult)
def calculate_risk(
    params: RawRiskParameters = Body(...),
    debris_id: str = Query(default="TGT_001", description="Identifier of the target debris object"),
    survey_id: Optional[str] = Query(default=None, description="Associated mission survey ID"),
    context_name: str = Query(default="STANDARD_BALANCED", description="Operational context profile")
):
    """
    Executes an authoritative 4-Layer IMO-Aligned Marine Debris Hazard Assessment:
    - Normalizes the 15 project parameters
    - Computes Layer A (Hazard Severity), Layer B (Likelihood), Layer C (Consequence), Layer D (Uncertainty)
    - Computes Multi-Context Risk Dimensions (Navigation, Ecological, Operational, Human-Safety)
    - Maps to 5x5 Maritime Decision Matrix (L1-5 x C1-5)
    - Projects 6h, 12h, 24h, 48h dynamic drift cones for mobile debris
    - Issues explainable risk-control recommendations
    - Persists audit log
    """
    try:
        result = _engine.assess_hazard(
            raw=params,
            debris_id=debris_id,
            survey_id=survey_id,
            context_name=context_name
        )
        # Save audit record asynchronously / locally
        _audit_service.save_risk_assessment(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")


@router.get("/config")
def get_risk_config():
    """Returns active risk engine configuration, weights, thresholds, and version."""
    return {
        "risk_model_version": RISK_MODEL_VERSION,
        "framework_designation": FRAMEWORK_DESIGNATION,
        "risk_thresholds": RISK_THRESHOLDS,
        "hazard_severity_weights": HAZARD_SEVERITY_WEIGHTS,
        "likelihood_weights": LIKELIHOOD_WEIGHTS,
        "consequence_context_profiles": CONSEQUENCE_CONTEXT_PROFILES
    }


@router.get("/history/{debris_id}")
def get_risk_history(debris_id: str):
    """Retrieves temporal risk assessment records for a specific debris target."""
    history = _audit_service.get_assessment_history(debris_id)
    return {
        "debris_id": debris_id,
        "total_assessments": len(history),
        "history": history
    }


@router.get("/drift/{debris_id}")
def get_drift_forecast(
    debris_id: str,
    lat: float = Query(...),
    lon: float = Query(...),
    mobility_class: str = Query(default="SUSPENDED_DRIFTING"),
    current_knots: float = Query(default=0.8),
    current_bearing_deg: float = Query(default=90.0),
    wind_knots: float = Query(default=12.0),
    wind_bearing_deg: float = Query(default=90.0)
):
    """Computes dynamic drift forecast trajectory for a target."""
    raw = RawRiskParameters(
        debris_type="marine_debris",
        latitude=lat,
        longitude=lon,
        mobility_class=mobility_class,
        current_speed_knots=current_knots,
        current_direction_deg=current_bearing_deg,
        wind_speed_knots=wind_knots,
        wind_direction_deg=wind_bearing_deg
    )
    projections = _engine.drift_predictor.forecast_drift(raw, current_lat=lat, current_lon=lon)
    return {
        "debris_id": debris_id,
        "origin_coordinates": [lat, lon],
        "mobility_class": mobility_class,
        "projections": projections
    }
