"""
Sea Sentinel: Risk Assessment Audit & History Service
Persists complete, reproducible audit trails of every risk assessment.
"""

import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
from .models import RiskAssessmentResult


class RiskAuditService:
    """
    Manages immutable audit logging for risk assessments, model versioning,
    and temporal recalculation tracking.
    """

    def __init__(self, db_connection_factory):
        self.get_connection = db_connection_factory

    def save_risk_assessment(self, result: RiskAssessmentResult) -> bool:
        """Saves a complete risk assessment record with raw parameters and weights."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    debris_id TEXT,
                    survey_id TEXT,
                    calculated_at TEXT,
                    risk_model_version TEXT,
                    hazard_severity_score REAL,
                    likelihood_score REAL,
                    consequence_score REAL,
                    risk_confidence REAL,
                    base_risk_score REAL,
                    final_risk_score REAL,
                    risk_priority_score REAL,
                    risk_level TEXT,
                    navigation_risk REAL,
                    ecological_risk REAL,
                    operational_economic_risk REAL,
                    human_safety_risk REAL,
                    matrix_cell TEXT,
                    data_completeness_percent REAL,
                    position_verification_required INTEGER,
                    top_contributing_factors_json TEXT,
                    recommended_actions_json TEXT,
                    raw_parameters_json TEXT,
                    normalized_parameters_json TEXT,
                    drift_projections_json TEXT
                );
                """)

                conn.execute("""
                INSERT OR REPLACE INTO risk_assessments (
                    assessment_id, debris_id, survey_id, calculated_at, risk_model_version,
                    hazard_severity_score, likelihood_score, consequence_score, risk_confidence,
                    base_risk_score, final_risk_score, risk_priority_score, risk_level,
                    navigation_risk, ecological_risk, operational_economic_risk, human_safety_risk,
                    matrix_cell, data_completeness_percent, position_verification_required,
                    top_contributing_factors_json, recommended_actions_json,
                    raw_parameters_json, normalized_parameters_json, drift_projections_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    f"RISK_{result.debris_id}_{int(datetime.utcnow().timestamp())}",
                    result.debris_id,
                    result.survey_id,
                    result.calculated_at,
                    result.risk_model_version,
                    result.hazard_severity_score,
                    result.likelihood_score,
                    result.consequence_score,
                    result.risk_confidence,
                    result.base_risk_score,
                    result.final_risk_score,
                    result.risk_priority_score,
                    result.risk_level,
                    result.navigation_risk,
                    result.ecological_risk,
                    result.operational_economic_risk,
                    result.human_safety_risk,
                    result.risk_matrix.matrix_cell,
                    result.data_completeness_percent,
                    1 if result.position_verification_required else 0,
                    json.dumps(result.top_contributing_factors),
                    json.dumps(result.recommended_actions),
                    json.dumps(result.raw_parameters),
                    json.dumps(result.normalized_parameters),
                    json.dumps([p.dict() for p in result.drift_projections])
                ))
            return True
        except Exception as e:
            print(f"[RiskAuditService] Warning saving risk assessment: {e}")
            return False

    def get_assessment_history(self, debris_id: str) -> List[Dict[str, Any]]:
        """Retrieves temporal history of risk calculations for a given debris object."""
        try:
            with self.get_connection() as conn:
                cur = conn.execute("""
                SELECT * FROM risk_assessments WHERE debris_id = ? ORDER BY calculated_at DESC;
                """, (debris_id,))
                rows = cur.fetchall()
                results = []
                for r in rows:
                    item = dict(r)
                    if item.get("top_contributing_factors_json"):
                        item["top_contributing_factors"] = json.loads(item["top_contributing_factors_json"])
                    if item.get("recommended_actions_json"):
                        item["recommended_actions"] = json.loads(item["recommended_actions_json"])
                    results.append(item)
                return results
        except Exception:
            return []
