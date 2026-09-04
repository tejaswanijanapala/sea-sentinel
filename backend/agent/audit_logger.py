"""
Stage 6: Survey Audit Logger and SQLite Persistence Engine
Maintains an immutable, queryable audit trail of all sonar survey processing sessions,
target detections, model decisions, geospatial locations, and execution metrics.
"""

from typing import Dict, Any, List, Optional
import os
import sqlite3
import json
import time
from datetime import datetime


class SurveyAuditLogger:
    """
    Manages SQLite database logging and JSON report archiving for hydrographic audit compliance.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "audit")
            os.makedirs(db_dir, exist_ok=True)
            self.db_path = os.path.join(db_dir, "survey_audit.db")
        else:
            self.db_path = db_path
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes relational tables for audit compliance."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 1. Survey Sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS survey_sessions (
                    session_id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    georef_case TEXT,
                    total_candidates INTEGER,
                    confirmed_count INTEGER,
                    suspicious_count INTEGER,
                    rejected_count INTEGER,
                    yolo_loaded BOOLEAN,
                    unet_loaded BOOLEAN,
                    autoencoder_loaded BOOLEAN,
                    execution_time_ms REAL
                )
            """)

            # 2. Target Detections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS target_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    raw_confidence REAL,
                    calibrated_confidence REAL,
                    status TEXT,
                    risk_score TEXT,
                    lat REAL,
                    lon REAL,
                    length_m REAL,
                    width_m REAL,
                    area_sq_m REAL,
                    reconstruction_error REAL,
                    shadow_verified BOOLEAN,
                    is_rock_cluster BOOLEAN,
                    executive_narrative TEXT,
                    FOREIGN KEY (session_id) REFERENCES survey_sessions(session_id)
                )
            """)

            # 3. Execution Trace Events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES survey_sessions(session_id)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def log_survey(
        self,
        session_id: str,
        image_path: str,
        georef_case: str,
        execution_trace: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        yolo_loaded: bool,
        unet_loaded: bool,
        autoencoder_loaded: bool,
        total_duration_ms: float
    ) -> bool:
        """
        Atomically records a complete survey analysis session.
        """
        confirmed = sum(1 for d in detections if d.get("anomaly_status") == "confirmed_debris" or d.get("status") == "confirmed_debris")
        suspicious = sum(1 for d in detections if d.get("anomaly_status") == "suspicious_anomaly" or d.get("status") == "suspicious_anomaly")
        rejected = sum(1 for d in detections if d.get("anomaly_status") == "noise_rejected" or d.get("status") == "noise_rejected")

        timestamp_str = datetime.utcnow().isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Insert Session
            cursor.execute("""
                INSERT INTO survey_sessions (
                    session_id, image_path, timestamp, georef_case,
                    total_candidates, confirmed_count, suspicious_count, rejected_count,
                    yolo_loaded, unet_loaded, autoencoder_loaded, execution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, image_path, timestamp_str, georef_case,
                len(detections), confirmed, suspicious, rejected,
                yolo_loaded, unet_loaded, autoencoder_loaded, total_duration_ms
            ))

            # Insert Detections
            for d in detections:
                cursor.execute("""
                    INSERT INTO target_detections (
                        session_id, object_id, class_name, raw_confidence,
                        calibrated_confidence, status, risk_score, lat, lon,
                        length_m, width_m, area_sq_m, reconstruction_error,
                        shadow_verified, is_rock_cluster, executive_narrative
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    d.get("object_id", "OBJ_UNKNOWN"),
                    d.get("class", "unknown"),
                    d.get("confidence", 0.0),
                    d.get("calibrated_confidence", d.get("confidence", 0.0)),
                    d.get("anomaly_status", d.get("status", "unknown")),
                    d.get("risk_score", "LOW"),
                    d.get("lat"),
                    d.get("lon"),
                    d.get("length_m"),
                    d.get("width_m"),
                    d.get("area_sq_m"),
                    d.get("reconstruction_error", 0.0),
                    d.get("shadow_verified", True),
                    d.get("is_rock_cluster", False),
                    d.get("explanation", {}).get("executive_narrative", "") if isinstance(d.get("explanation"), dict) else str(d.get("explanation", ""))
                ))

            # Insert Trace Events
            for ev in execution_trace:
                meta = {k: v for k, v in ev.items() if k not in ("stage", "status", "duration_ms")}
                cursor.execute("""
                    INSERT INTO execution_trace_events (
                        session_id, stage_name, status, duration_ms, metadata_json
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    ev.get("stage", "unknown"),
                    ev.get("status", "completed"),
                    ev.get("duration_ms", 0.0),
                    json.dumps(meta)
                ))

            conn.commit()
            return True
        except Exception as e:
            print(f"[SurveyAuditLogger] Error writing audit log: {e}")
            return False
        finally:
            conn.close()

    def query_high_risk_targets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves high-risk confirmed debris targets across all past surveys.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM target_detections
                WHERE risk_score = 'HIGH'
                ORDER BY calibrated_confidence DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches full session details including targets and execution trace.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM survey_sessions WHERE session_id = ?", (session_id,))
            session = cursor.fetchone()
            if not session:
                return None

            cursor.execute("SELECT * FROM target_detections WHERE session_id = ?", (session_id,))
            targets = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT * FROM execution_trace_events WHERE session_id = ?", (session_id,))
            traces = [dict(r) for r in cursor.fetchall()]

            res = dict(session)
            res["targets"] = targets
            res["execution_trace"] = traces
            return res
        finally:
            conn.close()
