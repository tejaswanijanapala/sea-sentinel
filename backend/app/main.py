"""
Stage 9: Production FastAPI REST API Server for Sea Sentinel
Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology (NIOT)
Provides high-performance RESTful endpoints for:
  - Acoustic image & GeoTIFF upload
  - End-to-end AI Agent survey orchestration
  - Real-time geospatial target query (GeoJSON / CSV)
  - Historical audit retrieval from SQLite
  - System health diagnostics & model introspection
"""

from typing import Dict, Any, List, Optional
import os
import sys
import uuid
import json
import shutil
import sqlite3
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Ensure backend root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.orchestrator import SIHPipelineAgent
from ai.geospatial.geotagger import GeospatialEngine

app = FastAPI(
    title="Sea Sentinel — AI Underwater Debris & Anomaly Detection API",
    description="MoES / NIOT Autonomous Side-Scan Sonar Debris Detection & Geotagging Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate Core Pipeline Agent & Geospatial Engine
agent = SIHPipelineAgent()
geotagger = GeospatialEngine()

# Ensure Uploads Directory
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "outputs", "uploads")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# -----------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    image_path: str
    raster_meta: Optional[Dict[str, Any]] = None
    nav_log: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------
@app.get("/")
def root():
    """System health, metadata, and operational status."""
    return {
        "system": "Sea Sentinel AI Pipeline",
        "organisation": "Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology (NIOT)",
        "status": "OPERATIONAL",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/api/health",
            "upload": "POST /api/upload",
            "analyze": "POST /api/analyze",
            "results": "/api/results/{analysis_id}",
            "geospatial": "/api/geospatial",
            "high_risk": "/api/high-risk",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
def health_check():
    """Introspects underlying computer vision, anomaly, and geospatial engines."""
    db_ok = os.path.exists(agent.audit_logger.db_path)
    return {
        "status": "healthy",
        "models": {
            "yolo_detector_loaded": agent.detector.is_model_loaded,
            "unet_segmenter_loaded": agent.segmenter.is_model_loaded,
            "autoencoder_loaded": agent.anomaly_detector.is_model_loaded,
            "baseline_threshold": agent.anomaly_detector.threshold
        },
        "geospatial": {
            "target_crs": geotagger.target_crs,
            "pyproj_available": True
        },
        "audit_database": {
            "path": agent.audit_logger.db_path,
            "connected": db_ok
        }
    }


@app.post("/api/upload")
async def upload_sonar_file(file: UploadFile = File(...)):
    """
    Uploads raw sonar imagery, GeoTIFF rasters, or waterfall scans.
    """
    allowed_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{os.path.basename(file.filename)}"
    destination = os.path.join(UPLOADS_DIR, safe_name)

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    val_res = agent.preprocessor.validate_image(destination)
    raster_meta = geotagger.read_raster_metadata(destination)
    georef_case = geotagger.classify_georef_case(raster_meta)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "saved_path": destination,
        "size_bytes": os.path.getsize(destination),
        "valid_image": val_res.get("valid", False),
        "georeferencing_case": georef_case,
        "raster_metadata": raster_meta
    }


@app.post("/api/analyze")
def analyze_survey(req: AnalyzeRequest):
    """
    Executes end-to-end AI Agent survey analysis on the given sonar image.
    """
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at: {req.image_path}")

    res = agent.analyze_image(
        image_path=req.image_path,
        raster_meta_override=req.raster_meta
    )
    return res


@app.get("/api/results/{analysis_id}")
def get_survey_results(analysis_id: str):
    """
    Retrieves historical survey session results from the SQLite audit database.
    """
    summary = agent.audit_logger.get_session_summary(analysis_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Survey session {analysis_id} not found in audit logs.")
    return summary


@app.get("/api/geospatial")
def get_geospatial_targets(limit: int = Query(200, ge=1, le=1000)):
    """
    Returns all georeferenced subsea targets as a standard GeoJSON FeatureCollection.
    """
    conn = sqlite3.connect(agent.audit_logger.db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM target_detections
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            ORDER BY calibrated_confidence DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        targets = [dict(r) for r in rows]
    finally:
        conn.close()

    features = []
    for t in targets:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [t["lon"], t["lat"]]
            },
            "properties": {k: v for k, v in t.items() if k not in ("lat", "lon")}
        })

    return {
        "type": "FeatureCollection",
        "total_targets": len(features),
        "features": features,
        "targets": targets
    }


@app.get("/api/high-risk")
def get_high_risk_targets(limit: int = Query(50, ge=1, le=200)):
    """
    Queries high-priority targets flagged as HIGH hazard risk.
    """
    targets = agent.audit_logger.query_high_risk_targets(limit=limit)
    return {
        "count": len(targets),
        "high_risk_targets": targets
    }


@app.get("/api/report/{analysis_id}/csv")
def download_survey_csv(analysis_id: str):
    """
    Exports and downloads tabular hydrographic CSV for a survey session.
    """
    summary = agent.audit_logger.get_session_summary(analysis_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Survey session {analysis_id} not found.")

    csv_path = os.path.join(REPORTS_DIR, f"{analysis_id}_survey.csv")
    geotagger.export_csv(summary.get("targets", []), csv_path)

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=500, detail="Failed to generate CSV export.")

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"{analysis_id}_hydrographic_report.csv"
    )
