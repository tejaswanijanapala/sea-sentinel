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
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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
CACHED_ANALYSES = {}

# Ensure Output and Static Directories
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "outputs", "uploads")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "outputs", "preprocessed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed", "yolo_dataset", "images", "test")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount Static File Routes
app.mount("/static/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/static/preprocessed", StaticFiles(directory=PREPROCESSED_DIR), name="preprocessed")
if os.path.exists(SAMPLES_DIR):
    app.mount("/static/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


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
            "samples": "/api/samples",
            "image": "/api/image?path=...",
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


@app.get("/api/samples")
def get_sample_missions():
    """Returns curated benchmark acoustic sonar scans for immediate 1-click survey analysis."""
    samples = [
        {
            "id": "ghost_net_01",
            "name": "Ghost Fishing Net Mesh",
            "category": "fishing_net",
            "risk_hint": "HIGH",
            "filename": "quanzhou_HN_004.jpg",
            "description": "Dispersed synthetic polymer netting with high acoustic backscatter highlight and acoustic void shadow.",
            "path": os.path.join(SAMPLES_DIR, "quanzhou_HN_004.jpg"),
            "url": "/static/samples/quanzhou_HN_004.jpg",
            "georef_case": "A",
            "simulated_coords": {"lat": 42.747402, "lon": -73.794567}
        },
        {
            "id": "pipeline_cable_01",
            "name": "Subsea Pipeline / Power Cable",
            "category": "pipeline_or_cable",
            "risk_hint": "HIGH",
            "filename": "dongying_POC_017.jpg",
            "description": "Continuous linear acoustic signature with prominent relief shadow across seabed corridor.",
            "path": os.path.join(SAMPLES_DIR, "dongying_POC_017.jpg"),
            "url": "/static/samples/dongying_POC_017.jpg",
            "georef_case": "A",
            "simulated_coords": {"lat": 42.748950, "lon": -73.792840}
        },
        {
            "id": "rock_cluster_01",
            "name": "Natural Seabed Moraine / Riprap",
            "category": "riprap_debris",
            "risk_hint": "LOW",
            "filename": "quanzhou_RP_002.jpg",
            "description": "Dense clustered geological rock formation; filtered and suppressed by DBSCAN spatial clustering.",
            "path": os.path.join(SAMPLES_DIR, "quanzhou_RP_002.jpg"),
            "url": "/static/samples/quanzhou_RP_002.jpg",
            "georef_case": "A",
            "simulated_coords": {"lat": 42.746120, "lon": -73.796100}
        },
        {
            "id": "engine_part_01",
            "name": "Heavy Metallic Engine Debris",
            "category": "engine_part",
            "risk_hint": "HIGH",
            "filename": "dongying_EP_008.jpg",
            "description": "High-density specular acoustic reflector with sharp boundary and distinct acoustic shadow trailing down-range.",
            "path": os.path.join(SAMPLES_DIR, "dongying_EP_008.jpg"),
            "url": "/static/samples/dongying_EP_008.jpg",
            "georef_case": "A",
            "simulated_coords": {"lat": 42.745500, "lon": -73.791500}
        }
    ]
    valid_samples = [s for s in samples if os.path.exists(s["path"])]
    return {
        "status": "success",
        "total_samples": len(valid_samples),
        "samples": valid_samples
    }


@app.get("/api/image")
def get_image_file(path: str = Query(...)):
    """Safely streams image files to the frontend UI."""
    real_path = os.path.abspath(path)
    if not real_path.startswith(PROJECT_ROOT):
        raise HTTPException(status_code=403, detail="Access denied: path outside project root.")
    if not os.path.exists(real_path):
        raise HTTPException(status_code=404, detail="Image file not found.")

    ext = os.path.splitext(real_path)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp"
    }
    return FileResponse(real_path, media_type=media_types.get(ext, "image/jpeg"))


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
        "raster_metadata": raster_meta,
        "image_url": f"/static/uploads/{safe_name}"
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

    # Attach convenient relative URLs for frontend display
    analysis_id = res.get("analysis_id", "")
    res["raw_image_url"] = f"/api/image?path={os.path.abspath(req.image_path)}"
    
    enhanced_p = res.get("enhanced_image_path")
    if enhanced_p and os.path.exists(enhanced_p):
        res["enhanced_image_url"] = f"/static/preprocessed/{os.path.basename(enhanced_p)}"

    annotated_p = res.get("annotated_image_path")
    if annotated_p and os.path.exists(annotated_p):
        res["annotated_image_url"] = f"/static/preprocessed/{os.path.basename(annotated_p)}"

    # Cache for report endpoints
    CACHED_ANALYSES[analysis_id] = res
    CACHED_ANALYSES["latest"] = res

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


@app.get("/api/report/{analysis_id}")
def get_survey_report_data(analysis_id: str):
    """
    Returns structured survey mission report containing image classification, confidence scores,
    priority levels (>75% HIGHER, <=75% LOWER), all candidate classes, coordinates, and physical dimensions.
    """
    data = CACHED_ANALYSES.get(analysis_id)
    if not data:
        if analysis_id == "latest" and CACHED_ANALYSES:
            data = CACHED_ANALYSES.get("latest")
        else:
            summary = agent.audit_logger.get_session_summary(analysis_id)
            if not summary:
                raise HTTPException(status_code=404, detail=f"Survey session {analysis_id} not found.")
            data = summary

    report_summary = data.get("report_summary")
    if not report_summary:
        targets = data.get("detections") or data.get("targets") or []
        best_t = max(targets, key=lambda x: x.get("calibrated_confidence") or x.get("confidence") or 0.0, default={})
        conf = float(best_t.get("calibrated_confidence") or best_t.get("confidence") or 0.0)
        prio = "HIGHER" if conf > 0.75 else "LOWER"
        report_summary = {
            "obtained_image_class": best_t.get("class", "unclassified_debris"),
            "confidence_score": round(conf, 3),
            "confidence_pct": round(conf * 100, 1),
            "priority_level": prio,
            "priority_label": f"{prio} PRIORITY ({'> 75%' if prio == 'HIGHER' else '<= 75%'})",
            "candidate_classes_breakdown": best_t.get("all_detected_classes", []),
            "spatial_location": {
                "latitude": best_t.get("latitude"),
                "longitude": best_t.get("longitude"),
                "total_area_sq_m": best_t.get("area_sq_m"),
                "max_length_m": best_t.get("length_m"),
                "max_width_m": best_t.get("width_m")
            }
        }

    return {
        "analysis_id": analysis_id,
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "report_summary": report_summary,
        "detections": data.get("detections") or data.get("targets") or [],
        "raw_image_url": data.get("raw_image_url"),
        "enhanced_image_url": data.get("enhanced_image_url"),
        "annotated_image_url": data.get("annotated_image_url"),
        "georeferencing_case": data.get("georeferencing_case", "A")
    }


@app.get("/api/report/{analysis_id}/html", response_class=HTMLResponse)
def get_survey_report_html(analysis_id: str):
    """
    Renders a publication-ready, printable Hydrographic Survey Mission Report.
    """
    report_data = get_survey_report_data(analysis_id)
    rep = report_data.get("report_summary", {})
    prio_level = rep.get("priority_level", "LOWER")
    prio_color = "#ef4444" if prio_level == "HIGHER" else "#0284c7"
    prio_badge = f'<span style="background: {prio_color}; color: #ffffff; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; text-transform: uppercase;">▲ HIGHER PRIORITY (&gt; 75%)</span>' if prio_level == "HIGHER" else f'<span style="background: {prio_color}; color: #ffffff; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; text-transform: uppercase;">▼ LOWER PRIORITY (≤ 75%)</span>'

    spatial = rep.get("spatial_location", {})
    lat = spatial.get("latitude")
    lon = spatial.get("longitude")
    has_coords = lat is not None and lon is not None
    lat_str = f"{lat:.6f}° N" if lat is not None else "Unreferenced (Case C)"
    lon_str = f"{abs(lon):.6f}° {'W' if lon and lon < 0 else 'E'}" if lon is not None else "Unreferenced"
    len_m = spatial.get("max_length_m") or "Estimated"
    wid_m = spatial.get("max_width_m") or "Estimated"
    area_m = spatial.get("total_area_sq_m") or "Estimated"

    raw_img = report_data.get("raw_image_url") or "/static/uploads/default.png"
    annot_img = report_data.get("annotated_image_url") or report_data.get("enhanced_image_url") or raw_img

    # Candidates table
    candidates_html = ""
    for c in rep.get("candidate_classes_breakdown", []):
        c_prio = c.get("priority_level", "LOWER")
        c_badge = '<span style="color: #ef4444; font-weight: 700;">HIGHER (&gt;75%)</span>' if c_prio == "HIGHER" else '<span style="color: #64748b; font-weight: 600;">LOWER (≤75%)</span>'
        candidates_html += f"""
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 10px 12px; font-weight: 600; text-transform: capitalize;">{c.get('class', '').replace('_', ' ')}</td>
          <td style="padding: 10px 12px; font-family: monospace; font-size: 1rem;">{c.get('confidence_pct', 0)}%</td>
          <td style="padding: 10px 12px;">{c_badge}</td>
        </tr>
        """

    # Target table
    targets_html = ""
    for idx, t in enumerate(report_data.get("detections", [])):
        t_conf = round(float(t.get("calibrated_confidence") or t.get("confidence") or 0.0) * 100, 1)
        t_prio = "HIGHER" if t_conf > 75.0 else "LOWER"
        t_prio_badge = '<span style="color: #ef4444; font-weight: 700;">HIGHER</span>' if t_prio == "HIGHER" else '<span style="color: #64748b; font-weight: 600;">LOWER</span>'
        t_coords = f"{t.get('latitude', 0):.5f}, {t.get('longitude', 0):.5f}" if t.get('latitude') else "Unreferenced"
        t_dims = f"{t.get('length_m', '-')}m × {t.get('width_m', '-')}m"
        targets_html += f"""
        <tr style="border-bottom: 1px solid #e2e8f0;">
          <td style="padding: 8px 10px; font-weight: 600; font-family: monospace;">{t.get('object_id', f'TGT_{idx+1:03d}')}</td>
          <td style="padding: 8px 10px; text-transform: capitalize;">{t.get('class', '').replace('_', ' ')}</td>
          <td style="padding: 8px 10px; font-family: monospace;">{t_conf}%</td>
          <td style="padding: 8px 10px;">{t_prio_badge}</td>
          <td style="padding: 8px 10px; font-family: monospace; font-size: 0.85rem;">{t_coords}</td>
          <td style="padding: 8px 10px; font-size: 0.85rem;">{t_dims}</td>
          <td style="padding: 8px 10px; font-size: 0.85rem;">{t.get('anomaly_status', 'evaluated')}</td>
        </tr>
        """

    map_lat = lat if has_coords else 42.7474
    map_lon = lon if has_coords else -73.7945

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Hydrographic Mission Report — {analysis_id}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 30px;
      background: #f8fafc;
      color: #0f172a;
      line-height: 1.5;
    }}
    .report-card {{
      max-width: 1040px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 36px 44px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 2px solid #0f172a;
      padding-bottom: 18px;
      margin-bottom: 26px;
    }}
    .header-sub {{
      font-size: 0.78rem;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      font-weight: 700;
      color: #0369a1;
      margin-bottom: 4px;
    }}
    .header-title {{
      font-size: 1.6rem;
      font-weight: 800;
      color: #0f172a;
      margin: 0;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 16px 20px;
    }}
    .section-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: #0f172a;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 8px;
      margin-top: 24px;
      margin-bottom: 16px;
    }}
    .img-box {{
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      text-align: center;
    }}
    .img-box img {{
      max-width: 100%;
      height: auto;
      max-height: 280px;
      display: block;
      margin: 0 auto;
    }}
    .img-label {{
      background: #0f172a;
      color: #ffffff;
      font-size: 0.75rem;
      padding: 6px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }}
    #map {{
      height: 320px;
      width: 100%;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th {{
      background: #0f172a;
      color: #ffffff;
      padding: 10px;
      text-align: left;
      font-weight: 600;
      font-size: 0.8rem;
      letter-spacing: 0.5px;
    }}
    .btn-print {{
      background: #0f172a;
      color: #ffffff;
      border: none;
      padding: 10px 20px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
    }}
    .btn-print:hover {{ background: #1e293b; }}
    @media print {{
      body {{ background: #fff; padding: 0; }}
      .report-card {{ border: none; box-shadow: none; padding: 0; }}
      .no-print {{ display: none !important; }}
    }}
  </style>
</head>
<body>

  <div class="report-card">
    <div class="header">
      <div>
        <div class="header-sub">Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology (NIOT)</div>
        <h1 class="header-title">Autonomous Hydrographic Survey Mission Report</h1>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">
          Mission ID: <b>{analysis_id}</b> | Generated: <b>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</b>
        </div>
      </div>
      <div class="no-print">
        <button class="btn-print" onclick="window.print()">🖨️ Print / Save PDF</button>
      </div>
    </div>

    <!-- Rule Callout -->
    <div style="background: #e0f2fe; border-left: 4px solid #0284c7; padding: 12px 16px; border-radius: 4px; margin-bottom: 22px; font-size: 0.88rem;">
      <b>Evaluation Priority Standard:</b> Target confidence score <b>&gt; 75.0%</b> is categorized as <b>HIGHER PRIORITY</b> (Immediate intervention/inspection); confidence score <b>≤ 75.0%</b> is categorized as <b>LOWER PRIORITY</b> (Passive seabed monitoring).
    </div>

    <!-- Obtained Classification & Priority Banner -->
    <div class="grid-2">
      <div class="stat-card" style="border-left: 5px solid {prio_color};">
        <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #64748b; margin-bottom: 4px;">Obtained Primary Image Class</div>
        <div style="font-size: 1.6rem; font-weight: 800; text-transform: capitalize; color: #0f172a;">{rep.get('obtained_image_class', 'N/A').replace('_', ' ')}</div>
        <div style="margin-top: 8px; display: flex; align-items: center; gap: 12px;">
          <span style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">Confidence: {rep.get('confidence_pct', 0)}%</span>
          {prio_badge}
        </div>
      </div>

      <div class="stat-card">
        <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #64748b; margin-bottom: 4px;">Geospatial Survey Location & Dimensions</div>
        <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 3px;">
          <b>Latitude:</b> <span style="font-family: monospace;">{lat_str}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 6px;">
          <b>Longitude:</b> <span style="font-family: monospace;">{lon_str}</span>
        </div>
        <div style="font-size: 0.85rem; color: #475569;">
          <b>Physical Dimensions:</b> {len_m}m (Length) × {wid_m}m (Width) | <b>Area:</b> {area_m} m²
        </div>
      </div>
    </div>

    <!-- Multi-Class Candidate Confidence Breakdown -->
    <div class="section-title">Detected Classes & Confidence Distribution</div>
    <table>
      <thead>
        <tr>
          <th>Detected Debris Class</th>
          <th>Confidence Score</th>
          <th>Operational Priority Threshold</th>
        </tr>
      </thead>
      <tbody>
        {candidates_html}
      </tbody>
    </table>

    <!-- Sonar Imagery Section -->
    <div class="section-title">Sonar Imagery Analysis (Input vs Processed)</div>
    <div class="grid-2">
      <div class="img-box">
        <div class="img-label">INPUT RAW ACOUSTIC SONAR SCAN</div>
        <img src="{raw_img}" alt="Input Sonar Scan" />
      </div>
      <div class="img-box">
        <div class="img-label">AI PROCESSED & ANNOTATED DETECTIONS</div>
        <img src="{annot_img}" alt="Annotated Sonar Scan" />
      </div>
    </div>

    <!-- Location Map Section -->
    <div class="section-title">Georeferenced Survey Location Map (WGS84)</div>
    <div id="map"></div>
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
      Georeferencing Datum: <b>WGS84 (EPSG:4326)</b> | Survey Coordinates: <b>{lat_str}, {lon_str}</b> | Basemap: <b>ESRI World Dark Canvas</b>
    </div>

    <!-- Target Inventory Table -->
    <div class="section-title">Comprehensive Target Inventory ({len(report_data.get('detections', []))} Objects)</div>
    <table>
      <thead>
        <tr>
          <th>Target ID</th>
          <th>Class</th>
          <th>Confidence</th>
          <th>Priority</th>
          <th>WGS84 Coordinates</th>
          <th>Dimensions</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {targets_html}
      </tbody>
    </table>

    <div style="margin-top: 36px; padding-top: 16px; border-top: 1px solid #cbd5e1; font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between;">
      <span>Sea Sentinel Hydrographic Platform — NIOT / MoES Verification Audit</span>
      <span>Classification Standard: Confidence &gt; 75% = HIGHER PRIORITY</span>
    </div>
  </div>

  <script>
    const map = L.map('map', {{ center: [{map_lat}, {map_lon}], zoom: 14, zoomControl: true }});
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      attribution: '&copy; Esri &mdash; NIOT Sea Sentinel'
    }}).addTo(map);

    const marker = L.marker([{map_lat}, {map_lon}]).addTo(map);
    marker.bindPopup("<b>{rep.get('obtained_image_class', 'Debris Target').replace('_', ' ').title()}</b><br>Confidence: {rep.get('confidence_pct', 0)}%<br>Priority: {prio_level} PRIORITY<br>Coords: {lat_str}, {lon_str}").openPopup();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
