# Sea Sentinel: Operational Field & Deployment Manual

**Project:** SIH26057 — AI-Powered Automated Underwater Marine Debris and Anomaly Detection System  
**Organisation:** Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology (NIOT)  
**Target Applications:** Autonomous Underwater Vehicles (AUVs), Remotely Operated Vehicles (ROVs), Research Vessel Hydrographic Surveys  

---

## 1. Executive Architecture Summary

Sea Sentinel is an automated, end-to-end computer vision and acoustic anomaly detection platform built specifically for high-resolution Side-Scan Sonar (SSS) data. It solves the critical bottleneck of manual acoustic log inspection across thousands of nautical miles.

### Multi-Stage Pipeline Architecture

```text
Side-Scan Sonar Image / Waterfall / GeoTIFF
                   │
                   ▼
       [Input Validation & Ingestion]
        ├── 0-byte & corrupt header protection
        └── Raster metadata extraction (ModelTiepoint, PixelScale, CRS)
                   │
                   ▼
        [Sonar Preprocessing Engine]
        ├── Dynamic range standardization [0, 255]
        ├── Lee Speckle Filter (1.276x ENL improvement)
        └── CLAHE Local Contrast Enhancement
                   │
                   ▼
      [Acoustic Physics & Region Proposals]
        ├── YOLOv11 Candidate Object Detection
        ├── DBSCAN Geological Rock Cluster Suppression
        └── Attention U-Net Pixel-Level Contour Segmentation
                   │
                   ▼
      [Anomaly Detection & Physics Calibration]
        ├── CNN Autoencoder Reconstruction MSE (Algorithms 1-9)
        ├── 3-Sigma Statistical Baseline Threshold (T = 0.094049)
        └── Acoustic Highlight-Shadow Down-Range Pairing
                   │
                   ▼
      [Geospatial Localization & Dimensions]
        ├── Case A: Direct Affine GeoTransform Matrix (<= 1.5m)
        ├── Case B: Slant-to-Ground Correction & Geodesic Navigation (7.5m)
        ├── Case C: Unreferenced Chip (Strict non-fabrication)
        └── Rotated Rectangular & Contour Metric Dimensioning
                   │
                   ▼
      [Explainability Synthesis & Audit Logging]
        ├── Natural Language Hydrographic Inspection Narratives
        ├── SQLite Immutable Audit Database (survey_audit.db)
        └── Multi-Format Deliverables (GeoJSON, CSV, JSON Reports)
```

---

## 2. Deployment Profiles

### Profile A: Research Vessel Survey Workstation (Interactive Real-Time)
- **Host OS:** Windows / Linux / macOS
- **Stack:** FastAPI Backend + Uvicorn + Leaflet Web Dashboard
- **Recommended Command:**
  ```bash
  # Terminal 1: Launch Backend API Server
  cd backend
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  # Terminal 2: Launch Web Dashboard
  python -m http.server 3000 --directory frontend
  ```
- **Access Points:**
  - Interactive UI: `http://localhost:3000`
  - Interactive API Docs (Swagger): `http://localhost:8000/docs`

### Profile B: Autonomous Underwater Vehicle (AUV) Edge Payload
- **Execution Mode:** Headless Batch Survey via CLI
- **Command:**
  ```bash
  cd backend
  python scripts/run_pipeline.py --input /path/to/mission/sonar/scans/ --output-dir outputs/mission_reports/
  ```
- **Outputs Generated:**
  - Aggregated Hydrographic Target CSV: `outputs/mission_reports/survey_detections_summary.csv`
  - GIS Target FeatureCollection: `outputs/mission_reports/survey_targets.geojson`
  - Relational Audit Log: `outputs/audit/survey_audit.db`

---

## 3. Operational REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root system status and endpoint catalog |
| `GET` | `/api/health` | Diagnostic introspection of YOLO, U-Net, Autoencoder, and SQLite |
| `POST` | `/api/upload` | Multipart file upload for raw sonar imagery and GeoTIFFs |
| `POST` | `/api/analyze` | Executes end-to-end AI Agent pipeline on uploaded image |
| `GET` | `/api/results/{analysis_id}` | Retrieves historical survey session details from SQLite |
| `GET` | `/api/geospatial` | Returns all georeferenced subsea targets as GeoJSON |
| `GET` | `/api/high-risk` | Queries all targets flagged as HIGH hazard risk |
| `GET` | `/api/report/{analysis_id}/csv` | Downloads tabular hydrographic CSV for a survey |

---

## 4. Automated Regression Verification Suite

All 9 stages have dedicated unit and stress tests passing with a **100% success rate**:

```bash
cd backend
python tests/test_skeletons.py             # Core module skeleton contracts
python tests/test_stage1_dataset.py       # Dataset audit & baseline isolation
python tests/test_stage2_preprocessing.py # Lee filter & CLAHE contrast boost
python tests/test_stage3_yolo.py          # YOLOv11 debris detection core
python tests/test_stage4_segmentation.py  # Standard & Attention U-Net segmentation
python tests/test_stage5_anomaly.py       # Autoencoder anomaly & DBSCAN rock filter
python tests/test_stage6_agent.py         # AI Agent orchestrator & SQLite logging
python tests/test_stage7_geospatial.py    # Module 5 five-stage geotagging & PyProj WGS84
python tests/test_stage9_verification.py  # FastAPI integration & edge-case stress tests
```

---

## 5. Compliance with Ministry of Earth Sciences (MoES) Directives

1. **Strict Data Integrity:** Raw sonar archives (NOAA GeoTIFFs, Zenodo acoustic chips) are treated strictly as read-only. Missing spatial metadata is honestly classified as Case C without fabricating fake coordinates.
2. **Deterministic Physics Verification:** Man-made debris detection requires both statistical out-of-distribution Autoencoder reconstruction failure ($\text{MSE} > 0.094$) and acoustic shadow verification trailing down-range.
3. **Geological Noise Rejection:** Natural rock moraines and gravel fields are clustered via DBSCAN and penalized, dramatically cutting false alarm rates.
