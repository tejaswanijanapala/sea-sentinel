# Sea Sentinel — Interactive Web UI Dashboard

**Frontend for Ministry of Earth Sciences (MoES) / National Institute of Ocean Technology (NIOT)**  
**Problem Statement:** SIH26057 — AI-Powered Automated Underwater Marine Debris & Anomaly Detection

---

## 1. Overview

The `frontend/` application provides an interactive, real-time oceanographic visualization dashboard for hydrographers, marine scientists, and AUV mission operators.

### Key Capabilities:
- **Dual Waterfall Sonar Viewer:** Interactive HTML5 Canvas rendering dual-channel (Port/Starboard) acoustic scans with central Nadir line, grazing-angle correction, and bounding box overlays.
- **Georeferenced Leaflet GIS Map:** CartoDB Dark Matter bathymetric map displaying subsea targets projected to WGS84 (EPSG:4326) with color-coded risk pulse markers.
- **Real-Time Telemetry Cards:** Live metrics for total targets, confirmed debris, suspicious anomalies, suppressed rock fields, and high-risk obstacles.
- **Target Inspector & Explainability:** Displays multi-factor confidence calibration, Autoencoder anomaly MSE, trailing acoustic shadow verification, and natural language action recommendations.
- **Direct Export:** Instant in-browser export of GeoJSON FeatureCollections (for QGIS/ArcGIS) and tabular hydrographic CSVs.

---

## 2. Quickstart: Running Locally

You can serve the frontend with any static web server or Python's built-in HTTP server:

```bash
# Option 1: Python HTTP Server (Zero-install)
python -m http.server 3000 --directory frontend

# Option 2: Node.js npx serve
npx -y serve frontend -p 3000
```

Then open your browser at:
`http://localhost:3000`

---

## 3. Directory Structure

```
frontend/
├── index.html          # Main application dashboard
├── css/
│   └── style.css       # Oceanographic glassmorphism design system
├── js/
│   ├── api.js          # API client connecting to backend FastAPI (/api/...)
│   ├── app.js          # Main UI controller & telemetry manager
│   ├── map.js          # Leaflet GIS bathymetric map component
│   └── waterfall.js    # Side-Scan Sonar dual waterfall canvas renderer
└── README.md           # Documentation
```
