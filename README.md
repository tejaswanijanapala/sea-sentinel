# SIH26057 — AI-Powered Automated Underwater Marine Debris & Anomaly Detection System

**Organisation:** Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology (NIOT)  
**Problem Statement ID:** SIH26057  
**Category:** Software | **Theme:** Renewable / Sustainable Energy & Blue Economy (SDG 14: Life Below Water)

---

## 1. Project Overview

Side-Scan Sonar (SSS), towed behind research vessels or mounted on Autonomous Underwater Vehicles (AUVs), acoustically maps the seafloor to detect lost fishing gear ("ghost nets"), sunken pipelines/cables, shipwrecks, and hazardous anthropogenic debris. Manual acoustic log inspection across thousands of nautical miles is tedious, slow, and error-prone due to high speckle noise, varying pixel resolutions, and natural acoustic shadows.

This project delivers a production-quality prototype system built around four mandated deliverables:
1. **Object Detection & Semantic Segmentation Core:** Domain-aware candidate extraction (YOLO) and pixel-level region refinement (U-Net).
2. **Confidence Scoring & Noise Filtering Module:** Acoustic shadow-highlight geometric pairing and Autoencoder-based anomaly filtering to eliminate rock clusters, sand ripples, and speckle noise.
3. **Anomalous Reporting & Geotagging Engine:** Deterministic location determination (Case A: Affine GeoTransform; Case B: Sonar geometry + Navigation log), dimension estimation (length, width, area), and JSON/CSV reporting.
4. **Interactive Dashboard:** Modern UI displaying sonar waterfall overlays, detection tables, Leaflet/Mapbox geospatial mapping, and automated PDF/CSV reports.

---

## 2. Six-Layer Architecture

```
Raw SSS Survey / XTF / GeoTIFF
             │
             ▼
[Layer 1: Data Ingestion & Tiling]
  ├── Multi-format reader (GeoTIFF, VRT, STAC, XTF, PNG Waterfall)
  └── Slant-range & ground-range correction
             │
             ▼
[Layer 2: Preprocessing Pipeline]
  ├── Speckle noise reduction (Lee filter / adaptive median)
  ├── Contrast enhancement (CLAHE / histogram equalization)
  └── Highlight & acoustic shadow extraction
             │
             ▼
[Layer 3: AI Detection & Segmentation]
  ├── Fast candidate detection (YOLOv11/v8)
  └── Pixel-level mask refinement (U-Net)
             │
             ▼
[Layer 4: Confidence Calibration & Anomaly Filtering]
  ├── CNN Autoencoder reconstruction error (Algorithms 1-9)
  ├── Shadow-highlight geometric consistency check
  └── Natural rock field suppression (DBSCAN + texture)
             │
             ▼
[Layer 5: Geotagging & Dimension Engine (Module 5)]
  ├── Stage 1: Dataset & Metadata Ingestion
  ├── Stage 2: Georef Case Classification (Case A / Case B)
  ├── Stage 3: Deterministic Location Determination
  ├── Stage 4: WGS84 Lat/Lon Conversion (PyProj EPSG:4326)
  └── Stage 5: Metric Dimensions & Report Assembly
             │
             ▼
[Layer 6: AI Orchestrator, Storage & Dashboard]
  ├── AI Agent Orchestration (Traceable, audit-logged)
  ├── SQLite / PostGIS Detection Database
  └── Full-Stack Interactive Web Dashboard
```

---

## 3. Dataset Audit & Inventory

| Dataset / Source | Format & Scale | Annotations / Ground Truth | Best Suited Component |
|---|---|---|---|
| **China Offshore SSS-AI (Zenodo)** | 3,255 image chips (Dongying, Yantai, Quanzhou, Shenzhen) | Harmonized image-level class labels (`HN`: Fishing Net, `POC`: Pipeline, `RP`: Riprap, `SS`: Seabed, etc.) | Target Classification, Autoencoder Anomaly Baseline (`SS` normal seabed), Stage 3 YOLO proposal chips |
| **NOAA H11584** | 2 GeoTIFF mosaics (~1.2 GB, 1m/px resolution, 445 kHz) | Embedded CRS & GeoTransform | Layer 1 Ingestion, Layer 2 Preprocessing & Tiling, Layer 5 Case A Geotagging & Dimension Validation |
| **Hudson River 2009 (EPSG:26918)** | 4 GeoTIFFs + VRT mosaic + STAC Catalog (1m/px) | Full NAD83 UTM 18N georeferencing | Geospatial Case A Georeferencing, Tiling & Map Overlay |
| **NOAA Boston Harbor (DH_NOAA)** | 1 GeoTIFF + World file (`.tfw`) + XML metadata | UTM Zone 19N, 1m pixel scale | Case A Geotagging & Metric Dimension Benchmarking |
| **Monrovia IVER SSS** | High-resolution waterfall PNG (2.89 MB) | Raw port/starboard acoustic waterfall + nadir zone | Case B Sonar Geometry Reconstruction & Shadow-Highlight Pair Modeling |

---

## 4. Development Rules & Integrity Principles

1. **No Fabricated Data or Metrics:** All models, coordinates, and metrics are verified against real inputs. If segmentation masks are absent, the training pipeline is implemented and documented rather than inventing fake masks.
2. **Clear Operational Separation:** REAL MODE (real model weights & georeferenced rasters) vs. DEMO MODE (interactive preview with clear labeling).
3. **Independent Testability:** Every module has isolated unit/smoke tests with zero hidden circular dependencies.
4. **Credit-Efficient Step-by-Step Execution:** One development stage at a time, strictly awaiting user instruction before advancing.

---

## 5. Verification & Testing

To test the current module skeletons:
```bash
python tests/test_skeletons.py
```
Output:
```
Testing Preprocessor...
Testing YOLODetector...
Testing UNetSegmenter...
Testing DimensionEstimator...
Testing GeospatialEngine Case A...
Testing Agent Orchestrator...
All module skeletons verified successfully!
```
