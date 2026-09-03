# SIH26057 — AI-Powered Automated Underwater Marine Debris & Anomaly Detection System

**Organisation:** Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology (NIOT)  
**Problem Statement ID:** SIH26057  
**Category:** Software | **Theme:** Renewable / Sustainable Energy & Blue Economy (SDG 14: Life Below Water)

---

## 1. Project Overview

Side-Scan Sonar (SSS), towed behind research vessels or mounted on Autonomous Underwater Vehicles (AUVs), acoustically maps the seafloor to detect lost fishing gear ("ghost nets"), sunken pipelines/cables, shipwrecks, and hazardous anthropogenic debris. Manual acoustic log inspection across thousands of nautical miles is tedious, slow, and error-prone due to high speckle noise, varying pixel resolutions, and natural acoustic shadows.

This project delivers an end-to-end, production-quality system built around four mandated deliverables:
1. **Object Detection & Semantic Segmentation Core:** Domain-aware candidate extraction (YOLOv11) and pixel-level region refinement (Standard U-Net & Attention U-Net).
2. **Confidence Scoring & Noise Filtering Module:** Acoustic shadow-highlight geometric pairing and Autoencoder-based anomaly filtering to eliminate rock clusters, sand ripples, and speckle noise.
3. **Anomalous Reporting & Geotagging Engine:** Deterministic location determination (Case A: Affine GeoTransform; Case B: Sonar geometry + Navigation log), dimension estimation (length, width, area), and JSON/CSV reporting.
4. **Interactive Dashboard:** Modern UI displaying sonar waterfall overlays, detection tables, Leaflet/Mapbox geospatial mapping, and automated PDF/CSV reports.

---

## 2. Implementation Progress & Roadmap

| Stage | Module | Status | Key Deliverables |
|---|---|---|---|
| **Stage 1** | **Dataset Preparation & Audit** | **COMPLETED** | Non-destructive audit (3,255 Zenodo chips, NOAA GeoTIFFs), inventory JSON, baseline isolation, YOLO split. |
| **Stage 2** | **Sonar Preprocessing Pipeline** | **COMPLETED** | Lee speckle filter ($1.276\times$ ENL gain), CLAHE contrast boost, shadow-highlight extraction, high-res mosaic tiling ($640\times640$). |
| **Stage 3** | **YOLO Debris Detection Core** | **COMPLETED** | YOLOv11 detection engine, training script, mAP evaluation, confusion matrix generator, batch inference CLI. |
| **Stage 4** | **U-Net Semantic Segmentation** | **COMPLETED** | Standard U-Net & Attention U-Net with Attention Gates, BCEDiceLoss, FocalLoss, PatchTiler with cosine blending, dry-run & synthetic demo. |
| **Stage 5** | **Anomaly Detection & False-Positive Filtering** | *Up Next* | CNN Autoencoder reconstruction error (Algorithms 1-9), DBSCAN rock cluster filtering, confidence calibration. |
| **Stage 6** | **AI Agent / Orchestrator** | *Pending* | Traceable execution flow, audit logging, coordinate synthesis, explainability logs. |
| **Stage 7** | **Dimension Estimation & Geotagging** | *Pending* | Case A Affine & Case B navigation math, PyProj WGS84 conversion, metric calculation. |
| **Stage 8** | **Interactive UI Dashboard** | *Pending* | Dual waterfall viewer, Leaflet GIS map, split-view detector/segmenter overlays, report exporter. |
| **Stage 9** | **End-to-End System Verification** | *Pending* | Edge-case verification, FastAPI integration, final documentation. |

---

## 3. Architecture Flow

```
Side-Scan Sonar (SSS) Image / Survey Mosaic
                   │
                   ▼
      [Input Validation & Ingestion]
       ├── Format & corrupt file check
       └── Raster metadata extraction (CRS, Transform, Resolution)
                   │
                   ▼
       [Sonar Preprocessing Engine]
       ├── Normalization & Grayscale standardization
       ├── Speckle noise reduction (Lee filter / adaptive median)
       └── CLAHE contrast enhancement & TVG normalization
                   │
                   ▼
     [Acoustic Shadow-Highlight Pairing]
       ├── Highlight extraction (backscatter peak)
       └── Far-range acoustic shadow trailing verification
                   │
                   ▼
         [YOLO Candidate Detection]
       ├── YOLOv11 inference for candidate region proposals
       └── Bounding box extraction & class confidence
                   │
                   ▼
        [U-Net Region Segmentation]
       ├── Attention U-Net with Oktay et al. Attention Gates
       └── Patch-based sliding window with cosine seam blending
                   │
                   ▼
  [Anomaly Detection & False-Positive Filtering]
       ├── CNN Autoencoder Reconstruction Error (Algorithms 1-9)
       ├── DBSCAN rock cluster suppression
       └── Calibrated confidence scoring (Platt / Temperature scaling)
                   │
                   ▼
  [Geospatial Engine & Dimension Estimation]
       ├── Case A: Direct Affine Matrix Transform
       ├── Case B: Navigation GPS + Slant-to-Ground Range Projection
       ├── Lat/Lon conversion via PyProj (EPSG:4326 WGS84)
       └── Physical length, width, and area metric calculation
                   │
                   ▼
             [Risk Assessment]
       └── Multi-factor risk scoring (debris category, size, confidence)
                   │
                   ▼
          [AI Pipeline Agent]
       ├── Traceable execution audit log
       ├── Structured SQLite database persistence
       └── Structured JSON / CSV report generation
                   │
                   ▼
    [Full-Stack Interactive Dashboard]
       ├── Dual Sonar Waterfall Viewer (Raw vs. Processed)
       ├── Bounding box & mask overlays
       ├── Leaflet / Mapbox interactive GIS map
       └── Downloadable survey inspection reports
```

---

## 4. Quickstart & Verification

### Running the Complete Test Suite

```bash
# 1. Verify all core module boundaries & orchestrator
python tests/test_skeletons.py

# 2. Verify Stage 1 dataset outputs
python tests/test_stage1_dataset.py

# 3. Verify Stage 2 preprocessing pipeline (Lee filter, CLAHE, Tiler)
python tests/test_stage2_preprocessing.py

# 4. Verify Stage 3 YOLO detection core
python tests/test_stage3_yolo.py

# 5. Verify Stage 4 U-Net & Attention U-Net segmentation core
python tests/test_stage4_segmentation.py
```

### Stage 4: U-Net Training & Inference

```bash
# Dry run verification (validates architecture, shapes, gradient flow, audits dataset)
python training/train_unet.py --dry-run

# Run synthetic demonstration training
python training/train_unet.py --synthetic-demo --epochs 3 --batch-size 8

# Run evaluation on test chips
python evaluation/evaluate_unet.py --checkpoint models/checkpoints/unet/attention_unet_best.pt --model attention_unet

# Segment custom sonar imagery
python inference/segment_debris.py --input <path_to_image_or_folder> --checkpoint models/checkpoints/unet/attention_unet_best.pt
```

---

## 5. Development Principles

1. **Strict Data Integrity:** Real input images and rasters are never modified destructively. Missing annotations are honestly reported rather than fabricating synthetic data.
2. **Clear Operational Separation:** REAL MODE (real weights and rasters) vs. DEMO MODE (controlled demonstration clearly identified).
3. **Deterministic Geospatial Math:** Strict implementation of Case A Affine Transformation and Case B Dead-Reckoning Navigation Projection.
4. **Credit-Efficient Step-by-Step Execution:** Systematic modular development with automated regression test coverage at each stage.
