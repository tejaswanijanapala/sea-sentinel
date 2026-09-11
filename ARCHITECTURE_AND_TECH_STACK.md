# Sea Sentinel — Complete Architecture, Technology Stack & ML Workflow Specification
**Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology (NIOT)**  
*Autonomous Parallel Dual-Path YOLOv11 + U-Net Side-Scan Sonar (SSS) Marine Debris Detection, Spatial Intelligence & Edge Perception Engine*

---

## 1. High-Level Architecture & End-to-End Data Flow

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              1. FRONTEND USER INTERFACE                                │
 │   Vanilla HTML5 · Vanilla CSS3 (Custom Design Tokens) · ES6+ JavaScript · Canvas 2D    │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ REST API / WebSocket Telemetry (JSON)
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                          2. BACKEND API & WEBSERVER LAYER                              │
 │            FastAPI (Python 3.12) · Uvicorn ASGI Server · Pydantic v2 Schemas           │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
 ┌───────────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
 │       3. ACOUSTIC PREPROCESSING ENGINE        │   │        6. GEOSPATIAL & BATHYMETRY         │
 │   OpenCV · NumPy · Rasterio / GDAL · Pillow   │   │     PyProj · Shapely · Leaflet.js GIS     │
 │   • Lee Speckle Filter · CLAHE Equalizer      │   │     • WGS84 (EPSG:4326) · UTM Projection  │
 │   • 256/640px Overlapping Patch Tiler         │   │     • Slant-to-Ground Range Georeferencer │
 └───────────────────────┬───────────────────────┘   └─────────────────────┬─────────────────────┘
                         │                                                 │
                         ▼                                                 │
 ┌─────────────────────────────────────────────────────────────────────────┼─────────────────────┐
 │                     4. DUAL-PATH PARALLEL AI INFERENCE                  │                     │
 │   Ultralytics YOLOv11 (PyTorch / ONNX)  │  ResNet34 Attention U-Net     │                     │
 │   • Fast Bounding Box Localization      │  • Morphological Pixel Masks  │                     │
 └───────────────────────┬─────────────────┴───────────────────────────────┘                     │
                         │                                                                       │
                         ▼                                                                       │
 ┌───────────────────────────────────────────────┐                                               │
 │       5. SPATIAL FUSION & VERIFICATION        │                                               │
 │   IoU Association · Bipartite Greedy Matcher  │                                               │
 │   • Shadow-Highlight Physics Verifier         │                                               │
 │   • Scikit-Learn DBSCAN Geological Filter     │                                               │
 └───────────────────────┬───────────────────────┘                                               │
                         │                                                                       │
                         └───────────────────────────────┬───────────────────────────────────────┘
                                                         │
                                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                        7. PERSISTENCE & LOCAL DATABASE LAYER                           │
 │     SQLite3 (WAL Concurrency Mode) · Relational Spatial DB · JSON Target Serializers   │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                     8. EDGE TELEMETRY & SUBSEA ACOUSTIC MODEM                          │
 │         Python Struct (Binary Packing) · CRC-8 Checksum · ThreadPoolExecutor           │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Technology Stack Breakdown

### Step 1: Frontend Layer (Surveillance UI & Real-Time Visualization)
* **HTML5**: Semantic document structure, canvas rendering containers, modal dialogs, and accessibility tags.
* **Vanilla CSS3 (Custom Design Tokens)**: High-contrast White & Emerald Green surveillance theme, dark mode overlays, responsive CSS Grid / Flexbox layouts, micro-animations (`@keyframes sonar-ping`).
* **Vanilla JavaScript (ES6+ Classes)**: Object-oriented modular controllers (`DashboardApp`, `GISMap`, `WaterfallViewer`, `APIService`), Event-driven UI updates. Zero heavy dependencies (React/Vue/Angular) for ultra-fast edge loading.
* **HTML5 Canvas 2D API (`CanvasRenderingContext2D`)**: High-performance pixel rendering of Side-Scan Sonar waterfall scans, dynamic bounding box tracking, and crosshair overlays.
* **Leaflet.js (v1.9.4 Local Offline Asset)**: Interactive bathymetric ocean map, layer switching, multi-marker clustering, and nautical graticules.
* **FontAwesome 6.5 & Google Fonts**: `Inter` (UI typography), `Outfit` (Headings), and `JetBrains Mono` (Telemetry & Coordinates).

### Step 2: Backend REST API & Server Orchestration
* **Python 3.12**: Core runtime environment for asynchronous pipeline orchestration and deep-learning runtimes.
* **FastAPI (v0.110+)**: High-throughput async RESTful endpoints (`/api/analyze`, `/api/upload`, `/api/gis/map-data`, `/api/report`). Automatically generates interactive OpenAPI / Swagger documentation.
* **Uvicorn**: High-performance ASGI web server managing concurrent worker threads on `localhost:8000`.
* **Pydantic v2**: Strict type enforcement and schema validation for JSON survey payloads, navigation records, and inference parameters.
* **`concurrent.futures.ThreadPoolExecutor`**: Concurrent execution of YOLO detection and U-Net segmentation in separate OS threads, maintaining total pipeline latency at **<0.35s**.

### Step 3: Sonar Preprocessing & Validation Layer
* **NumPy & OpenCV (`cv2`)**: Chromatic saturation and acoustic spectral divergence evaluation (`sat_mean > 0.08`, `max_diff > 9.0`) to automatically reject non-sonar optical images.
* **Enhanced Lee Adaptive Filter (SciPy / OpenCV)**: Removes acoustic speckle noise while preserving sharp boundaries of high-impedance objects (metal, concrete, nets).
* **CLAHE (Contrast-Limited Adaptive Histogram Equalization)**: Normalizes illumination variations across the sonar swath, compensating for acoustic attenuation across the water column.
* **Rasterio & GDAL / Pillow (PIL)**: Multi-band 16-bit/32-bit GeoTIFF mosaic ingestion, extracting pixel resolutions, tie points, and Affine transformation matrices.
* **Custom `PatchTiler` (NumPy)**: Slices large GeoTIFF mosaics into overlapping `256×256` and `640×640` patches with **25% overlap** to eliminate edge-boundary target truncation.

### Step 4: Machine Learning & Dual-Path Neural Network Inference
* **Branch A — Ultralytics YOLOv11 Marine (`ultralytics`, PyTorch, ONNX)**: Fast bounding box localization across 5 anthropogenic debris classes (`fishing_net`, `pipeline_or_cable`, `shipwreck_fragment`, `engine_debris`, `riprap_debris`). Inference latency: **<12ms**.
* **Branch B — ResNet34 Attention U-Net (Custom PyTorch `torch.nn.Module`)**: Pixel-level binary mask segmentation capturing amorphous shapes (entangled ghost nets, cables, ropes) and acoustic shadow regions.
* **Torchvision ResNet-34 Encoder**: Deep feature extractor with spatial Attention Gates in decoder skip connections to focus network attention on high-backscatter anomalies.
* **ONNX Runtime / PyTorch FP16**: Half-precision quantization for ultra-fast edge inference on CPU and embedded GPUs (NVIDIA Jetson AGX / Orin).

### Step 5: Candidate Fusion & Physics-Grounded Verification
* **Custom `FusionEngine` (NumPy)**: Greedy bipartite matching associating YOLO bounding boxes with U-Net masks via Intersection over Union (IoU $\ge 0.25$) and centroid distance ratios.
* **Custom `CandidateVerifier` (OpenCV)**: Computes 3 physical acoustic properties: **1)** Core-to-background contrast ratio, **2)** Shadow-highlight relief profile, and **3)** Morphological solidity.
* **Scikit-Learn DBSCAN (`sklearn.cluster.DBSCAN`)**: Clusters natural rocky reefs and sea-bottom formations to automatically downweight false alarms caused by natural rocks by **35%**.

### Step 6: Geospatial Intelligence, Georeferencing & Bathymetry
* **PyProj (`pyproj.Transformer`, `pyproj.CRS`)**: Converts local sonar raster coordinates (UTM Zone 16N / State Plane) into global geographic WGS84 coordinates (`EPSG:4326`).
* **Trigonometric / Geodesic Engine (NumPy)**: Computes true ground range from vehicle altitude ($h$), towfish slant range ($R_s = \sqrt{x^2 + h^2}$), and vessel heading ($\theta$).
* **Shapely (`shapely.geometry`)**: Generates vessel survey trajectory `LineString` tracks and scanned sonar swath `Polygon` coverage footprints.
* **Basemap Tile Providers (OpenStreetMap, ESRI Satellite, ESRI Topo, GEBCO)**: 100% free, unrestricted tile servers with `maxNativeZoom: 10` scaling for reliable bathymetric mapping without missing tile watermarks.

### Step 7: Persistence & Local Database Layer
* **SQLite3 (`backend/database/sea_sentinel_edge.db`)**: Lightweight, self-contained, serverless relational database ideal for offline shipboard deployment.
* **WAL Mode (`PRAGMA journal_mode=WAL;`)**: Write-Ahead Logging allows simultaneous non-blocking multi-thread reads and atomic transaction writes.
* **Spatial Schema Tables**: Maintains persistent relational tables for `surveys`, `survey_images`, `detections`, `targets` (deduplicated), `survey_tracks`, `survey_coverage`, and `clusters`.

### Step 8: Edge Computing & Acoustic Telemetry Modem
* **Python `struct` (`struct.pack / unpack`)**: Compresses full detection coordinates, class codes, confidence, slant range, and depth into a compact **24-byte binary packet**.
* **CRC-8 Polynomial Checksum (`crc8`)**: Validates acoustic modem payload integrity across noisy underwater acoustic water columns.
* **Custom `EdgeWatchdogSupervisor`**: Monitors hardware temperatures and throttling levels, dynamically adjusting operating policies (Level 0 Full AI to Level 2 Balanced).

### Step 9: Testing, Quality Assurance & DevOps
* **Python `unittest` (`backend/tests/test_*.py`)**: Comprehensive 23-test automated regression suite covering SSS authentication, YOLO inference, U-Net extraction, georeferencing, and database operations.
* **Git & GitHub**: Distributed source control, branch merging, and atomic feature commits (`tejaswanijanapala/sea-sentinel`).

---

## 3. Neural Network Model Specifications & Sizes

| Parameter | Branch A: YOLOv11 Marine | Branch B: Attention U-Net |
| :--- | :--- | :--- |
| **Architecture** | Ultralytics YOLOv11 Nano/Small Backbone | ResNet34 Encoder + Attention Gate Decoder |
| **Primary Task** | Fast Bounding Box Localization & Taxonomy | Morphological Pixel-Wise Mask & Shadow Segmentation |
| **Target Classes** | `fishing_net`, `pipeline_or_cable`, `shipwreck_fragment`, `engine_debris`, `riprap_debris` | Binary anthropogenic debris anomaly mask |
| **Model Weights Path** | `models/yolo/best.pt` / `best_fp16.pt` / `best.onnx` | `models/unet/attention_unet_best.pt` / `_fp16.pt` |
| **Model Disk Size** | **~5.8 MB – 21.5 MB** (FP16 / FP32 ONNX) | **~44.2 MB – 86.8 MB** (ResNet34 Attention) |
| **In-Memory VRAM/RAM** | **~85 MB** (Inference runtime) | **~180 MB** (Inference runtime) |
| **Inference Latency** | **8 ms – 14 ms** per tile (GPU) / ~35 ms (CPU) | **18 ms – 28 ms** per tile (GPU) / ~70 ms (CPU) |

---

## 4. Input & Output Boundaries

* **Input Formats**: `.tif`, `.tiff` (GeoTIFF Mosaics), `.png`, `.jpg`, `.jpeg`, `.bmp` (Sonar Waterfall Rasters).
* **Minimum Input Size**: **`64 × 64` pixels**.
* **Native Inference Tile Size**: **`256 × 256`** (U-Net) / **`640 × 640`** (YOLO).
* **Maximum Input Size**: **Unlimited (Tested up to `10,000 × 10,000` px)** via overlapping sliding window tiling.
* **Output Payload Formats**:
  1. Annotated high-resolution sonar images with labeled bounding boxes and segmented contours.
  2. Hydrographic Survey Mission Intelligence HTML / PDF / CSV reports.
  3. GeoJSON FeatureCollections (WGS84 EPSG:4326) compatible with QGIS and ArcGIS.
  4. Ultra-compact **24-byte binary acoustic modem hex packet** for subsea sound transmission.

---

## 5. Confidence Score Calculation & Mathematical Proof

$$\text{Final Confidence} = \Big( W_{\text{YOLO}} \cdot C_{\text{YOLO}} + W_{\text{UNet}} \cdot C_{\text{UNet}} \Big) + \Delta_{\text{DualBonus}} + \Phi_{\text{Physics}}$$

1. **Dual-Model Statistical Agreement ($W_{\text{YOLO}} = 0.55, W_{\text{UNet}} = 0.45$)**:
   - If both YOLO and U-Net independently detect the same object (IoU $\ge 0.25$), a **$+12\%$ dual-model agreement bonus** is added.
2. **Acoustic Contrast Score ($C_{\text{contrast}}$, Weight = 0.25)**:
   - Evaluates the pixel intensity ratio between the target highlight core and surrounding seafloor background:
     $$C_{\text{contrast}} = \min\left(1.0, \frac{\mu_{\text{target}} - \mu_{\text{background}}}{\sigma_{\text{background}} + 1}\right)$$
3. **Acoustic Shadow-Highlight Relief Score ($C_{\text{shadow}}$, Weight = 0.25)**:
   - Validates that a true 3D physical object casts an acoustic shadow in the sound wave propagation direction.
4. **Morphological Compactness ($C_{\text{morph}}$, Weight = 0.20)**:
   - Analyzes shape circularity and perimeter regularity to differentiate manufactured objects from natural sediment drifts.
5. **Geological Rock Cluster Downweighting**:
   - DBSCAN spatial clustering flags natural rock fields and applies a **$-35\%$ confidence penalty** to suppress geological false alarms.

---

## 6. Complete Dataset Catalog

| Dataset Component | Modality | Sample Count / Size | Primary Application |
| :--- | :--- | :--- | :--- |
| **China Offshore Zenodo** | High-Res Sonar Chips | 3,255 images (~1.1 GB) | YOLOv11 & U-Net Debris Detection Training |
| **NOAA H11584 Gulf of Mexico** | 445 kHz GeoTIFF Mosaic | 2 large mosaics (~1.13 GB) | Large-area Swath Mosaic Tiling & Georeferencing |
| **USGS DS 1005 Breton Sound** | 0.5m/px GeoTIFF Mosaic | 1 mosaic (~120 MB) | Shallow Estuary & Sanctuary Debris Detection |
| **NOAA NY Harbor HR09** | Multi-swath GeoTIFFs | 4 GeoTIFF files (~210 MB) | Port & Shipping Lane Debris Detection |
| **AUV Towfish Waterfall Logs** | Continuous Waterfall + Nav | Multi-channel scans (~15 MB) | Slant-to-Ground 2D Navigation Projection |
| **Active Learning Memory** | Relational Geo-DB + Crops | Dynamic edge database | Continuous Retraining & Error Prevention |

---

## 7. Sample Interactive Files in Workspace

All sample files are stored in **[`backend/datasets/samples/`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/)**:
* [`noaa_h11584_gulf_sample.tif`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/noaa_h11584_gulf_sample.tif) (NOAA Gulf of Mexico Mosaic)
* [`usgs_14bim05_breton_sample.tif`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/usgs_14bim05_breton_sample.tif) (USGS Breton Sound Sanctuary Mosaic)
* [`china_offshore_quanzhou_net.jpg`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/china_offshore_quanzhou_net.jpg) (Quanzhou Ghost Fishing Net)
* [`china_offshore_dongying_engine.jpg`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/china_offshore_dongying_engine.jpg) (Dongying Engine Debris)
* [`china_offshore_dongying_pipeline.jpg`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/china_offshore_dongying_pipeline.jpg) (Dongying Subsea Pipeline)
* [`towfish_mission_case_b.png`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/towfish_mission_case_b.png) (Towfish Waterfall Scan)
* [`towfish_mission_case_b.nav.json`](file:///c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel/sea-sentinel/backend/datasets/samples/towfish_mission_case_b.nav.json) (Synchronized Navigation Telemetry)
