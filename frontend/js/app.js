/**
 * Sea Sentinel: Main Application Controller
 * Manages Dashboard state, user interactions, upload ingestion, AI pipeline execution,
 * export actions, and synchronized multi-component updates.
 */

class DashboardApp {
  constructor() {
    this.targets = [];
    this.selectedTargetId = null;
    this.waterfall = null;
    this.map = null;

    this.samples = [];
    this.currentSample = null;
    this.uploadedFile = null;
    this.isBackendOnline = false;

    this._init();
  }

  async _init() {
    // 1. Initialize Visual Components
    this.waterfall = new WaterfallViewer('sonarCanvas');
    this.map = new GISMap('leafletMap');

    // 2. Introspect Backend Health
    await this.checkBackendStatus();

    // 3. Load Sample Catalog
    await this.loadSampleCatalog();

    // 4. Fetch Initial Targets & Setup Controls
    await this.loadTargets();
    this._setupEventListeners();
  }

  async checkBackendStatus() {
    const health = await window.apiService.checkHealth();
    this.isBackendOnline = (health.status === "healthy");

    const statusBadge = document.getElementById('surveyStatusBadge');
    if (statusBadge) {
      if (this.isBackendOnline) {
        statusBadge.textContent = "API ACTIVE";
        statusBadge.className = "badge-status-online";
      } else {
        statusBadge.textContent = "DEMO MODE";
        statusBadge.className = "brand-badge";
      }
    }
  }

  async loadSampleCatalog() {
    this.samples = await window.apiService.fetchSamples();
    const container = document.getElementById('sampleChipsContainer');
    if (!container) return;

    container.innerHTML = '';
    this.samples.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.className = `sample-chip ${idx === 0 ? 'active' : ''}`;
      btn.dataset.sampleId = s.id;

      let icon = "fa-network-wired";
      if (s.category === "pipeline_or_cable") icon = "fa-bolt";
      else if (s.category === "riprap_debris") icon = "fa-mountain";
      else if (s.category === "engine_part") icon = "fa-gears";

      btn.innerHTML = `<i class="fa-solid ${icon}"></i> ${s.name.split(' ')[0]} ${s.name.split(' ')[1] || ''}`;
      btn.title = s.description || s.name;
      btn.onclick = () => this.selectSampleMission(s.id);
      container.appendChild(btn);
    });

    if (this.samples.length > 0) {
      this.currentSample = this.samples[0];
    }
  }

  selectSampleMission(sampleId) {
    this.currentSample = this.samples.find(s => s.id === sampleId);
    this.uploadedFile = null;

    // Reset upload UI
    const fileInfo = document.getElementById('uploadFileInfo');
    if (fileInfo) fileInfo.style.display = 'none';
    const dropzone = document.getElementById('uploadDropzone');
    if (dropzone) dropzone.style.display = 'block';

    // Update active chip state
    document.querySelectorAll('.sample-chip').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.sampleId === sampleId);
    });

    const modeText = document.getElementById('ingestionModeText');
    if (modeText) modeText.textContent = "BENCHMARK";

    // If sample has URL or path, load preview in waterfall
    if (this.currentSample && this.currentSample.path && this.isBackendOnline) {
      const imgUrl = `${window.apiService.baseUrl}/api/image?path=${encodeURIComponent(this.currentSample.path)}`;
      this.waterfall.loadSonarImages({ rawUrl: imgUrl });
    }
  }

  async loadTargets() {
    this.targets = await window.apiService.getSurveyTargets();
    this.waterfall.setTargets(this.targets);
    this.map.setTargets(this.targets);

    this.updateKPIs();
    this.renderTargetList();

    if (this.targets.length > 0) {
      this.onTargetSelected(this.targets[0].object_id);
    }
  }

  async executeAIPipeline() {
    const btn = document.getElementById('btnRunPipeline');
    const btnText = document.getElementById('btnRunText');
    const stepper = document.getElementById('pipelineStepper');

    btn.disabled = true;
    btnText.textContent = "ANALYZING ACOUSTICS...";
    if (stepper) stepper.style.display = 'block';

    const steps = [
      { id: "stepPrep", msId: "stepPrepMs", label: "Preprocessing" },
      { id: "stepYolo", msId: "stepYoloMs", label: "YOLO Candidate" },
      { id: "stepRock", msId: "stepRockMs", label: "DBSCAN Moraine" },
      { id: "stepUnet", msId: "stepUnetMs", label: "Attention U-Net" },
      { id: "stepAuto", msId: "stepAutoMs", label: "CNN Autoencoder" },
      { id: "stepGeo",  msId: "stepGeoMs",  label: "Geotagging" }
    ];

    // Reset stepper
    steps.forEach(s => {
      const el = document.getElementById(s.id);
      if (el) {
        el.className = "step-row";
        document.getElementById(s.msId).textContent = "--";
      }
    });

    // Start animated stepper progression
    let currentStepIdx = 0;
    const animateNextStep = () => {
      if (currentStepIdx > 0 && currentStepIdx <= steps.length) {
        const prev = document.getElementById(steps[currentStepIdx - 1].id);
        if (prev) prev.className = "step-row completed";
      }
      if (currentStepIdx < steps.length) {
        const cur = document.getElementById(steps[currentStepIdx].id);
        if (cur) cur.className = "step-row active";
        currentStepIdx++;
      }
    };

    const stepInterval = setInterval(animateNextStep, 250);

    try {
      let analysisResult = null;

      if (this.isBackendOnline) {
        let imagePathToAnalyze = null;

        if (this.uploadedFile) {
          // Upload file first
          const uploadRes = await window.apiService.uploadFile(this.uploadedFile);
          imagePathToAnalyze = uploadRes.saved_path;
        } else if (this.currentSample && this.currentSample.path) {
          imagePathToAnalyze = this.currentSample.path;
        }

        if (imagePathToAnalyze) {
          analysisResult = await window.apiService.analyzeImage(imagePathToAnalyze);
        }
      }

      clearInterval(stepInterval);

      // Finish all stepper rows
      steps.forEach(s => {
        const el = document.getElementById(s.id);
        if (el) el.className = "step-row completed";
      });

      if (analysisResult && analysisResult.status === "success") {
        this.applyAnalysisResult(analysisResult);
      } else {
        // Fallback simulation demonstration
        this.applySimulatedResult();
      }

    } catch (err) {
      clearInterval(stepInterval);
      console.error("Pipeline execution error:", err);
      this.applySimulatedResult();
    } finally {
      btn.disabled = false;
      btnText.textContent = "EXECUTE AI PIPELINE";
    }
  }

  applyAnalysisResult(result) {
    this.targets = result.detections || [];
    this.waterfall.setTargets(this.targets);
    this.map.setTargets(this.targets);

    // Update Waterfall Rasters if available
    const baseUrl = window.apiService.baseUrl;
    const rawUrl = result.raw_image_url ? `${baseUrl}${result.raw_image_url}` : null;
    const enhancedUrl = result.enhanced_image_url ? `${baseUrl}${result.enhanced_image_url}` : null;
    const annotatedUrl = result.annotated_image_url ? `${baseUrl}${result.annotated_image_url}` : null;

    this.waterfall.loadSonarImages({ rawUrl, enhancedUrl, annotatedUrl });

    // Update Trace Timings in Stepper
    if (result.execution_trace) {
      result.execution_trace.forEach(tr => {
        if (tr.stage === "preprocessing") {
          const ms = document.getElementById("stepPrepMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "candidate_detection") {
          const ms = document.getElementById("stepYoloMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "geological_clustering") {
          const ms = document.getElementById("stepRockMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "unet_segmentation") {
          const ms = document.getElementById("stepUnetMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "anomaly_filtering") {
          const ms = document.getElementById("stepAutoMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "synthesis_and_calibration") {
          const ms = document.getElementById("stepGeoMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        }
      });
    }

    this.updateKPIs();
    this.renderTargetList();

    if (this.targets.length > 0) {
      this.onTargetSelected(this.targets[0].object_id);
    }
  }

  applySimulatedResult() {
    // Fill simulated timings
    const simulatedTimes = {
      stepPrepMs: "18.4ms",
      stepYoloMs: "32.1ms",
      stepRockMs: "4.8ms",
      stepUnetMs: "26.5ms",
      stepAutoMs: "12.0ms",
      stepGeoMs: "8.2ms"
    };

    Object.entries(simulatedTimes).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    });

    this.targets = window.BENCHMARK_TARGETS || [];
    this.waterfall.setTargets(this.targets);
    this.map.setTargets(this.targets);

    this.updateKPIs();
    this.renderTargetList();

    if (this.targets.length > 0) {
      this.onTargetSelected(this.targets[0].object_id);
    }
  }

  updateKPIs() {
    const total = this.targets.length;
    const confirmed = this.targets.filter(t => t.anomaly_status === "confirmed_debris").length;
    const suspicious = this.targets.filter(t => t.anomaly_status === "suspicious_anomaly").length;
    const highRisk = this.targets.filter(t => t.risk_score === "HIGH").length;

    document.getElementById('kpiTotal').textContent = total;
    document.getElementById('kpiConfirmed').textContent = confirmed;
    document.getElementById('kpiSuspicious').textContent = suspicious;
    document.getElementById('kpiHighRisk').textContent = highRisk;

    const mapCount = document.getElementById('mapTargetCount');
    if (mapCount) {
      const plotted = this.targets.filter(t => t.latitude && t.longitude).length;
      mapCount.textContent = `${plotted} MARKERS PLOTTED`;
    }
  }

  renderTargetList() {
    const container = document.getElementById('targetListContainer');
    if (!container) return;
    container.innerHTML = '';

    this.targets.forEach(t => {
      const item = document.createElement('div');
      item.className = `target-item ${t.object_id === this.selectedTargetId ? 'active' : ''}`;
      item.onclick = () => this.onTargetSelected(t.object_id);

      const conf = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
      const dims = (t.length_m && t.width_m) ? `${t.length_m}m × ${t.width_m}m` : "Estimated";

      item.innerHTML = `
        <div class="target-item-header">
          <span class="target-id">${t.object_id}</span>
          <span class="badge-risk ${t.risk_score}">${t.risk_score}</span>
        </div>
        <div class="target-meta-row">
          <span><b>Class:</b> ${t.class}</span>
          <span><b>Conf:</b> ${conf}%</span>
        </div>
        <div class="target-meta-row">
          <span><b>Size:</b> ${dims}</span>
          <span><b>Status:</b> ${t.anomaly_status}</span>
        </div>
      `;
      container.appendChild(item);
    });
  }

  onTargetSelected(targetId) {
    this.selectedTargetId = targetId;
    const target = this.targets.find(t => t.object_id === targetId);
    if (!target) return;

    this.waterfall.selectTarget(targetId);
    this.map.flyToTarget(targetId);

    document.querySelectorAll('.target-item').forEach(el => {
      const idEl = el.querySelector('.target-id');
      if (idEl && idEl.textContent === targetId) {
        el.classList.add('active');
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        el.classList.remove('active');
      }
    });

    this.renderTargetNarrative(target);
  }

  renderTargetNarrative(target) {
    const narrativeEl = document.getElementById('targetNarrative');
    const recEl = document.getElementById('targetActionRec');
    const physicsEl = document.getElementById('targetPhysicsDetails');

    const exp = target.explanation || {};
    if (narrativeEl) narrativeEl.textContent = exp.executive_narrative || "Target evaluated by Sea Sentinel pipeline.";
    if (recEl) recEl.textContent = exp.action_recommendation || "Maintain monitoring.";

    const shadowStr = target.shadow_verified ? "Verified (down-range void)" : "Unverified / low relief";
    const mseStr = target.reconstruction_error ? target.reconstruction_error.toFixed(4) : "N/A";
    const coordsStr = (target.latitude && target.longitude) ? `${target.latitude.toFixed(5)}°N, ${target.longitude.toFixed(5)}°W (WGS84)` : "Unreferenced (Case C)";

    if (physicsEl) {
      physicsEl.innerHTML = `
        <div><b>Acoustic Shadow:</b> ${shadowStr}</div>
        <div><b>Autoencoder MSE:</b> ${mseStr} (Baseline T: 0.094)</div>
        <div><b>GPS Coordinates:</b> ${coordsStr}</div>
        <div><b>Geological Density:</b> ${target.is_rock_cluster ? 'Rock Field Moraine (Suppressed)' : 'Isolated Anthropogenic Anomaly'}</div>
      `;
    }
  }

  _setupEventListeners() {
    // 1. Run Pipeline Button
    const runBtn = document.getElementById('btnRunPipeline');
    if (runBtn) {
      runBtn.addEventListener('click', () => this.executeAIPipeline());
    }

    // 2. View Mode Toggles
    const viewButtons = document.querySelectorAll('.view-mode-btn');
    viewButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        viewButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.mode;
        this.waterfall.setViewMode(mode);
      });
    });

    // 3. File Upload & Drag-and-Drop
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('sonarFileInput');
    const fileInfo = document.getElementById('uploadFileInfo');
    const fileName = document.getElementById('uploadFileName');
    const clearBtn = document.getElementById('btnClearUpload');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          this.handleFileSelection(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFileSelection(e.target.files[0]);
        }
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.uploadedFile = null;
        if (fileInput) fileInput.value = '';
        if (fileInfo) fileInfo.style.display = 'none';
        if (dropzone) dropzone.style.display = 'block';

        if (this.samples.length > 0) {
          this.selectSampleMission(this.samples[0].id);
        }
      });
    }

    // 4. Export GeoJSON
    const btnGeoJSON = document.getElementById('btnExportGeoJSON');
    if (btnGeoJSON) {
      btnGeoJSON.addEventListener('click', () => {
        const geojson = {
          type: "FeatureCollection",
          features: this.targets
            .filter(t => t.latitude && t.longitude)
            .map(t => ({
              type: "Feature",
              geometry: { type: "Point", coordinates: [t.longitude, t.latitude] },
              properties: { ...t }
            }))
        };
        this._downloadFile(JSON.stringify(geojson, null, 2), "survey_targets.geojson", "application/json");
      });
    }

    // 5. Export CSV
    const btnCSV = document.getElementById('btnExportCSV');
    if (btnCSV) {
      btnCSV.addEventListener('click', () => {
        const headers = ["object_id", "class", "calibrated_confidence", "anomaly_status", "risk_score", "latitude", "longitude", "length_m", "width_m"];
        const rows = this.targets.map(t => [
          t.object_id, t.class, t.calibrated_confidence || t.confidence,
          t.anomaly_status, t.risk_score, t.latitude || "", t.longitude || "",
          t.length_m || "", t.width_m || ""
        ]);
        const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        this._downloadFile(csvContent, "survey_targets_summary.csv", "text/csv");
      });
    }
  }

  handleFileSelection(file) {
    this.uploadedFile = file;
    this.currentSample = null;

    // Deselect sample chips
    document.querySelectorAll('.sample-chip').forEach(b => b.classList.remove('active'));

    const modeText = document.getElementById('ingestionModeText');
    if (modeText) modeText.textContent = "CUSTOM SCAN";

    const dropzone = document.getElementById('uploadDropzone');
    const fileInfo = document.getElementById('uploadFileInfo');
    const fileName = document.getElementById('uploadFileName');

    if (dropzone) dropzone.style.display = 'none';
    if (fileInfo) fileInfo.style.display = 'flex';
    if (fileName) fileName.textContent = file.name;

    // Preview image locally via FileReader in waterfall canvas
    const reader = new FileReader();
    reader.onload = (e) => {
      this.waterfall.loadSonarImages({ rawUrl: e.target.result });
    };
    reader.readAsDataURL(file);
  }

  _downloadFile(content, fileName, contentType) {
    const blob = new Blob([content], { type: contentType });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(a.href);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new DashboardApp();
});

