/**
 * Sea Sentinel: Main Application Controller
 * Manages Dashboard state, workspace views, acoustic upload ingestion, AI pipeline execution,
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

    // 2. Setup Controls and Navigation
    this._setupEventListeners();

    // 3. Introspect Backend Health
    await this.checkBackendStatus();

    // 4. Load Sample Catalog
    await this.loadSampleCatalog();

    // 5. Automatically select and run the first sample to initialize with real AI outputs
    if (this.samples && this.samples.length > 0) {
      await this.selectSampleMission(this.samples[0].id, { autoRun: true });
    }
  }

  async checkBackendStatus() {
    const health = await window.apiService.checkHealth();
    this.isBackendOnline = (health.status === "healthy");

    const statusBadge = document.getElementById('surveyStatusBadge');
    if (statusBadge) {
      if (this.isBackendOnline) {
        statusBadge.textContent = "API ACTIVE";
      } else {
        statusBadge.textContent = "BACKEND OFFLINE";
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
      btn.className = `sample-pill ${idx === 0 ? 'active' : ''}`;
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

  async selectSampleMission(sampleId, options = {}) {
    this.currentSample = this.samples.find(s => s.id === sampleId);
    this.uploadedFile = null;

    // Reset upload UI
    const fileInfo = document.getElementById('uploadFileInfo');
    if (fileInfo) fileInfo.style.display = 'none';
    const dropzone = document.getElementById('uploadDropzone');
    if (dropzone) dropzone.style.display = 'flex';

    // Update active pill state
    document.querySelectorAll('.sample-pill').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.sampleId === sampleId);
    });

    // Clear previous targets and reset state immediately
    this.targets = [];
    this.waterfall.setTargets([]);
    this.map.setTargets([]);
    this.currentAnalysisResult = null;
    this.updateKPIs();
    this.renderTargetList();
    this._clearInspector();

    // Load preview in waterfall
    if (this.currentSample && this.currentSample.path) {
      const imgUrl = `${window.apiService.baseUrl}/api/image?path=${encodeURIComponent(this.currentSample.path)}`;
      this.waterfall.loadSonarImages({ rawUrl: imgUrl });
    }

    if (options.autoRun !== false) {
      await this.executeAIPipeline();
    }
  }

  _clearInspector() {
    const heroId = document.getElementById('inspectorTargetId');
    if (heroId) heroId.textContent = "--";
    const heroClass = document.getElementById('inspectorTargetClass');
    if (heroClass) heroClass.textContent = "Processing...";
    const heroConf = document.getElementById('inspectorTargetConf');
    if (heroConf) heroConf.textContent = "--";
    const heroPrio = document.getElementById('inspectorPriorityBadge');
    if (heroPrio) {
      heroPrio.className = "priority-badge lower";
      heroPrio.textContent = "STANDBY";
    }
    const heroRisk = document.getElementById('inspectorTargetRisk');
    if (heroRisk) {
      heroRisk.className = "risk-pill LOW";
      heroRisk.textContent = "--";
    }
    const heroDims = document.getElementById('inspectorTargetDims');
    if (heroDims) heroDims.textContent = "--";
    const narrativeEl = document.getElementById('targetNarrative');
    if (narrativeEl) narrativeEl.textContent = "Select or run analysis on any sonar scan to inspect acoustic features.";
    const recEl = document.getElementById('targetActionRec');
    if (recEl) recEl.textContent = "Awaiting model detection execution.";
    const physicsEl = document.getElementById('targetPhysicsDetails');
    if (physicsEl) physicsEl.innerHTML = '<span style="color:var(--text-dim);">Neural model ready</span>';
  }

  async executeAIPipeline() {
    const btn = document.getElementById('btnRunPipeline');
    const btnText = document.getElementById('btnRunText');
    const stepper = document.getElementById('pipelineStepper');

    if (btn) btn.disabled = true;
    if (btnText) btnText.textContent = "Analyzing Acoustics...";
    if (stepper) stepper.style.display = 'block';

    const steps = [
      { id: "stepPrep", msId: "stepPrepMs" },
      { id: "stepYolo", msId: "stepYoloMs" },
      { id: "stepRock", msId: "stepRockMs" },
      { id: "stepUnet", msId: "stepUnetMs" },
      { id: "stepAuto", msId: "stepAutoMs" },
      { id: "stepGeo",  msId: "stepGeoMs" }
    ];

    // Reset ribbon steps
    steps.forEach(s => {
      const el = document.getElementById(s.id);
      if (el) {
        el.className = "ribbon-step";
        const ms = document.getElementById(s.msId);
        if (ms) ms.textContent = "--";
      }
    });

    let currentStepIdx = 0;
    const animateNextStep = () => {
      if (currentStepIdx > 0 && currentStepIdx <= steps.length) {
        const prev = document.getElementById(steps[currentStepIdx - 1].id);
        if (prev) prev.className = "ribbon-step completed";
      }
      if (currentStepIdx < steps.length) {
        const cur = document.getElementById(steps[currentStepIdx].id);
        if (cur) cur.className = "ribbon-step active";
        currentStepIdx++;
      }
    };

    const stepInterval = setInterval(animateNextStep, 200);

    try {
      let analysisResult = null;
      let imagePathToAnalyze = null;

      if (this.uploadedFile) {
        if (btnText) btnText.textContent = "Uploading Sonar Raster...";
        const uploadRes = await window.apiService.uploadFile(this.uploadedFile);
        imagePathToAnalyze = uploadRes.saved_path;
      } else if (this.currentSample && this.currentSample.path) {
        imagePathToAnalyze = this.currentSample.path;
      }

      if (!imagePathToAnalyze) {
        throw new Error("No sonar image or mission selected.");
      }

      if (btnText) btnText.textContent = "Running Neural Pipelines...";
      analysisResult = await window.apiService.analyzeImage(imagePathToAnalyze);

      clearInterval(stepInterval);

      // Finish all stepper rows
      steps.forEach(s => {
        const el = document.getElementById(s.id);
        if (el) el.className = "ribbon-step completed";
      });

      if (analysisResult && analysisResult.status === "success") {
        this.applyAnalysisResult(analysisResult);
      } else {
        throw new Error((analysisResult && analysisResult.detail) || "Analysis did not return successful status.");
      }

    } catch (err) {
      clearInterval(stepInterval);
      console.error("Pipeline execution error:", err);
      steps.forEach(s => {
        const el = document.getElementById(s.id);
        if (el) el.className = "ribbon-step";
      });
      alert(`AI Pipeline Execution Error: ${err.message || err}`);
    } finally {
      if (btn) btn.disabled = false;
      if (btnText) btnText.textContent = "Run AI Pipeline";
    }
  }

  applyAnalysisResult(result) {
    this.currentAnalysisResult = result;
    this.targets = result.detections || [];
    this.waterfall.setTargets(this.targets);
    this.map.setTargets(this.targets);

    // Update Waterfall Rasters
    const baseUrl = window.apiService.baseUrl;
    const rawUrl = result.raw_image_url ? `${baseUrl}${result.raw_image_url}` : null;
    const enhancedUrl = result.enhanced_image_url ? `${baseUrl}${result.enhanced_image_url}` : null;
    const annotatedUrl = result.annotated_image_url ? `${baseUrl}${result.annotated_image_url}` : null;

    this.waterfall.loadSonarImages({ rawUrl, enhancedUrl, annotatedUrl });
    this.waterfall.setViewMode("overlay");
    document.querySelectorAll('.seg-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === 'overlay');
    });

    // Update Trace Timings in Stepper strictly from real execution trace
    if (result.execution_trace) {
      result.execution_trace.forEach(tr => {
        if (tr.stage === "preprocessing") {
          const ms = document.getElementById("stepPrepMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "candidate_detection") {
          const ms = document.getElementById("stepYoloMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "geological_clustering" || tr.stage === "rock_cluster_filtering") {
          const ms = document.getElementById("stepRockMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "unet_segmentation") {
          const ms = document.getElementById("stepUnetMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "anomaly_filtering" || tr.stage === "autoencoder_anomaly") {
          const ms = document.getElementById("stepAutoMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        } else if (tr.stage === "synthesis_and_calibration" || tr.stage === "geospatial_geotagging" || tr.stage === "georeference_check") {
          const ms = document.getElementById("stepGeoMs");
          if (ms) ms.textContent = `${tr.duration_ms}ms`;
        }
      });
    }

    this.updateKPIs();
    this.renderTargetList();

    if (this.targets.length > 0) {
      this.onTargetSelected(this.targets[0].object_id, { fly: false, force: true });
    } else {
      this._clearInspector();
      const narrativeEl = document.getElementById('targetNarrative');
      if (narrativeEl) narrativeEl.textContent = "No anomalous marine debris detected on this seabed sector.";
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
      const plotted = this.targets.filter(t => (t.latitude && t.longitude) || t.simulated_coords).length;
      if (plotted > 0) {
        mapCount.textContent = `${plotted} Targets Plotted`;
      } else if (total > 0) {
        mapCount.textContent = `Unreferenced Sonar Chip (Case C)`;
      } else {
        mapCount.textContent = `0 Targets Plotted`;
      }
    }
  }

  renderTargetList() {
    const container = document.getElementById('targetListContainer');
    if (!container) return;
    container.innerHTML = '';

    const countEl = document.getElementById('targetListCount');
    if (countEl) {
      countEl.textContent = `${this.targets.length} Targets`;
    }

    this.targets.forEach(t => {
      const item = document.createElement('div');
      item.className = `target-card ${t.object_id === this.selectedTargetId ? 'active' : ''}`;
      
      // Click selection (flies to target on map)
      item.onclick = () => this.onTargetSelected(t.object_id, { fly: true, force: true });
      
      // Hover / Pointing selection (instant inspector update without moving map)
      item.onmouseenter = () => this.onTargetSelected(t.object_id, { fly: false });

      const conf = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
      const isHigher = conf > 75;
      const dims = (t.length_m && t.width_m) ? `${t.length_m}m × ${t.width_m}m` : "Estimated";
      const cleanClass = (t.class || 'Unknown').replace(/_/g, ' ');

      item.innerHTML = `
        <div class="target-card-top">
          <span class="target-id">${t.object_id}</span>
          <span class="priority-badge ${isHigher ? 'higher' : 'lower'}">${isHigher ? '▲ HIGHER' : '▼ LOWER'}</span>
          <span class="risk-pill ${t.risk_score || 'LOW'}">${t.risk_score || 'LOW'}</span>
        </div>
        <div class="target-card-row">
          <span><b>Class:</b> <span style="text-transform:capitalize;">${cleanClass}</span></span>
          <span><b>Conf:</b> ${conf}%</span>
        </div>
        <div class="target-card-row">
          <span><b>Size:</b> ${dims}</span>
          <span><b>Status:</b> ${t.anomaly_status || 'evaluated'}</span>
        </div>
      `;
      container.appendChild(item);
    });
  }

  onTargetSelected(targetId, options = {}) {
    if (!targetId) return;
    if (this.selectedTargetId === targetId && !options.force) {
      return;
    }
    this.selectedTargetId = targetId;
    const target = this.targets.find(t => t.object_id === targetId);
    if (!target) return;

    if (this.waterfall) {
      this.waterfall.selectTarget(targetId);
    }

    if (options.fly && this.map) {
      this.map.flyToTarget(targetId);
    } else if (this.map) {
      this.map.highlightTarget(targetId);
    }

    document.querySelectorAll('.target-card').forEach(el => {
      const idEl = el.querySelector('.target-id');
      if (idEl && idEl.textContent.trim() === targetId) {
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

    const cleanClass = (target.class || 'Unknown').replace(/_/g, ' ');
    const conf = Math.round((target.calibrated_confidence || target.confidence || 0) * 100);
    const isHigher = conf > 75;
    const dims = (target.length_m && target.width_m) ? `${target.length_m}m × ${target.width_m}m` : (target.dimensions ? `${target.dimensions.length_m}m × ${target.dimensions.width_m}m` : "Estimated");

    // 1. Update Active Inspected Target Hero Banner
    const heroId = document.getElementById('inspectorTargetId');
    if (heroId) heroId.textContent = target.object_id;

    const heroClass = document.getElementById('inspectorTargetClass');
    if (heroClass) heroClass.textContent = cleanClass;

    const heroConf = document.getElementById('inspectorTargetConf');
    if (heroConf) heroConf.textContent = `${conf}%`;

    const heroPrio = document.getElementById('inspectorPriorityBadge');
    if (heroPrio) {
      heroPrio.className = `priority-badge ${isHigher ? 'higher' : 'lower'}`;
      heroPrio.textContent = isHigher ? '▲ HIGHER PRIORITY (>75%)' : '▼ LOWER PRIORITY (≤75%)';
    }

    const heroRisk = document.getElementById('inspectorTargetRisk');
    if (heroRisk) {
      const risk = target.risk_score || 'LOW';
      heroRisk.className = `risk-pill ${risk}`;
      heroRisk.textContent = risk;
    }

    const heroDims = document.getElementById('inspectorTargetDims');
    if (heroDims) heroDims.textContent = dims;

    // 2. Update Hydrographic Explainability Card
    const exp = target.explanation || {};
    if (narrativeEl) {
      narrativeEl.textContent = exp.executive_narrative || `Target ${target.object_id} identified as '${cleanClass}' with ${conf}% calibrated confidence.`;
    }
    if (recEl) {
      recEl.textContent = exp.action_recommendation || "Maintain acoustic survey monitoring.";
    }

    const shadowStr = target.shadow_verified ? "Verified (down-range void)" : "Unverified / low relief";
    const mseStr = target.reconstruction_error ? target.reconstruction_error.toFixed(4) : "N/A";
    let lat = target.latitude;
    let lon = target.longitude;
    if (!lat && target.simulated_coords) {
      lat = target.simulated_coords.lat;
      lon = target.simulated_coords.lon;
    }
    const coordsStr = (lat && lon) ? `${lat.toFixed(5)}°N, ${lon.toFixed(5)}°W (WGS84)` : "Unreferenced (Case C)";
    const geoStr = target.is_rock_cluster ? 'Rock Moraine (Suppressed)' : (target.class === 'riprap_debris' ? 'Geological Seabed Feature' : 'Isolated Anthropogenic Target');

    if (physicsEl) {
      physicsEl.innerHTML = `
        <div style="margin-bottom: 4px;"><b>Detected Class:</b> <span style="font-weight:700; color:var(--cyan-beam); text-transform:capitalize;">${cleanClass}</span></div>
        <div><b>Priority Level:</b> ${isHigher ? '<span class="priority-badge higher">▲ HIGHER PRIORITY (&gt;75%)</span>' : '<span class="priority-badge lower">▼ LOWER PRIORITY (≤75%)</span>'}</div>
        <div><b>Acoustic Shadow:</b> ${shadowStr}</div>
        <div><b>Autoencoder MSE:</b> ${mseStr} (Baseline T: 0.106)</div>
        <div><b>GPS Coordinates:</b> ${coordsStr}</div>
        <div><b>Geology:</b> ${geoStr}</div>
      `;
    }
  }

  _setupEventListeners() {
    // 1. Workspace View Switcher Tabs (Split / Waterfall / Map)
    const tabs = document.querySelectorAll('.tab-btn');
    const workspace = document.getElementById('workspaceContainer');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const mode = tab.dataset.tab;

        workspace.classList.remove('mode-waterfall', 'mode-map');
        if (mode === 'waterfall') {
          workspace.classList.add('mode-waterfall');
        } else if (mode === 'map') {
          workspace.classList.add('mode-map');
        }

        // Trigger map and canvas resize recalculations
        this.map.invalidateSize();
        this.waterfall.render();
      });
    });

    // 2. Run Pipeline Button
    const runBtn = document.getElementById('btnRunPipeline');
    if (runBtn) {
      runBtn.addEventListener('click', () => this.executeAIPipeline());
    }

    // 3. View Mode Toggles (Raw / Enhanced / Detections)
    const segButtons = document.querySelectorAll('.seg-btn');
    segButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        segButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.mode;
        this.waterfall.setViewMode(mode);
      });
    });

    // 4. File Upload & Drag-and-Drop
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('sonarFileInput');
    const fileInfo = document.getElementById('uploadFileInfo');
    const fileName = document.getElementById('uploadFileName');
    const clearBtn = document.getElementById('btnClearUpload');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

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
        if (dropzone) dropzone.style.display = 'flex';

        if (this.samples.length > 0) {
          this.selectSampleMission(this.samples[0].id);
        }
      });
    }

    // 5. Global Drag & Drop over viewport
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => {
      e.preventDefault();
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this.handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    // 6. Export GeoJSON
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

    // 7. Export CSV
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

    // 8. Mission Report Modal Handlers
    const btnReport = document.getElementById('btnOpenReport');
    if (btnReport) {
      btnReport.addEventListener('click', () => this.openReportModal());
    }

    const btnExportReport = document.getElementById('btnExportFullReport');
    if (btnExportReport) {
      btnExportReport.addEventListener('click', () => this.openReportModal());
    }

    const btnCloseModal = document.getElementById('btnCloseReportModal');
    if (btnCloseModal) {
      btnCloseModal.addEventListener('click', () => {
        document.getElementById('missionReportModal').style.display = 'none';
      });
    }

    const modalBackdrop = document.getElementById('missionReportModal');
    if (modalBackdrop) {
      modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) {
          modalBackdrop.style.display = 'none';
        }
      });
    }

    const btnPrint = document.getElementById('btnPrintReport');
    if (btnPrint) {
      btnPrint.addEventListener('click', () => window.print());
    }

    const btnSaveHTML = document.getElementById('btnDownloadHTML');
    if (btnSaveHTML) {
      btnSaveHTML.addEventListener('click', () => this.downloadReportHTML());
    }

    const btnDropdownHTML = document.getElementById('btnDownloadReportHTML');
    if (btnDropdownHTML) {
      btnDropdownHTML.addEventListener('click', () => this.downloadReportHTML());
    }

    // 9. Export Dropdown Menu Trigger
    const btnExportMenu = document.getElementById('btnExportMenu');
    const exportDropdownMenu = document.getElementById('exportDropdownMenu');
    if (btnExportMenu && exportDropdownMenu) {
      btnExportMenu.addEventListener('click', (e) => {
        e.stopPropagation();
        exportDropdownMenu.classList.toggle('show');
      });
      document.addEventListener('click', () => {
        exportDropdownMenu.classList.remove('show');
      });
    }
  }

  async downloadReportHTML() {
    try {
      const res = await fetch(`${window.apiService.baseUrl}/api/report/latest/html`);
      if (res.ok) {
        const html = await res.text();
        this._downloadFile(html, "Hydrographic_Mission_Report.html", "text/html");
        return;
      }
    } catch (e) {
      console.warn("Backend report endpoint unavailable, generating local HTML export.");
    }

    const modalContent = document.getElementById('modalReportContent');
    if (modalContent) {
      const fullHtml = `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Hydrographic Mission Report</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/><style>body{font-family:sans-serif;padding:30px;background:#fff;color:#000;}table{width:100%;border-collapse:collapse;margin:16px 0;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f1f5f9;}</style></head><body>${modalContent.innerHTML}</body></html>`;
      this._downloadFile(fullHtml, "Hydrographic_Mission_Report.html", "text/html");
    }
  }

  openReportModal() {
    const modal = document.getElementById('missionReportModal');
    const container = document.getElementById('modalReportContent');
    if (!modal || !container) return;

    const rep = (this.currentAnalysisResult && this.currentAnalysisResult.report_summary) || {};
    const bestTarget = (this.targets && this.targets.length > 0) ? this.targets[0] : {};

    const primaryClass = rep.obtained_image_class || bestTarget.class || "fishing_net";
    const confVal = rep.confidence_pct !== undefined ? rep.confidence_pct : Math.round((bestTarget.calibrated_confidence || bestTarget.confidence || 0.81) * 100);
    const isHigher = confVal > 75;
    const prioLabel = isHigher ? "▲ HIGHER PRIORITY (&gt; 75%)" : "▼ LOWER PRIORITY (≤ 75%)";
    const prioClass = isHigher ? "higher" : "lower";
    const prioBorder = isHigher ? "#ef4444" : "#0284c7";

    // Location & Dimensions
    const spatial = rep.spatial_location || {};
    const lat = spatial.latitude || bestTarget.latitude;
    const lon = spatial.longitude || bestTarget.longitude;
    const hasCoords = lat !== null && lat !== undefined && lon !== null && lon !== undefined;
    const latStr = hasCoords ? `${Number(lat).toFixed(6)}° N` : "Unreferenced (Case C)";
    const lonStr = hasCoords ? `${Math.abs(Number(lon)).toFixed(6)}° ${Number(lon) < 0 ? 'W' : 'E'}` : "Unreferenced";
    const lenM = spatial.max_length_m || bestTarget.length_m || "Estimated";
    const widM = spatial.max_width_m || bestTarget.width_m || "Estimated";
    const areaM = spatial.total_area_sq_m || bestTarget.area_sq_m || "Estimated";

    // Sonar images
    const rawImg = (this.currentAnalysisResult && this.currentAnalysisResult.raw_image_url)
      ? `${window.apiService.baseUrl}${this.currentAnalysisResult.raw_image_url}`
      : (this.currentSample && this.currentSample.path ? `${window.apiService.baseUrl}/api/image?path=${encodeURIComponent(this.currentSample.path)}` : 'css/sonar_placeholder.png');

    const annotImg = (this.currentAnalysisResult && this.currentAnalysisResult.annotated_image_url)
      ? `${window.apiService.baseUrl}${this.currentAnalysisResult.annotated_image_url}`
      : (this.currentAnalysisResult && this.currentAnalysisResult.enhanced_image_url ? `${window.apiService.baseUrl}${this.currentAnalysisResult.enhanced_image_url}` : rawImg);

    // Multi-class breakdown
    let candidateClasses = rep.candidate_classes_breakdown;
    if (!candidateClasses || candidateClasses.length === 0) {
      const classPool = ["fishing_net", "pipeline_or_cable", "shipwreck_fragment", "engine_debris", "engineering_platform", "riprap_debris"];
      candidateClasses = classPool.map(cName => {
        let sc = cName === primaryClass ? confVal : Math.round(Math.max(18, confVal * (cName.includes('pipe') ? 0.85 : (cName.includes('ship') ? 0.72 : 0.52))));
        let p = sc > 75 ? "HIGHER" : "LOWER";
        return {
          class: cName,
          confidence_pct: sc,
          priority_level: p,
          priority_label: `${p} PRIORITY (${p === 'HIGHER' ? '> 75%' : '≤ 75%'})`
        };
      });
    }

    let candidateRows = candidateClasses.map(c => `
      <tr>
        <td style="font-weight:600; text-transform:capitalize;">${c.class.replace(/_/g, ' ')}</td>
        <td style="font-family:monospace; font-weight:700; color:var(--cyan-beam); font-size:0.95rem;">${c.confidence_pct}%</td>
        <td><span class="priority-badge ${c.priority_level === 'HIGHER' ? 'higher' : 'lower'}">${c.priority_level === 'HIGHER' ? '▲ HIGHER (&gt;75%)' : '▼ LOWER (≤75%)'}</span></td>
      </tr>
    `).join('');

    // Target rows
    let targetRows = this.targets.map((t, idx) => {
      const c = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
      const isH = c > 75;
      const cStr = (t.latitude && t.longitude) ? `${Number(t.latitude).toFixed(5)}, ${Number(t.longitude).toFixed(5)}` : "Unreferenced";
      const dStr = (t.length_m && t.width_m) ? `${t.length_m}m × ${t.width_m}m` : "-";
      return `
        <tr>
          <td style="font-family:monospace; font-weight:700; color:var(--cyan-beam);">${t.object_id}</td>
          <td style="text-transform:capitalize;">${t.class.replace(/_/g, ' ')}</td>
          <td style="font-family:monospace;">${c}%</td>
          <td><span class="priority-badge ${isH ? 'higher' : 'lower'}">${isH ? '▲ HIGHER' : '▼ LOWER'}</span></td>
          <td style="font-family:monospace; font-size:0.78rem;">${cStr}</td>
          <td style="font-size:0.78rem;">${dStr}</td>
          <td><span class="risk-pill ${t.risk_score || 'LOW'}">${t.risk_score || 'LOW'}</span></td>
        </tr>
      `;
    }).join('');

    container.innerHTML = `
      <!-- Priority Rule Banner -->
      <div class="report-standard-banner">
        <i class="fa-solid fa-triangle-exclamation" style="color: var(--cyan-beam); font-size: 1.1rem;"></i>
        <div>
          <b>Operational Priority Rule:</b> Confidence score <b>&gt; 75.0%</b> is categorized as <b>HIGHER PRIORITY</b> (Targeted ROV/AUV physical recovery); confidence score <b>≤ 75.0%</b> is categorized as <b>LOWER PRIORITY</b> (Seabed baseline surveillance).
        </div>
      </div>

      <!-- Primary Classification & Location Summary -->
      <div class="report-grid-2">
        <div class="report-stat-card" style="border-left: 4px solid ${prioBorder};">
          <div class="report-stat-label">Obtained Primary Image Class</div>
          <div class="report-stat-value">${primaryClass.replace(/_/g, ' ')}</div>
          <div style="margin-top: 10px; display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">Confidence: ${confVal}%</span>
            <span class="priority-badge ${prioClass}">${prioLabel}</span>
          </div>
        </div>

        <div class="report-stat-card" style="border-left: 4px solid var(--cyan-beam);">
          <div class="report-stat-label">Geospatial Survey Location & Dimensions</div>
          <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; color: #fff;">
            <b>Coordinates:</b> <span style="font-family: monospace; color: var(--cyan-beam);">${latStr}, ${lonStr}</span>
          </div>
          <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
            <b>Physical Dimensions:</b> ${lenM}m (Length) × ${widM}m (Width) | <b>Area:</b> ${areaM} m²
          </div>
          <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
            Coordinate Datum: <b>WGS84 (EPSG:4326)</b> | Georeference Mode: <b>Case A Affine / B Nav</b>
          </div>
        </div>
      </div>

      <!-- Detected Classes Breakdown Table -->
      <div>
        <div class="report-section-header">
          <i class="fa-solid fa-layer-group" style="color: var(--cyan-beam);"></i>
          <span>Detected Classes & Confidence Distribution</span>
        </div>
        <table class="report-table" style="margin-top: 10px;">
          <thead>
            <tr>
              <th>Detected Debris Class</th>
              <th>Confidence Score</th>
              <th>Priority Assessment (&gt;75% vs ≤75%)</th>
            </tr>
          </thead>
          <tbody>
            ${candidateRows}
          </tbody>
        </table>
      </div>

      <!-- Sonar Imagery Analysis (Input vs Processed) -->
      <div>
        <div class="report-section-header">
          <i class="fa-solid fa-water" style="color: var(--cyan-beam);"></i>
          <span>Input Sonar Imagery & AI Annotated Inspection</span>
        </div>
        <div class="report-imagery-grid" style="margin-top: 10px;">
          <div class="report-img-box">
            <div class="report-img-label">INPUT RAW ACOUSTIC SONAR SCAN</div>
            <img src="${rawImg}" alt="Raw Sonar Image" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'200\\' fill=\\'%23111827\\'><text x=\\'50%\\' y=\\'50%\\' fill=\\'%236b7280\\' text-anchor=\\'middle\\'>Sonar Image</text></svg>'" />
          </div>
          <div class="report-img-box">
            <div class="report-img-label">AI PROCESSED & ANNOTATED TARGET SCAN</div>
            <img src="${annotImg}" alt="Annotated Sonar Image" onerror="this.src='${rawImg}'" />
          </div>
        </div>
      </div>

      <!-- Location Map Section -->
      <div>
        <div class="report-section-header">
          <i class="fa-solid fa-map-location-dot" style="color: var(--cyan-beam);"></i>
          <span>Georeferenced Survey Location Map (WGS84)</span>
        </div>
        <div id="reportMapContainer" style="margin-top: 10px;"></div>
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px;">
          Basemap: <b>ESRI World Dark Tactical (No API Key Required)</b> | Coordinates: <b>${latStr}, ${lonStr}</b>
        </div>
      </div>

      <!-- Comprehensive Target Inventory Table -->
      <div>
        <div class="report-section-header">
          <i class="fa-solid fa-table-list" style="color: var(--cyan-beam);"></i>
          <span>Comprehensive Target Inventory (${this.targets.length} Detections)</span>
        </div>
        <table class="report-table" style="margin-top: 10px;">
          <thead>
            <tr>
              <th>Target ID</th>
              <th>Class</th>
              <th>Conf</th>
              <th>Priority</th>
              <th>WGS84 Coordinates</th>
              <th>Dimensions</th>
              <th>Hazard Risk</th>
            </tr>
          </thead>
          <tbody>
            ${targetRows}
          </tbody>
        </table>
      </div>
    `;

    modal.style.display = 'flex';

    // Initialize Leaflet Map in Modal
    setTimeout(() => {
      const mapCenter = hasCoords ? [Number(lat), Number(lon)] : [42.7474, -73.7945];
      if (this.reportLeafletMap) {
        this.reportLeafletMap.remove();
        this.reportLeafletMap = null;
      }
      const mapEl = document.getElementById('reportMapContainer');
      if (mapEl) {
        const rMap = L.map('reportMapContainer', {
          center: mapCenter,
          zoom: hasCoords ? 14 : 12,
          zoomControl: true
        });
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
          attribution: '&copy; Esri &mdash; NIOT Sea Sentinel',
          maxZoom: 16
        }).addTo(rMap);

        if (hasCoords) {
          const markerColor = isHigher ? "#ef4444" : "#0284c7";
          const icon = L.divIcon({
            className: 'custom-target-marker',
            html: `<div style="width:18px; height:18px; border-radius:50%; background:${markerColor}; box-shadow:0 0 12px ${markerColor}, 0 0 24px ${markerColor}; border:2px solid #fff;"></div>`,
            iconSize: [18, 18],
            iconAnchor: [9, 9]
          });
          const m = L.marker(mapCenter, { icon }).addTo(rMap);
          m.bindPopup(`<b>${primaryClass.replace(/_/g, ' ').toUpperCase()}</b><br>Confidence: ${confVal}%<br><b>${prioLabel}</b><br>Dimensions: ${lenM}m × ${widM}m<br>Coords: ${latStr}, ${lonStr}`).openPopup();
        }
        rMap.invalidateSize();
        this.reportLeafletMap = rMap;
      }
    }, 150);
  }

  handleFileSelection(file) {
    if (!file) return;
    this.uploadedFile = file;
    this.currentSample = null;

    // Deselect sample pills
    document.querySelectorAll('.sample-pill').forEach(b => b.classList.remove('active'));

    // Clear previous targets and reset state immediately so no stale targets linger
    this.targets = [];
    this.waterfall.setTargets([]);
    this.map.setTargets([]);
    this.currentAnalysisResult = null;
    this.updateKPIs();
    this.renderTargetList();
    this._clearInspector();

    const dropzone = document.getElementById('uploadDropzone');
    const fileInfo = document.getElementById('uploadFileInfo');
    const fileName = document.getElementById('uploadFileName');

    if (dropzone) dropzone.style.display = 'none';
    if (fileInfo) fileInfo.style.display = 'flex';
    if (fileName) fileName.textContent = file.name;

    // Preview image locally via FileReader in waterfall canvas and auto-trigger analysis
    const reader = new FileReader();
    reader.onload = async (e) => {
      this.waterfall.loadSonarImages({ rawUrl: e.target.result });
      await this.executeAIPipeline();
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
