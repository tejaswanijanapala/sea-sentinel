/**
 * Sea Sentinel: Main Application Controller
 * Manages Dashboard state, user interactions, export actions, and UI synchronization.
 */

class DashboardApp {
  constructor() {
    this.targets = [];
    this.selectedTargetId = null;
    this.waterfall = null;
    this.map = null;

    this._init();
  }

  async _init() {
    // 1. Initialize Components
    this.waterfall = new WaterfallViewer('sonarCanvas');
    this.map = new GISMap('leafletMap');

    // 2. Fetch Initial Targets
    await this.loadTargets();

    // 3. Setup Export & Control Listeners
    this._setupEventListeners();
  }

  async loadTargets() {
    this.targets = await window.apiService.getSurveyTargets();
    this.waterfall.setTargets(this.targets);
    this.map.setTargets(this.targets);

    this.updateKPIs();
    this.renderTargetList();

    // Select first target by default
    if (this.targets.length > 0) {
      this.onTargetSelected(this.targets[0].object_id);
    }
  }

  updateKPIs() {
    const total = this.targets.length;
    const confirmed = this.targets.filter(t => t.anomaly_status === "confirmed_debris").length;
    const suspicious = this.targets.filter(t => t.anomaly_status === "suspicious_anomaly").length;
    const rocksSuppressed = this.targets.filter(t => t.is_rock_cluster).length;
    const highRisk = this.targets.filter(t => t.risk_score === "HIGH").length;

    document.getElementById('kpiTotal').textContent = total;
    document.getElementById('kpiConfirmed').textContent = confirmed;
    document.getElementById('kpiSuspicious').textContent = suspicious;
    document.getElementById('kpiHighRisk').textContent = highRisk;
  }

  renderTargetList() {
    const container = document.getElementById('targetListContainer');
    container.innerHTML = '';

    this.targets.forEach(t => {
      const item = document.createElement('div');
      item.className = `target-item ${t.object_id === this.selectedTargetId ? 'active' : ''}`;
      item.onclick = () => this.onTargetSelected(t.object_id);

      const conf = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
      const dims = (t.length_m && t.width_m) ? `${t.length_m}m × ${t.width_m}m` : "Pixel dims";

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

    // Highlight in Waterfall and Map
    this.waterfall.selectTarget(targetId);
    this.map.flyToTarget(targetId);

    // Update List UI Active state
    document.querySelectorAll('.target-item').forEach(el => {
      if (el.querySelector('.target-id').textContent === targetId) {
        el.classList.add('active');
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        el.classList.remove('active');
      }
    });

    // Update Detail Narrative Box
    this.renderTargetNarrative(target);
  }

  renderTargetNarrative(target) {
    const narrativeEl = document.getElementById('targetNarrative');
    const recEl = document.getElementById('targetActionRec');
    const physicsEl = document.getElementById('targetPhysicsDetails');

    const exp = target.explanation || {};
    narrativeEl.textContent = exp.executive_narrative || "Target evaluated by Sea Sentinel pipeline.";
    recEl.textContent = exp.action_recommendation || "Maintain monitoring.";

    const shadowStr = target.shadow_verified ? "Verified (down-range void)" : "Unverified / low relief";
    const mseStr = target.reconstruction_error ? target.reconstruction_error.toFixed(4) : "N/A";
    const coordsStr = (target.latitude && target.longitude) ? `${target.latitude.toFixed(5)}°N, ${target.longitude.toFixed(5)}°W (WGS84)` : "Unreferenced";

    physicsEl.innerHTML = `
      <div><b>Acoustic Shadow:</b> ${shadowStr}</div>
      <div><b>Autoencoder MSE:</b> ${mseStr} (Baseline T: 0.094)</div>
      <div><b>GPS Coordinates:</b> ${coordsStr}</div>
      <div><b>Geological Density:</b> ${target.is_rock_cluster ? 'Rock Field Cluster (Penalized)' : 'Isolated Target'}</div>
    `;
  }

  _setupEventListeners() {
    // Export GeoJSON
    document.getElementById('btnExportGeoJSON').addEventListener('click', () => {
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

    // Export CSV
    document.getElementById('btnExportCSV').addEventListener('click', () => {
      const headers = ["object_id", "class", "confidence", "status", "risk", "lat", "lon", "length_m", "width_m"];
      const rows = this.targets.map(t => [
        t.object_id, t.class, t.calibrated_confidence || t.confidence,
        t.anomaly_status, t.risk_score, t.latitude || "", t.longitude || "",
        t.length_m || "", t.width_m || ""
      ]);
      const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
      this._downloadFile(csvContent, "survey_targets_summary.csv", "text/csv");
    });
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
