/**
 * Sea Sentinel: GIS UI & Spatial Intelligence Controller (Offline-Native)
 * Handles workspace view switching, batch imports, HITL target reviews,
 * spatial search/filtering, and multi-format exports.
 */

class GISUIController {
  constructor() {
    this.currentReviewDecision = 'VERIFIED';
    this.reviewingTargetId = null;
    this._init();
  }

  _init() {
    const getGisMap = () => (window.gisMap || (window.app && window.app.map));
    window.getGisMap = getGisMap;

    // 1. Workspace Tab Switching (Sonar Scan vs Split View vs GIS Map)
    const tabWaterfall = document.getElementById('tabWaterfall');
    const tabSplit = document.getElementById('tabSplit');
    const tabMap = document.getElementById('tabMap');
    const cardWaterfall = document.getElementById('cardWaterfall') || document.querySelector('.waterfall-panel');
    const cardMap = document.getElementById('cardMap');

    const switchView = (mode) => {
      [tabWaterfall, tabSplit, tabMap].forEach(t => t && t.classList.remove('active'));
      if (mode === 'waterfall') {
        if (tabWaterfall) tabWaterfall.classList.add('active');
        if (cardWaterfall) cardWaterfall.style.display = 'block';
        if (cardMap) cardMap.style.display = 'none';
      } else if (mode === 'map') {
        if (tabMap) tabMap.classList.add('active');
        if (cardWaterfall) cardWaterfall.style.display = 'none';
        if (cardMap) {
          cardMap.style.display = 'block';
          [50, 150, 300].forEach(delay => {
            setTimeout(() => {
              const m = getGisMap();
              if (m && m.map) {
                m.map.invalidateSize();
                m.renderTargets();
                if (m.currentScanTargets && m.currentScanTargets.length > 0) {
                  m.fitCurrentScan();
                } else {
                  m.fitEntireOcean({ initialOnly: true });
                }
              }
            }, delay);
          });
        }
      } else if (mode === 'split') {
        if (tabSplit) tabSplit.classList.add('active');
        if (cardWaterfall) cardWaterfall.style.display = 'block';
        if (cardMap) {
          cardMap.style.display = 'block';
          [50, 150, 300].forEach(delay => {
            setTimeout(() => {
              const m = getGisMap();
              if (m && m.map) {
                m.map.invalidateSize();
                m.renderTargets();
                if (m.currentScanTargets && m.currentScanTargets.length > 0) {
                  m.fitCurrentScan();
                } else {
                  m.fitEntireOcean({ initialOnly: true });
                }
              }
            }, delay);
          });
        }
      }
    };

    if (tabWaterfall) tabWaterfall.onclick = () => switchView('waterfall');
    if (tabSplit) tabSplit.onclick = () => switchView('split');
    if (tabMap) tabMap.onclick = () => switchView('map');

    // 2. GIS Search & Filtering
    const searchInput = document.getElementById('gisSearchInput');
    const classFilter = document.getElementById('gisClassFilter');
    const confFilter = document.getElementById('gisConfFilter');
    const btnRecluster = document.getElementById('btnRecalcClusters');

    const applyGisFilters = () => {
      const gMap = getGisMap();
      if (!gMap) return;
      const q = (searchInput ? searchInput.value.trim().toLowerCase() : '');
      const cls = classFilter ? classFilter.value : 'all';
      const minConf = confFilter ? parseFloat(confFilter.value) : 0.0;

      gMap.filters.searchQuery = q;
      gMap.filters.classFilter = cls;
      gMap.filters.minConfidence = minConf;

      let filtered = gMap.allTargets.filter(t => {
        const tConf = Number(t.confidence || 0.8);
        if (tConf < minConf) return false;
        if (cls !== 'all' && (t.class_name || '').toLowerCase() !== cls.toLowerCase()) return false;
        if (q) {
          const tId = (t.target_id || '').toLowerCase();
          const tCls = (t.class_name || '').toLowerCase();
          const tClust = (t.cluster_id || '').toLowerCase();
          if (!tId.includes(q) && !tCls.includes(q) && !tClust.includes(q)) return false;
        }
        return true;
      });

      gMap.allTargets = filtered;
      gMap.renderTargets();
    };

    if (searchInput) searchInput.oninput = applyGisFilters;
    if (classFilter) classFilter.onchange = () => { const m = getGisMap(); if (m) m.loadGISDataset(); };
    if (confFilter) confFilter.onchange = () => { const m = getGisMap(); if (m) m.loadGISDataset(); };

    if (btnRecluster) {
      btnRecluster.onclick = async () => {
        try {
          btnRecluster.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Clustering...';
          const res = await fetch('http://localhost:8000/api/gis/cluster/recalculate?epsilon_meters=50.0&min_samples=2', { method: 'POST' });
          if (res.ok) {
            if (window.app && window.app.showToast) {
              window.app.showToast({ type: "success", title: "DBSCAN Re-clustered", message: "Spatial clusters updated successfully." });
            }
            const m = getGisMap();
            if (m) m.loadGISDataset();
          }
        } catch (e) {
          console.warn("Re-cluster error:", e);
        } finally {
          btnRecluster.innerHTML = '<i class="fa-solid fa-shapes"></i> Re-Cluster';
        }
      };
    }

    // 3. Batch Modal Controls
    const btnOpenBatch = document.getElementById('btnOpenBatchModal');
    const modalBatch = document.getElementById('gisBatchModal');
    const btnCloseBatch = document.getElementById('btnCloseBatchModal');

    if (btnOpenBatch && modalBatch) {
      btnOpenBatch.onclick = () => { modalBatch.style.display = 'flex'; };
    }
    if (btnCloseBatch && modalBatch) {
      btnCloseBatch.onclick = () => { modalBatch.style.display = 'none'; };
    }

    // 4. Export GIS Menu
    const btnExportGis = document.getElementById('btnExportGisMenu');
    if (btnExportGis) {
      btnExportGis.onclick = () => {
        const choice = prompt("Choose GIS Export Format:\n1. geojson (GeoJSON FeatureCollection)\n2. csv (Tabular Target Inventory)\n3. kml (3D Google Earth Placemarks)", "geojson");
        if (choice) {
          const fmt = choice.toLowerCase().trim();
          if (["geojson", "csv", "kml"].includes(fmt)) {
            window.open(`http://localhost:8000/api/gis/export?format=${fmt}&layer=all`, '_blank');
          }
        }
      };
    }

    // 5. Target Review Modal Controls
    const modalReview = document.getElementById('gisTargetReviewModal');
    const btnCloseReview = document.getElementById('btnCloseTargetReviewModal');
    const btnCancelReview = document.getElementById('btnCancelReviewModal');
    const btnSaveReview = document.getElementById('btnSaveTargetReview');

    if (btnCloseReview && modalReview) {
      btnCloseReview.onclick = () => { modalReview.style.display = 'none'; };
    }
    if (btnCancelReview && modalReview) {
      btnCancelReview.onclick = () => { modalReview.style.display = 'none'; };
    }
    if (btnSaveReview) {
      btnSaveReview.onclick = () => this.saveTargetReview();
    }

    // Review decision toggle buttons
    const btnActVerify = document.getElementById('btnActionVerify');
    const btnActReject = document.getElementById('btnActionReject');
    const btnActReclass = document.getElementById('btnActionReclassify');
    const reclassWrap = document.getElementById('reclassifySelectWrap');

    const setReviewAction = (action) => {
      this.currentReviewDecision = action;
      [btnActVerify, btnActReject, btnActReclass].forEach(b => b && b.classList.remove('active'));
      if (action === 'VERIFIED' && btnActVerify) btnActVerify.classList.add('active');
      else if (action === 'REJECTED' && btnActReject) btnActReject.classList.add('active');
      else if (action === 'RECLASSIFIED' && btnActReclass) btnActReclass.classList.add('active');
      if (reclassWrap) reclassWrap.style.display = action === 'RECLASSIFIED' ? 'block' : 'none';
    };

    // View Controls: Current Input vs Entire Ocean
    const btnFocusCurrent = document.getElementById('btnFocusCurrentScan');
    const btnFitOcean = document.getElementById('btnFitEntireOcean');
    const btnFitMap = document.getElementById('btnFitMap');
    const btnToggleSwath = document.getElementById('btnToggleSwath');

    if (btnFocusCurrent) {
      btnFocusCurrent.onclick = () => {
        const m = getGisMap();
        if (m) {
          m.fitCurrentScan();
          btnFocusCurrent.classList.add('active');
          if (btnFitOcean) btnFitOcean.classList.remove('active');
        }
      };
    }

    if (btnFitOcean) {
      btnFitOcean.onclick = () => {
        const m = getGisMap();
        if (m) {
          m.fitEntireOcean();
          btnFitOcean.classList.add('active');
          if (btnFocusCurrent) btnFocusCurrent.classList.remove('active');
        }
      };
    }

    if (btnFitMap) {
      btnFitMap.onclick = () => {
        const m = getGisMap();
        if (m) m.fitEntireOcean();
      };
    }

    if (btnToggleSwath) {
      btnToggleSwath.onclick = () => {
        const m = getGisMap();
        if (m && m.map && m.layers.coverage) {
          if (m.map.hasLayer(m.layers.coverage)) {
            m.map.removeLayer(m.layers.coverage);
            btnToggleSwath.classList.remove('active');
          } else {
            m.map.addLayer(m.layers.coverage);
            btnToggleSwath.classList.add('active');
          }
        }
      };
    }

    if (btnActVerify) btnActVerify.onclick = () => setReviewAction('VERIFIED');
    if (btnActReject) btnActReject.onclick = () => setReviewAction('REJECTED');
    if (btnActReclass) btnActReclass.onclick = () => setReviewAction('RECLASSIFIED');

    // Default to Split View on initialization
    setTimeout(() => {
      switchView('split');
    }, 200);
  }

  // Handle Batch File Ingestion
  async handleBatchFileSelect(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const progressBox = document.getElementById('batchProgressContainer');
    const progressLabel = document.getElementById('batchProgressLabel');
    const progressPercent = document.getElementById('batchProgressPercent');
    const progressBar = document.getElementById('batchProgressBar');
    const summaryBox = document.getElementById('batchResultsSummary');

    if (progressBox) progressBox.style.display = 'block';
    if (summaryBox) summaryBox.style.display = 'none';

    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const pct = Math.round(((i + 1) / files.length) * 100);
      if (progressLabel) progressLabel.textContent = `Processing ${i + 1} / ${files.length}: ${f.name}...`;
      if (progressPercent) progressPercent.textContent = `${pct}%`;
      if (progressBar) progressBar.style.width = `${pct}%`;

      try {
        const uploadRes = await window.apiService.uploadFile(f);
        if (uploadRes && uploadRes.saved_path) {
          await window.apiService.analyzeImage(uploadRes.saved_path, null, null, 1, 'balanced');
          successCount++;
        }
      } catch (err) {
        console.warn(`Batch file ${f.name} error:`, err);
        failCount++;
      }
    }

    if (summaryBox) {
      summaryBox.style.display = 'block';
      summaryBox.innerHTML = `
        <div style="font-weight:700; color:#10b981; margin-bottom:4px;"><i class="fa-solid fa-circle-check"></i> Batch Ingestion Complete</div>
        <div>Successfully Processed: <b>${successCount}</b> images</div>
        ${failCount > 0 ? `<div style="color:#ef4444;">Failed / Warnings: <b>${failCount}</b></div>` : ''}
        <div style="margin-top:6px; color:#38bdf8;">Persistent GIS database and debris map updated automatically.</div>
      `;
    }

    if (window.gisMap) window.gisMap.loadGISDataset();
    else if (window.app && window.app.map) window.app.map.loadGISDataset();
  }

  // Open Human-in-the-Loop Target Review Modal
  async openTargetReviewModal(targetId) {
    const modal = document.getElementById('gisTargetReviewModal');
    if (!modal) return;
    this.reviewingTargetId = targetId;

    try {
      const res = await fetch(`http://localhost:8000/api/gis/target/${targetId}`);
      if (!res.ok) throw new Error("Target not found");
      const data = await res.json();
      const target = data.target || {};

      const idEl = document.getElementById('reviewModalTargetId');
      if (idEl) idEl.textContent = target.target_id || targetId;

      const aiClassEl = document.getElementById('reviewModalAiClass');
      if (aiClassEl) aiClassEl.textContent = `${(target.class_name || 'debris').replace(/_/g, ' ').toUpperCase()} (${Math.round((target.confidence || 0.8) * 100)}%)`;

      const obsEl = document.getElementById('reviewModalObsCount');
      if (obsEl) obsEl.textContent = `${target.observation_count || 1} Observations`;

      const classSelect = document.getElementById('reviewModalClassSelect');
      if (classSelect) classSelect.value = (target.class_name || 'ghost_net').toLowerCase();

      const commentEl = document.getElementById('reviewModalComment');
      if (commentEl) commentEl.value = '';

      modal.style.display = 'flex';
    } catch (e) {
      console.warn("Target review modal error:", e);
    }
  }

  // Save Target Review Decision
  async saveTargetReview() {
    if (!this.reviewingTargetId) return;
    const targetId = this.reviewingTargetId;
    const modal = document.getElementById('gisTargetReviewModal');
    const classSelect = document.getElementById('reviewModalClassSelect');
    const commentEl = document.getElementById('reviewModalComment');

    const decision = this.currentReviewDecision || 'VERIFIED';
    const correctClass = decision === 'RECLASSIFIED' ? (classSelect ? classSelect.value : null) : null;
    const comments = commentEl ? commentEl.value.trim() : '';

    try {
      const payload = {
        reviewer_decision: decision,
        correct_class: correctClass,
        comments: comments,
        reviewer_name: 'Hydrographic Operator'
      };

      const res = await fetch(`http://localhost:8000/api/gis/target/${targetId}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        if (window.app && window.app.showToast) {
          window.app.showToast({
            type: "success",
            title: "Target Review Saved",
            message: `Target ${targetId} marked as ${decision}. Historical AI record preserved.`
          });
        }
        if (modal) modal.style.display = 'none';
        if (window.gisMap) window.gisMap.loadGISDataset();
        else if (window.app && window.app.map) window.app.map.loadGISDataset();
      }
    } catch (e) {
      console.error("Save review error:", e);
    }
  }

  inspectTargetDetails(targetId) {
    if (window.gisMap) window.gisMap.focusTarget(targetId);
    else if (window.app && window.app.map) window.app.map.focusTarget(targetId);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.gisUI = new GISUIController();
  // Bridge helper methods to window.app for popup button calls
  if (window.app) {
    window.app.openTargetReviewModal = (id) => window.gisUI.openTargetReviewModal(id);
    window.app.inspectTargetDetails = (id) => window.gisUI.inspectTargetDetails(id);
    window.app.handleBatchFileSelect = (e) => window.gisUI.handleBatchFileSelect(e);
  }
});
