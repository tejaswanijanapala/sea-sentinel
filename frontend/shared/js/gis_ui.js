/**
 * Sea Sentinel: GIS UI & Spatial Intelligence Controller (Offline-Native)
 * Manages Level 1 (Current Input GIS) and Level 2 (Entire Ocean Map) interfaces,
 * workspace view switching, minimize/maximize viewport controls, and HITL workflows.
 */

class GISUIController {
  constructor() {
    this.currentReviewDecision = 'VERIFIED';
    this.reviewingTargetId = null;
    this._init();
  }

  _init() {
    // Helper to get Current Input GIS map instance
    const getGisMap = () => (window.gisMap || (window.app && window.app.map));
    window.getGisMap = getGisMap;

    // Helper to get Global Ocean GIS map instance
    const getEntireOceanMap = () => {
      if (!window.entireOceanMap && typeof window.GlobalOceanGISMap === 'function') {
        const el = document.getElementById('globalLeafletMap');
        if (el) {
          window.entireOceanMap = new window.GlobalOceanGISMap('globalLeafletMap');
        }
      }
      return window.entireOceanMap;
    };
    window.getEntireOceanMap = getEntireOceanMap;

    // 1. Workspace Tab Switching (Sonar Scan vs Split View vs Current Input GIS Map)
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
                m.invalidateSize();
                m.fitCurrentInput();
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
                m.invalidateSize();
                m.fitCurrentInput();
              }
            }, delay);
          });
        }
      }
    };

    if (tabWaterfall) tabWaterfall.onclick = () => switchView('waterfall');
    if (tabSplit) tabSplit.onclick = () => switchView('split');
    if (tabMap) tabMap.onclick = () => switchView('map');

    // 2. Navigation Sidebar & Header Triggers
    const navEntireOcean = document.getElementById('navItemEntireOcean') || document.getElementById('navEntireOcean');
    if (navEntireOcean) {
      navEntireOcean.onclick = (e) => {
        e.preventDefault();
        this.openEntireOceanMap({ maximize: false });
      };
    }

    const navCurrentGis = document.getElementById('navItemCurrentGis') || document.getElementById('navCurrentInputGis');
    if (navCurrentGis) {
      navCurrentGis.onclick = (e) => {
        e.preventDefault();
        switchView('map');
        const m = getGisMap();
        if (m) m.fitCurrentInput();
      };
    }

    const btnTopEntireOcean = document.getElementById('btnOpenEntireOceanTop');
    if (btnTopEntireOcean) {
      btnTopEntireOcean.onclick = (e) => {
        e.preventDefault();
        this.openEntireOceanMap({ maximize: false });
      };
    }

    const btnCloseEntireOcean = document.getElementById('btnCloseEntireOceanModal');
    if (btnCloseEntireOcean) {
      btnCloseEntireOcean.onclick = () => this.closeEntireOceanMap();
    }

    // Modal background click to close
    const modalEntireOcean = document.getElementById('entireOceanMapModal');
    if (modalEntireOcean) {
      modalEntireOcean.onclick = (e) => {
        if (e.target === modalEntireOcean) this.closeEntireOceanMap();
      };
    }

    // Layer checkboxes in Entire Ocean Modal
    const bindLayerToggle = (chkId, layerKey) => {
      const chk = document.getElementById(chkId);
      if (chk) {
        chk.addEventListener('change', (e) => {
          const oMap = getEntireOceanMap();
          if (oMap && oMap.layers && oMap.layers[layerKey]) {
            if (e.target.checked) {
              oMap.layers[layerKey].addTo(oMap.map);
            } else {
              oMap.map.removeLayer(oMap.layers[layerKey]);
            }
          }
          const lbl = chk.closest('.layer-toggle-btn');
          if (lbl) lbl.classList.toggle('active', e.target.checked);
        });
      }
    };

    bindLayerToggle('chkGlobalDebris', 'debrisMarkers');
    bindLayerToggle('chkGlobalClusters', 'clusters');
    bindLayerToggle('chkGlobalTracks', 'tracks');
    bindLayerToggle('chkGlobalSwaths', 'coverage');
    bindLayerToggle('chkGlobalReefs', 'coral_reefs');
    bindLayerToggle('chkGlobalMPAs', 'marine_protected_areas');
    bindLayerToggle('chkGlobalSeagrass', 'seagrass_meadows');
    bindLayerToggle('chkGlobalInfra', 'underwater_infrastructure');

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
      btnExportGis.onclick = async () => {
        const format = prompt("Choose GIS Export Format:\n1. geojson (Default)\n2. csv\n3. kml", "geojson");
        if (!format) return;
        const fmt = format.trim().toLowerCase();
        try {
          const res = await fetch(`http://localhost:8000/api/gis/export/${fmt}`);
          if (!res.ok) throw new Error("Export failed");
          const blob = await res.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `sea_sentinel_export_${Date.now()}.${fmt === 'csv' ? 'csv' : fmt === 'kml' ? 'kml' : 'geojson'}`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          window.URL.revokeObjectURL(url);
          if (window.app && window.app.showToast) {
            window.app.showToast({ type: "success", title: "Export Complete", message: `Spatial data exported as ${fmt.toUpperCase()}` });
          }
        } catch (e) {
          console.error("GIS Export error:", e);
        }
      };
    }

    // 5. Target Review Modal Handlers
    const btnCloseReview = document.getElementById('btnCloseTargetReviewModal');
    const modalReview = document.getElementById('gisTargetReviewModal');
    if (btnCloseReview && modalReview) {
      btnCloseReview.onclick = () => { modalReview.style.display = 'none'; };
    }

    document.querySelectorAll('.decision-pill-btn').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('.decision-pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentReviewDecision = btn.dataset.decision;
        const reclassGroup = document.getElementById('reclassifyClassGroup');
        if (reclassGroup) {
          reclassGroup.style.display = this.currentReviewDecision === 'RECLASSIFIED' ? 'block' : 'none';
        }
      };
    });

    const btnSubmitReview = document.getElementById('btnSubmitTargetReview');
    if (btnSubmitReview) {
      btnSubmitReview.onclick = () => this.saveTargetReview();
    }
  }

  // Open Entire Ocean Map Modal (Starts MINIMIZED by default; ADMIN ONLY)
  openEntireOceanMap(options = {}) {
    if (window.authManager && !window.authManager.isAdmin()) {
      if (window.app && window.app.showToast) {
        window.app.showToast("Admin Access Required: Entire Ocean Map is restricted to Administrators.", "warning");
      }
      const authModal = document.getElementById('roleAuthModal');
      if (authModal) authModal.style.display = 'flex';
      return;
    }

    const modal = document.getElementById('entireOceanMapModal');
    if (!modal) return;

    modal.style.display = 'flex';

    const getEntireOceanMap = window.getEntireOceanMap || (() => window.entireOceanMap);
    let oMap = getEntireOceanMap();

    // Default to minimized unless explicitly requested to maximize
    const shouldMaximize = Boolean(options.maximize || options.focusCurrent);

    if (oMap) {
      oMap.setMinimized(!shouldMaximize);
      oMap.loadDataset({ fit: shouldMaximize });
    }

    [50, 150, 300, 500].forEach(delay => {
      setTimeout(() => {
        if (!oMap) oMap = getEntireOceanMap();
        if (oMap && oMap.map) {
          if (shouldMaximize) {
            oMap.invalidateSize();
            if (options.focusCurrent) {
              if (window.gisMap && window.gisMap.currentInputMapState) {
                const st = window.gisMap.currentInputMapState;
                oMap.focusCurrentInput(st.detections, st.coordinates);
              }
              if (options.targetId) {
                oMap.focusTarget(options.targetId);
              }
            } else {
              oMap.fitAllBounds();
            }
          }
        }
      }, delay);
    });
  }

  closeEntireOceanMap() {
    const modal = document.getElementById('entireOceanMapModal');
    if (modal) modal.style.display = 'none';
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
        if (window.entireOceanMap) window.entireOceanMap.loadDataset();
      }
    } catch (e) {
      console.error("Save review error:", e);
    }
  }

  inspectTargetDetails(targetId) {
    if (window.entireOceanMap && !window.entireOceanMap.entireOceanMapState.isMinimized) {
      window.entireOceanMap.focusTarget(targetId);
    } else if (window.gisMap) {
      window.gisMap.focusTarget(targetId);
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.gisUI = new GISUIController();
  if (window.app) {
    window.app.openTargetReviewModal = (id) => window.gisUI.openTargetReviewModal(id);
    window.app.inspectTargetDetails = (id) => window.gisUI.inspectTargetDetails(id);
    window.app.openEntireOceanMap = (opts) => window.gisUI.openEntireOceanMap(opts);
  }
});
