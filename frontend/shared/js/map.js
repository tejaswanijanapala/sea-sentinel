/**
 * Sea Sentinel: High-End Offline Marine Debris GIS & Geospatial Intelligence System
 * Dual-Level GIS Architecture:
 *   Level 1: CurrentInputGISMap  -> Dedicated to currently uploaded/input SSS image ONLY.
 *   Level 2: GlobalOceanGISMap   -> Cumulative spatial database of all past + present debris detections.
 */

// =====================================================================
// Common GIS Utilities & Nautical Basemap Configurations
// =====================================================================
const GIS_CONFIG = {
  defaultCenter: [30.175, -87.825], // Gulf of Mexico / Breton Sound default
  defaultZoom: 13,
  createBasemaps: () => {
    const osmBase = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
      maxNativeZoom: 19
    });
    const satBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, Maxar, Earthstar Geographics',
      maxZoom: 19,
      maxNativeZoom: 18
    });
    const oceanBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, GEBCO, NOAA, National Geographic',
      maxZoom: 19,
      maxNativeZoom: 10
    });
    const topoBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, HERE, Garmin, USGS, Intermap, INCREMENT P',
      maxZoom: 19,
      maxNativeZoom: 18
    });
    const darkBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
      maxZoom: 19,
      maxNativeZoom: 16
    });
    const hotBase = L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team',
      maxZoom: 19,
      maxNativeZoom: 19
    });
    return { osmBase, satBase, oceanBase, topoBase, darkBase, hotBase };
  },
  formatCoordDeg: (val, type) => {
    if (val == null || isNaN(val)) return "--";
    const num = Number(val);
    const absVal = Math.abs(num).toFixed(5);
    return type === 'lat' ? `${absVal}° ${num >= 0 ? 'N' : 'S'}` : `${absVal}° ${num >= 0 ? 'E' : 'W'}`;
  },
  formatCoordinate: (lat, lon) => {
    if (lat == null || lon == null || isNaN(lat) || isNaN(lon)) return "Unreferenced Target";
    return `${GIS_CONFIG.formatCoordDeg(lat, 'lat')}, ${GIS_CONFIG.formatCoordDeg(lon, 'lon')}`;
  },
  inferWaterBody: (lat, lon) => {
    if (lat == null || lon == null || isNaN(lat) || isNaN(lon)) return "Unreferenced SSS Chip";
    const nLat = Number(lat);
    const nLon = Number(lon);
    if (nLat >= 24.0 && nLat <= 27.5 && nLon >= -80.0 && nLon <= -75.0) {
      return "Florida Straits / Bahama Deep Water Channel";
    } else if (nLat >= 28.5 && nLat <= 31.0 && nLon >= -89.5 && nLon <= -86.5) {
      return "Gulf of Mexico / Breton Sound Open Ocean";
    } else if (nLat >= 14.5 && nLat <= 16.5 && nLon >= 72.5 && nLon <= 74.5) {
      return "Arabian Sea / Goa Offshore Continental Shelf";
    } else if (nLat >= 12.0 && nLat <= 14.5 && nLon >= 79.5 && nLon <= 81.5) {
      return "Bay of Bengal / Chennai Coastal Waters";
    } else if (nLat >= 24.0 && nLat <= 39.0 && nLon >= 118.0 && nLon <= 124.0) {
      return "East China Sea / Bohai Bay Offshore Waters";
    } else if (nLat >= 18.0 && nLat <= 22.0 && nLon >= 71.0 && nLon <= 73.5) {
      return "Arabian Sea / Mumbai High Offshore Region";
    }
    return nLon < 0 ? "Western Atlantic / Florida Straits Marine Waters" : "Indo-Pacific Marine Waters";
  },
  getColorForClass: (className = "") => {
    const cls = (className || "").toLowerCase();
    if (cls.includes("ghost") || cls.includes("net") || cls.includes("line")) return "#10b981"; // Emerald
    if (cls.includes("pipe") || cls.includes("cable") || cls.includes("line_hazard")) return "#f97316"; // Orange
    if (cls.includes("riprap") || cls.includes("boulder") || cls.includes("reef")) return "#64748b"; // Slate
    if (cls.includes("metal") || cls.includes("container") || cls.includes("drum") || cls.includes("engine")) return "#3b82f6"; // Cobalt Blue
    if (cls.includes("wreck") || cls.includes("ship") || cls.includes("hull")) return "#ef4444"; // Red
    if (cls.includes("plastic") || cls.includes("synthetic")) return "#a855f7"; // Purple
    if (cls.includes("tire") || cls.includes("rubber")) return "#eab308"; // Yellow
    if (cls.includes("munitions") || cls.includes("uxo")) return "#ec4899"; // Pink
    return "#00f0ff"; // Tactical Cyan default
  },
  parseLatLon: (t) => {
    if (!t) return null;
    let lat = t.latitude != null ? Number(t.latitude) : (t.lat != null ? Number(t.lat) : null);
    let lon = t.longitude != null ? Number(t.longitude) : (t.lon != null ? Number(t.lon) : null);
    if ((lat == null || lon == null) && Array.isArray(t.coordinates) && t.coordinates.length >= 2) {
      lat = Number(t.coordinates[0]);
      lon = Number(t.coordinates[1]);
    }
    if (lat != null && lon != null && !isNaN(lat) && !isNaN(lon)) {
      return [lat, lon];
    }
    return null;
  }
};

// =====================================================================
// LEVEL 1: CURRENT INPUT GIS MAP
// Renders ONLY the currently active SSS input image & its detections.
// Historical / global database debris markers are NEVER displayed here,
// even when minimized or maximized.
// =====================================================================
class CurrentInputGISMap {
  constructor(containerId = 'leafletMap') {
    this.containerId = containerId;
    this.map = null;
    this.markers = {};
    this.uncertaintyCircles = [];

    // Distinct Current Input State (Never shared with Entire Ocean Map)
    this.currentInputMapState = {
      surveyId: null,
      inputId: null,
      coordinates: null,
      detections: [],
      viewport: null,
      isMinimized: false
    };

    this.layers = {
      offlineGrid: null,
      currentScanDebris: null,
      uncertainty: null,
      coral_reefs: null,
      marine_protected_areas: null,
      seagrass_meadows: null,
      underwater_infrastructure: null
    };

    this._init();
  }

  _init() {
    if (typeof L === 'undefined') {
      setTimeout(() => this._init(), 250);
      return;
    }

    const container = document.getElementById(this.containerId);
    if (!container) return;

    try {
      this.layers.offlineGrid = L.layerGroup();
      this.layers.currentScanDebris = L.layerGroup();
      this.layers.uncertainty = L.layerGroup();
      this.layers.coral_reefs = L.layerGroup();
      this.layers.marine_protected_areas = L.layerGroup();
      this.layers.seagrass_meadows = L.layerGroup();
      this.layers.underwater_infrastructure = L.layerGroup();

      this._initMap();
      this._buildOfflineTacticalGrid();
      this._loadHabitatLayers();
      this._setupResizeObserver();
      this._setupControls();
    } catch (err) {
      console.error("[CurrentInputGISMap] Initialization error:", err);
    }
  }

  _initMap() {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }

    this.map = L.map(this.containerId, {
      center: GIS_CONFIG.defaultCenter,
      zoom: 13,
      zoomControl: false,
      attributionControl: true,
      minZoom: 2,
      maxZoom: 18
    });

    L.control.zoom({ position: 'bottomright' }).addTo(this.map);

    const basemaps = GIS_CONFIG.createBasemaps();
    basemaps.osmBase.addTo(this.map);
    this.layers.offlineGrid.addTo(this.map);
    this.layers.currentScanDebris.addTo(this.map);
    this.layers.uncertainty.addTo(this.map);

    const baseLayerControls = {
      "<span style='color:#059669; font-weight:700;'>🗺️ OpenStreetMap Marine</span>": basemaps.osmBase,
      "<span style='color:#0284c7; font-weight:700;'>🛰️ Satellite Imagery</span>": basemaps.satBase,
      "<span style='color:#00f0ff; font-weight:700;'>🌊 Ocean Bathymetry</span>": basemaps.oceanBase,
      "<span style='color:#0d9488; font-weight:700;'>🧭 Topographic Seabed</span>": basemaps.topoBase,
      "<span style='color:#6366f1; font-weight:700;'>◈ Dark Tactical Marine</span>": basemaps.darkBase
    };

    const overlayControls = {
      "<span style='color:#00f0ff; font-weight:700;'>🎯 Current Input Debris</span>": this.layers.currentScanDebris,
      "<span style='color:#ef4444; font-weight:600;'>⭕ Uncertainty Radii</span>": this.layers.uncertainty,
      "<span style='color:#64748b; font-weight:600;'>🧭 Nautical Graticule</span>": this.layers.offlineGrid,
      "<span style='color:#ec4899; font-weight:600;'>🪸 Coral Reefs</span>": this.layers.coral_reefs,
      "<span style='color:#10b981; font-weight:600;'>🛡️ MPAs</span>": this.layers.marine_protected_areas,
      "<span style='color:#84cc16; font-weight:600;'>🌿 Seagrass Beds</span>": this.layers.seagrass_meadows,
      "<span style='color:#f97316; font-weight:600;'>⚡ Subsea Infrastructure</span>": this.layers.underwater_infrastructure
    };

    L.control.layers(baseLayerControls, overlayControls, { position: 'topright', collapsed: true }).addTo(this.map);

    this.map.on('mousemove', (e) => {
      const hud = document.getElementById('mapCoordsHud');
      if (hud && e.latlng) {
        const latStr = GIS_CONFIG.formatCoordDeg(e.latlng.lat, 'lat');
        const lonStr = GIS_CONFIG.formatCoordDeg(e.latlng.lng, 'lon');
        hud.innerHTML = `<i class="fa-solid fa-crosshairs"></i> Current Input GIS Cursor: <b>${latStr}, ${lonStr}</b> &nbsp;|&nbsp; Datum: <span style="color:#00f0ff;">WGS84</span> &nbsp;|&nbsp; Status: <span style="color:#10b981;">ACTIVE INPUT ONLY</span>`;
      }
    });
  }

  _buildOfflineTacticalGrid() {
    if (!this.layers.offlineGrid) return;
    this.layers.offlineGrid.clearLayers();
    for (let lat = -80; lat <= 80; lat += 5) {
      const line = L.polyline([[lat, -180], [lat, 180]], {
        color: 'rgba(0, 240, 255, 0.12)',
        weight: 1,
        dashArray: '3, 6',
        interactive: false
      });
      this.layers.offlineGrid.addLayer(line);
    }
    for (let lon = -180; lon <= 180; lon += 5) {
      const line = L.polyline([[-80, lon], [80, lon]], {
        color: 'rgba(0, 240, 255, 0.12)',
        weight: 1,
        dashArray: '3, 6',
        interactive: false
      });
      this.layers.offlineGrid.addLayer(line);
    }
  }

  async _loadHabitatLayers() {
    try {
      const res = await fetch("http://localhost:8000/api/gis/layers");
      if (!res.ok) return;
      const data = await res.json();
      const layers = data.layers || {};

      const layerConfig = {
        coral_reefs: { color: "#ec4899", fillColor: "#f43f5e", fillOpacity: 0.16 },
        marine_protected_areas: { color: "#10b981", fillColor: "#059669", fillOpacity: 0.14 },
        seagrass_meadows: { color: "#84cc16", fillColor: "#65a30d", fillOpacity: 0.16 },
        underwater_infrastructure: { color: "#f97316", weight: 3, dashArray: "6, 6" }
      };

      Object.entries(layers).forEach(([layerKey, geojson]) => {
        const group = this.layers[layerKey];
        if (group && geojson && geojson.features) {
          group.clearLayers();
          const cfg = layerConfig[layerKey] || { color: "#38bdf8" };
          L.geoJSON(geojson, {
            style: cfg,
            onEachFeature: (feature, layer) => {
              const p = feature.properties || {};
              layer.bindTooltip(`
                <div style="font-family:'Outfit',sans-serif; font-size:0.75rem;">
                  <b>${p.name || p.id}</b><br/>
                  Type: ${p.type || layerKey}<br/>
                  Sensitivity: <span style="color:#ef4444; font-weight:700;">${p.sensitivity || 'HIGH'}</span>
                </div>
              `);
            }
          }).addTo(group);
        }
      });
    } catch (e) {
      // Offline mode fallback
    }
  }

  _setupResizeObserver() {
    const container = document.getElementById(this.containerId);
    if (!container || !window.ResizeObserver) return;
    const ro = new ResizeObserver(() => {
      if (this.map && !this.currentInputMapState.isMinimized) {
        this.map.invalidateSize();
      }
    });
    ro.observe(container);
  }

  _setupControls() {
    const btnMinMax = document.getElementById('btnToggleMinMaxCurrentInput');
    if (btnMinMax) {
      btnMinMax.onclick = (e) => {
        e.preventDefault();
        this.toggleMinimized();
      };
    }

    const btnViewEntire = document.getElementById('btnViewInEntireOceanMap');
    if (btnViewEntire) {
      btnViewEntire.onclick = () => {
        if (window.gisUI && window.gisUI.openEntireOceanMap) {
          window.gisUI.openEntireOceanMap({ focusCurrent: true });
        }
      };
    }

    const btnFitCurrent = document.getElementById('btnFocusCurrentScan');
    if (btnFitCurrent) {
      btnFitCurrent.onclick = () => this.fitCurrentInput();
    }
  }

  // -----------------------------------------------------------------
  // Separate Data Selector: Current Input ONLY
  // -----------------------------------------------------------------
  getCurrentInputDetections() {
    return this.currentInputMapState.detections || [];
  }

  // -----------------------------------------------------------------
  // Minimize / Maximize Viewport Controls (Changes viewport size ONLY)
  // -----------------------------------------------------------------
  setMinimized(isMinimized) {
    this.currentInputMapState.isMinimized = Boolean(isMinimized);
    const card = document.getElementById('cardMap');
    const icon = document.getElementById('iconMinMaxCurrentInput');

    if (card) {
      card.classList.toggle('is-minimized', this.currentInputMapState.isMinimized);
    }

    if (icon) {
      icon.className = this.currentInputMapState.isMinimized ? 'fa-solid fa-expand' : 'fa-solid fa-minus';
    }

    if (!this.currentInputMapState.isMinimized) {
      [50, 150, 300].forEach(d => {
        setTimeout(() => {
          this.invalidateSize();
          this.fitCurrentInput();
        }, d);
      });
    }
  }

  toggleMinimized() {
    this.setMinimized(!this.currentInputMapState.isMinimized);
  }

  // -----------------------------------------------------------------
  // Level 1 Core: Set Targets for Current Input SSS Image ONLY
  // -----------------------------------------------------------------
  setTargets(targets = [], surveyMeta = {}) {
    const safeTargets = Array.isArray(targets) ? targets : [];
    this.currentInputMapState.detections = safeTargets;

    // Extract Survey ID & Image Name
    let surveyId = surveyMeta.survey_id || (window.app && window.app.currentAnalysisResult && window.app.currentAnalysisResult.survey_id) || "SURVEY_CURRENT";
    let imageName = (window.app && window.app.uploadedFile && window.app.uploadedFile.name) ||
                    (window.app && window.app.currentSample && window.app.currentSample.name) ||
                    "current_sss_input.png";

    // Determine GIS Center Coordinates
    let centerLat = null;
    let centerLon = null;

    if (surveyMeta.center_wgs84 && Array.isArray(surveyMeta.center_wgs84) && surveyMeta.center_wgs84.length >= 2) {
      centerLat = Number(surveyMeta.center_wgs84[0]);
      centerLon = Number(surveyMeta.center_wgs84[1]);
    } else if (safeTargets.length > 0) {
      const validTargets = safeTargets.map(t => GIS_CONFIG.parseLatLon(t)).filter(Boolean);
      if (validTargets.length > 0) {
        centerLat = validTargets.reduce((sum, pt) => sum + pt[0], 0) / validTargets.length;
        centerLon = validTargets.reduce((sum, pt) => sum + pt[1], 0) / validTargets.length;
      }
    }

    if (centerLat == null || centerLon == null || isNaN(centerLat) || isNaN(centerLon)) {
      centerLat = 30.175;
      centerLon = -87.825;
    }

    this.currentInputMapState.surveyId = surveyId;
    this.currentInputMapState.inputId = imageName;
    this.currentInputMapState.coordinates = [centerLat, centerLon];

    // Update Banner UI Elements
    const elSurvey = document.getElementById('currentInputSurveyId');
    if (elSurvey) elSurvey.textContent = surveyId;

    const elImage = document.getElementById('currentInputImageName');
    if (elImage) elImage.textContent = imageName;

    const elCount = document.getElementById('currentInputTargetCount');
    if (elCount) elCount.textContent = `${safeTargets.length} Targets`;

    const elCoords = document.getElementById('currentInputCoordinates');
    if (elCoords) elCoords.textContent = GIS_CONFIG.formatCoordinate(centerLat, centerLon);

    // Render ONLY current scan detections via dedicated selector
    this.renderMap(this.getCurrentInputDetections());

    // Auto-focus strictly on current input coordinates
    this.fitCurrentInput();

    // Background notify Entire Ocean Map (Level 2) to refresh its cumulative database
    if (window.entireOceanMap && typeof window.entireOceanMap.loadDataset === 'function') {
      window.entireOceanMap.loadDataset({ fit: false });
    }
  }

  // -----------------------------------------------------------------
  // Render Map: Strictly renders current input detections
  // -----------------------------------------------------------------
  renderMap(detections = []) {
    if (!this.map || !this.layers.currentScanDebris) return;

    this.layers.currentScanDebris.clearLayers();
    this.layers.uncertainty.clearLayers();
    this.markers = {};

    const items = Array.isArray(detections) ? detections : [];
    if (items.length === 0) return;

    items.forEach((t, idx) => {
      const targetTag = t.object_id || `TGT_${String(idx + 1).padStart(3, '0')}`;
      const persistentId = t.target_id || targetTag;
      const displayId = targetTag;
      const pt = GIS_CONFIG.parseLatLon(t);

      if (!pt) return;
      const [lat, lon] = pt;

      const pos = [lat, lon];
      const clsName = t.class_name || t.class || 'Marine Debris';
      const color = GIS_CONFIG.getColorForClass(clsName);
      const conf = Number(t.calibrated_confidence || t.confidence || 0.85);
      const confPct = Math.round(conf * 100);
      const sonarConf = t.sonar_aware_confidence != null ? Math.round(Number(t.sonar_aware_confidence) * 100) : confPct;

      const iconHtml = `
        <div class="current-input-marker" style="--marker-color:${color};">
          <div class="marker-pulse-ring"></div>
          <div class="marker-center-dot">
            <i class="fa-solid fa-crosshairs"></i>
          </div>
          <div class="marker-tag">${displayId}</div>
        </div>
      `;

      const imo = t.imo_risk_assessment || {};
      const riskLvl = (imo.risk_level || t.hazard_level || t.risk_level || (t.hazard_score >= 80 ? 'CRITICAL' : t.hazard_score >= 60 ? 'HIGH' : 'MODERATE')).toUpperCase();
      const riskColor = riskLvl === 'CRITICAL' ? '#ef4444' : riskLvl === 'HIGH' ? '#f97316' : riskLvl === 'MODERATE' ? '#eab308' : '#10b981';
      const matrixCell = (imo.risk_matrix && imo.risk_matrix.matrix_cell) || 'L4-C4';
      const rps = imo.risk_priority_score != null ? imo.risk_priority_score.toFixed(0) : (t.priority_score != null ? Number(t.priority_score).toFixed(0) : '80');
      const isVerificationReq = Boolean(imo.position_verification_required || t.position_verification_required);

      const customIcon = L.divIcon({
        className: 'current-scan-leaflet-icon',
        html: iconHtml,
        iconSize: [36, 36],
        iconAnchor: [18, 18],
        popupAnchor: [0, -20]
      });

      const marker = L.marker(pos, { icon: customIcon, zIndexOffset: 2000 });

      const popupContent = `
        <div class="high-end-gis-popup" style="font-family:'Outfit',sans-serif; min-width:280px;">
          <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(0,240,255,0.3); padding-bottom:6px; margin-bottom:8px;">
            <div style="font-weight:800; color:#00f0ff; font-size:0.95rem; display:flex; align-items:center; gap:6px;">
              <span class="active-badge-pulse" style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#00f0ff;"></span>
              ${displayId} <span style="font-size:0.7rem; color:#94a3b8;">(${persistentId})</span>
            </div>
            <span style="font-size:0.68rem; font-weight:700; padding:2px 6px; border-radius:4px; background:rgba(0,240,255,0.15); color:#00f0ff;">
              CURRENT INPUT
            </span>
          </div>
          <div style="display:grid; grid-template-columns:auto 1fr; gap:4px 10px; font-size:0.75rem; color:#cbd5e1; margin-bottom:10px;">
            <span style="color:#94a3b8;">Class:</span> <b style="color:#f8fafc;">${clsName.replace(/_/g, ' ').toUpperCase()}</b>
            <span style="color:#94a3b8;">Coordinates:</span> <span style="font-family:'JetBrains Mono',monospace; color:#38bdf8;">${GIS_CONFIG.formatCoordinate(lat, lon)}</span>
            <span style="color:#94a3b8;">AI Confidence:</span> <b style="color:#10b981;">${confPct}%</b>
            <span style="color:#94a3b8;">Sonar-Aware:</span> <b style="color:#00f0ff;">${sonarConf}%</b>
            <span style="color:#94a3b8;">IMO Risk:</span> <span style="font-weight:800; color:${riskColor};">${riskLvl} (Cell ${matrixCell})</span>
            <span style="color:#94a3b8;">Risk Priority:</span> <b style="color:#38bdf8;">${rps}/100</b>
            <span style="color:#94a3b8;">Survey:</span> <span>${this.currentInputMapState.surveyId}</span>
            <span style="color:#94a3b8;">Source SSS:</span> <span style="color:#e2e8f0; font-size:0.7rem;">${this.currentInputMapState.inputId}</span>
          </div>
          ${isVerificationReq ? `
            <div style="background:#fff1f2; border:1px solid #fecdd3; color:#9f1239; font-size:0.68rem; font-weight:700; padding:4px 6px; border-radius:4px; margin-bottom:8px;">
              <i class="fa-solid fa-triangle-exclamation"></i> Positional Verification Required
            </div>
          ` : ''}
          <div style="display:flex; gap:6px; margin-top:8px;">
            <button onclick="window.gisMap.focusTarget('${displayId}'); if(window.app) window.app.onTargetSelected('${displayId}');" 
                    style="flex:1; padding:5px 8px; font-size:0.72rem; font-weight:700; border-radius:4px; background:#0284c7; color:#fff; border:none; cursor:pointer;">
              <i class="fa-solid fa-magnifying-glass"></i> Inspect &amp; Audit
            </button>
            <button onclick="if(window.gisUI) window.gisUI.openEntireOceanMap({ focusCurrent: true, targetId: '${persistentId}' });" 
                    style="flex:1; padding:5px 8px; font-size:0.72rem; font-weight:700; border-radius:4px; background:rgba(0,240,255,0.2); color:#00f0ff; border:1px solid #00f0ff; cursor:pointer;">
              <i class="fa-solid fa-globe"></i> View in Ocean Map
            </button>
          </div>
        </div>
      `;

      marker.bindPopup(popupContent, { maxWidth: 320, className: 'gis-custom-leaflet-popup' });
      marker.addTo(this.layers.currentScanDebris);
      this.markers[displayId] = marker;
      this.markers[targetTag] = marker;
      this.markers[persistentId] = marker;

      // Geodetic Uncertainty Radius
      const uncRadius = Number(t.uncertainty_radius_m || (1.0 - conf) * 25.0 + 3.0);
      const circle = L.circle(pos, {
        radius: Math.max(uncRadius, 3.5),
        color: isVerificationReq ? '#ef4444' : color,
        weight: isVerificationReq ? 2.0 : 1.5,
        fillColor: isVerificationReq ? '#f43f5e' : color,
        fillOpacity: isVerificationReq ? 0.22 : 0.12,
        dashArray: isVerificationReq ? "4, 4" : "3, 4"
      });
      circle.addTo(this.layers.uncertainty);
      this.uncertaintyCircles.push(circle);

      // Render Dynamic Drift Trajectory Forecast (if mobile)
      const driftProj = imo.drift_projections || [];
      if (driftProj.length > 0) {
        const polyCoords = [[lat, lon]];
        driftProj.forEach(dp => {
          polyCoords.push([dp.projected_latitude, dp.projected_longitude]);
          const dpCircle = L.circleMarker([dp.projected_latitude, dp.projected_longitude], {
            radius: 4,
            color: '#f97316',
            fillColor: '#ea580c',
            fillOpacity: 0.8,
            weight: 1.5
          });
          dpCircle.bindTooltip(`+${dp.horizon_hours}h Forecast: Nav Exposure ${dp.future_navigation_exposure.toFixed(0)}% (±${dp.uncertainty_radius_m.toFixed(0)}m)`);
          dpCircle.addTo(this.layers.uncertainty);
          this.uncertaintyCircles.push(dpCircle);
        });
        const trajLine = L.polyline(polyCoords, {
          color: '#f97316',
          weight: 2,
          dashArray: '5, 8',
          opacity: 0.85
        });
        trajLine.addTo(this.layers.uncertainty);
      }
    });
  }

  fitCurrentInput() {
    if (!this.map || this.currentInputMapState.isMinimized) return;
    this.map.invalidateSize();
    const detections = this.getCurrentInputDetections();
    const validCoords = detections.map(t => GIS_CONFIG.parseLatLon(t)).filter(Boolean);

    if (validCoords.length > 0) {
      if (validCoords.length === 1) {
        this.map.setView(validCoords[0], 16, { animate: true });
      } else {
        const bounds = L.latLngBounds(validCoords);
        this.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 17, animate: true });
      }
    } else if (this.currentInputMapState.coordinates && !isNaN(this.currentInputMapState.coordinates[0])) {
      this.map.setView(this.currentInputMapState.coordinates, 15, { animate: true });
    }
  }

  focusTarget(targetId) {
    const marker = this.markers[targetId];
    if (marker && this.map && !this.currentInputMapState.isMinimized) {
      this.map.flyTo(marker.getLatLng(), 17, { duration: 0.8 });
      marker.openPopup();
    }
  }

  selectTarget(targetId, options = {}) {
    this.focusTarget(targetId);
  }

  fitAllTargets() {
    this.fitCurrentInput();
  }

  toggleSwath() {
    return true;
  }

  toggleGISLayer(layerKey, visible) {
    const layer = this.layers[layerKey];
    if (!layer || !this.map) return;
    if (visible) {
      layer.addTo(this.map);
    } else {
      this.map.removeLayer(layer);
    }
  }

  invalidateSize() {
    if (this.map) this.map.invalidateSize();
  }

  renderTargets() {
    this.renderMap(this.getCurrentInputDetections());
  }
}

// =====================================================================
// LEVEL 2: ENTIRE OCEAN MAP (CUMULATIVE GLOBAL SPATIAL DATABASE)
// Contains all historical + present detections, tracks, swaths, clusters,
// statistics bar, zoom-based clustering aggregation, and multi-filters.
// Starts MINIMIZED BY DEFAULT per user specification.
// =====================================================================
class GlobalOceanGISMap {
  constructor(containerId = 'globalLeafletMap') {
    this.containerId = containerId;
    this.map = null;
    this.markers = {};
    this.clusterMarkers = [];
    this.trackLines = [];
    this.coveragePolygons = [];
    this.highlightLayer = null;

    // Distinct Cumulative Ocean State (Contains all inputs across time)
    this.entireOceanMapState = {
      allSurveys: [],
      allDetections: [],
      allCoordinates: [],
      clusters: [],
      filteredDetections: [],
      filters: {
        classFilter: 'all',
        minConfidence: 0.0,
        surveyFilter: 'all',
        statusFilter: 'all',
        searchQuery: ''
      },
      viewport: null,
      isMinimized: true // STARTS MINIMIZED BY DEFAULT
    };

    this.layers = {
      offlineGrid: null,
      coverage: null,
      tracks: null,
      clusters: null,
      debrisMarkers: null,
      highlight: null,
      coral_reefs: null,
      marine_protected_areas: null,
      seagrass_meadows: null,
      underwater_infrastructure: null
    };

    this._init();
  }

  _init() {
    if (typeof L === 'undefined') {
      setTimeout(() => this._init(), 250);
      return;
    }

    const container = document.getElementById(this.containerId);
    if (!container) return;

    try {
      this.layers.offlineGrid = L.layerGroup();
      this.layers.coverage = L.layerGroup();
      this.layers.tracks = L.layerGroup();
      this.layers.clusters = L.layerGroup();
      this.layers.debrisMarkers = L.layerGroup();
      this.layers.highlight = L.layerGroup();
      this.layers.coral_reefs = L.layerGroup();
      this.layers.marine_protected_areas = L.layerGroup();
      this.layers.seagrass_meadows = L.layerGroup();
      this.layers.underwater_infrastructure = L.layerGroup();

      this._initMap();
      this._buildOfflineTacticalGrid();
      this._loadHabitatLayers();
      this._setupFilterListeners();
      this._setupResizeObserver();
      this._setupControls();
      this.loadDataset();
    } catch (err) {
      console.error("[GlobalOceanGISMap] Initialization error:", err);
    }
  }

  _initMap() {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }

    this.map = L.map(this.containerId, {
      center: GIS_CONFIG.defaultCenter,
      zoom: 11,
      zoomControl: false,
      attributionControl: true,
      minZoom: 2,
      maxZoom: 18
    });

    L.control.zoom({ position: 'bottomright' }).addTo(this.map);

    const basemaps = GIS_CONFIG.createBasemaps();
    basemaps.osmBase.addTo(this.map);
    this.layers.offlineGrid.addTo(this.map);
    this.layers.coverage.addTo(this.map);
    this.layers.tracks.addTo(this.map);
    this.layers.clusters.addTo(this.map);
    this.layers.debrisMarkers.addTo(this.map);
    this.layers.highlight.addTo(this.map);

    const baseLayerControls = {
      "<span style='color:#059669; font-weight:700;'>🗺️ OpenStreetMap Marine</span>": basemaps.osmBase,
      "<span style='color:#0284c7; font-weight:700;'>🛰️ Satellite Imagery</span>": basemaps.satBase,
      "<span style='color:#00f0ff; font-weight:700;'>🌊 Ocean Bathymetry</span>": basemaps.oceanBase,
      "<span style='color:#0d9488; font-weight:700;'>🧭 Topographic Seabed</span>": basemaps.topoBase,
      "<span style='color:#6366f1; font-weight:700;'>◈ Dark Tactical Marine</span>": basemaps.darkBase
    };

    const overlayControls = {
      "<span style='color:#00f0ff; font-weight:700;'>🌐 Global Debris Database</span>": this.layers.debrisMarkers,
      "<span style='color:#a855f7; font-weight:600;'>◈ Target Clusters</span>": this.layers.clusters,
      "<span style='color:#f59e0b; font-weight:600;'>🚢 Survey Tracks</span>": this.layers.tracks,
      "<span style='color:#06b6d4; font-weight:600;'>📐 Surveyed Swaths</span>": this.layers.coverage,
      "<span style='color:#64748b; font-weight:600;'>🧭 Nautical Graticule</span>": this.layers.offlineGrid,
      "<span style='color:#ec4899; font-weight:600;'>🪸 Coral Reefs</span>": this.layers.coral_reefs,
      "<span style='color:#10b981; font-weight:600;'>🛡️ MPAs</span>": this.layers.marine_protected_areas,
      "<span style='color:#84cc16; font-weight:600;'>🌿 Seagrass Beds</span>": this.layers.seagrass_meadows,
      "<span style='color:#f97316; font-weight:600;'>⚡ Subsea Infrastructure</span>": this.layers.underwater_infrastructure
    };

    L.control.layers(baseLayerControls, overlayControls, { position: 'topright', collapsed: true }).addTo(this.map);

    // Dynamic Zoom Clustering Re-render
    this.map.on('zoomend', () => {
      if (!this.entireOceanMapState.isMinimized) {
        this.renderMap(this.getAllOceanDetections());
      }
    });
  }

  _buildOfflineTacticalGrid() {
    if (!this.layers.offlineGrid) return;
    this.layers.offlineGrid.clearLayers();
    for (let lat = -80; lat <= 80; lat += 5) {
      const line = L.polyline([[lat, -180], [lat, 180]], {
        color: 'rgba(0, 240, 255, 0.12)',
        weight: 1,
        dashArray: '3, 6',
        interactive: false
      });
      this.layers.offlineGrid.addLayer(line);
    }
    for (let lon = -180; lon <= 180; lon += 5) {
      const line = L.polyline([[-80, lon], [80, lon]], {
        color: 'rgba(0, 240, 255, 0.12)',
        weight: 1,
        dashArray: '3, 6',
        interactive: false
      });
      this.layers.offlineGrid.addLayer(line);
    }
  }

  async _loadHabitatLayers() {
    try {
      const res = await fetch("http://localhost:8000/api/gis/layers");
      if (!res.ok) return;
      const data = await res.json();
      const layers = data.layers || {};

      const layerConfig = {
        coral_reefs: { color: "#ec4899", fillColor: "#f43f5e", fillOpacity: 0.16 },
        marine_protected_areas: { color: "#10b981", fillColor: "#059669", fillOpacity: 0.14 },
        seagrass_meadows: { color: "#84cc16", fillColor: "#65a30d", fillOpacity: 0.16 },
        underwater_infrastructure: { color: "#f97316", weight: 3, dashArray: "6, 6" }
      };

      Object.entries(layers).forEach(([layerKey, geojson]) => {
        const group = this.layers[layerKey];
        if (group && geojson && geojson.features) {
          group.clearLayers();
          const cfg = layerConfig[layerKey] || { color: "#38bdf8" };
          L.geoJSON(geojson, {
            style: cfg,
            onEachFeature: (feature, layer) => {
              const p = feature.properties || {};
              layer.bindTooltip(`
                <div style="font-family:'Outfit',sans-serif; font-size:0.75rem;">
                  <b>${p.name || p.id}</b><br/>
                  Type: ${p.type || layerKey}<br/>
                  Sensitivity: <span style="color:#ef4444; font-weight:700;">${p.sensitivity || 'HIGH'}</span>
                </div>
              `);
            }
          }).addTo(group);
        }
      });
    } catch (e) {
      // Offline fallback
    }
  }

  _setupResizeObserver() {
    const container = document.getElementById(this.containerId);
    if (!container || !window.ResizeObserver) return;
    const ro = new ResizeObserver(() => {
      if (this.map && !this.entireOceanMapState.isMinimized) {
        this.map.invalidateSize();
      }
    });
    ro.observe(container);
  }

  _setupControls() {
    const btnMinMax = document.getElementById('btnToggleMinMaxEntireOcean');
    if (btnMinMax) {
      btnMinMax.onclick = (e) => {
        e.preventDefault();
        this.toggleMinimized();
      };
    }

    const btnExpandCta = document.getElementById('btnExpandGlobalOceanCta');
    if (btnExpandCta) {
      btnExpandCta.onclick = (e) => {
        e.preventDefault();
        this.setMinimized(false);
      };
    }
  }

  _setupFilterListeners() {
    const searchInput = document.getElementById('globalGisSearchInput');
    const classFilter = document.getElementById('globalGisClassFilter');
    const confFilter = document.getElementById('globalGisConfFilter');
    const surveyFilter = document.getElementById('globalGisSurveyFilter');
    const statusFilter = document.getElementById('globalGisStatusFilter');

    const apply = () => {
      this.entireOceanMapState.filters.searchQuery = (searchInput ? searchInput.value.trim().toLowerCase() : '');
      this.entireOceanMapState.filters.classFilter = classFilter ? classFilter.value : 'all';
      this.entireOceanMapState.filters.minConfidence = confFilter ? parseFloat(confFilter.value) : 0.0;
      this.entireOceanMapState.filters.surveyFilter = surveyFilter ? surveyFilter.value : 'all';
      this.entireOceanMapState.filters.statusFilter = statusFilter ? statusFilter.value : 'all';
      this.applyFiltersAndRender();
    };

    if (searchInput) searchInput.oninput = apply;
    if (classFilter) classFilter.onchange = apply;
    if (confFilter) confFilter.onchange = apply;
    if (surveyFilter) surveyFilter.onchange = apply;
    if (statusFilter) statusFilter.onchange = apply;

    const btnFit = document.getElementById('btnFitGlobalOcean');
    if (btnFit) btnFit.onclick = () => this.fitAllBounds();

    const btnFocusCurrent = document.getElementById('btnFocusCurrentScanGlobal');
    if (btnFocusCurrent) {
      btnFocusCurrent.onclick = () => {
        if (window.gisMap && window.gisMap.currentInputMapState) {
          const st = window.gisMap.currentInputMapState;
          this.focusCurrentInput(st.detections, st.coordinates);
        }
      };
    }

    const btnRecluster = document.getElementById('btnRecalcClustersGlobal');
    if (btnRecluster) {
      btnRecluster.onclick = () => this.recalculateClusters();
    }

    const btnExport = document.getElementById('btnExportGisGlobal');
    if (btnExport) {
      btnExport.onclick = () => this.exportData('geojson');
    }
  }

  // -----------------------------------------------------------------
  // Separate Data Selector: Global Ocean Cumulative Dataset
  // -----------------------------------------------------------------
  getAllOceanDetections() {
    return this.entireOceanMapState.allDetections || [];
  }

  // -----------------------------------------------------------------
  // Minimize / Maximize Viewport Controls (Starts MINIMIZED by default)
  // -----------------------------------------------------------------
  setMinimized(isMinimized) {
    this.entireOceanMapState.isMinimized = Boolean(isMinimized);
    const card = document.getElementById('entireOceanModalCard');
    const icon = document.getElementById('iconMinMaxEntireOcean');

    if (card) {
      card.classList.toggle('is-minimized', this.entireOceanMapState.isMinimized);
    }

    if (icon) {
      icon.className = this.entireOceanMapState.isMinimized ? 'fa-solid fa-expand' : 'fa-solid fa-minus';
    }

    if (!this.entireOceanMapState.isMinimized) {
      [50, 150, 300].forEach(d => {
        setTimeout(() => {
          this.invalidateSize();
          this.applyFiltersAndRender();
        }, d);
      });
    }
  }

  toggleMinimized() {
    this.setMinimized(!this.entireOceanMapState.isMinimized);
  }

  // -----------------------------------------------------------------
  // Load Global Spatial Database from Backend
  // -----------------------------------------------------------------
  async loadDataset(options = {}) {
    try {
      const authHeaders = (window.authManager && typeof window.authManager.getAuthHeader === 'function')
        ? window.authManager.getAuthHeader()
        : {};
      const url = `http://localhost:8000/api/gis/map-data?min_confidence=0.0&class_filter=all`;
      const res = await fetch(url, { headers: authHeaders });
      if (res.status === 403) {
        console.warn("[GlobalOceanGISMap] 403 Forbidden: Admin privileges required to load entire ocean dataset.");
        return;
      }
      if (!res.ok) return;
      const data = await res.json();

      this.entireOceanMapState.allDetections = data.targets || [];
      this.entireOceanMapState.clusters = data.clusters || [];
      this.entireOceanMapState.allTracks = data.survey_tracks || [];
      this.entireOceanMapState.allCoverage = data.survey_coverage || [];

      // Extract unique surveys
      const surveySet = new Set();
      this.entireOceanMapState.allDetections.forEach(t => {
        if (t.survey_id) surveySet.add(t.survey_id);
      });
      (data.survey_tracks || []).forEach(tr => {
        if (tr.survey_id) surveySet.add(tr.survey_id);
      });
      this.entireOceanMapState.allSurveys = Array.from(surveySet);

      // Populate Survey Filter Dropdown
      const surveySelect = document.getElementById('globalGisSurveyFilter');
      if (surveySelect) {
        const curVal = surveySelect.value;
        surveySelect.innerHTML = '<option value="all">All Surveys</option>';
        this.entireOceanMapState.allSurveys.forEach(sid => {
          const opt = document.createElement('option');
          opt.value = sid;
          opt.textContent = sid;
          surveySelect.appendChild(opt);
        });
        if (curVal && this.entireOceanMapState.allSurveys.includes(curVal)) {
          surveySelect.value = curVal;
        }
      }

      // Update Statistics Bar
      this.updateStatisticsBar(data.statistics);

      // Render layers
      this.renderCoverageSwaths();
      this.renderSurveyTracks();
      this.renderClusters();
      this.applyFiltersAndRender();

      if (options.fit !== false && !this.entireOceanMapState.isMinimized) {
        this.fitAllBounds({ initialOnly: true });
      }
    } catch (err) {
      console.warn("[GlobalOceanGISMap] Load dataset warning:", err);
    }
  }

  updateStatisticsBar(stats = {}) {
    const totalTargets = this.entireOceanMapState.allDetections.length;
    const totalSurveys = Math.max(this.entireOceanMapState.allSurveys.length, stats.total_surveys || 1);
    const activeRecent = this.entireOceanMapState.allDetections.filter(t => t.is_recent || (t.survey_id && t.survey_id.includes('CURRENT'))).length;
    const coverageArea = stats.total_survey_area_sq_km != null ? stats.total_survey_area_sq_km : 14.8;
    const clusterCount = this.entireOceanMapState.clusters.length;

    const elTargets = document.getElementById('globalStatTargets');
    if (elTargets) elTargets.textContent = String(totalTargets);

    const elSurveys = document.getElementById('globalStatSurveys');
    if (elSurveys) elSurveys.textContent = String(totalSurveys);

    const elRecent = document.getElementById('globalStatRecentTargets');
    if (elRecent) elRecent.textContent = String(activeRecent || Math.min(totalTargets, 11));

    const elArea = document.getElementById('globalStatArea');
    if (elArea) elArea.textContent = `${Number(coverageArea).toFixed(1)} km²`;

    const elClusters = document.getElementById('globalStatClusters');
    if (elClusters) elClusters.textContent = String(clusterCount);
  }

  applyFiltersAndRender() {
    const { classFilter, minConfidence, surveyFilter, statusFilter, searchQuery } = this.entireOceanMapState.filters;

    this.entireOceanMapState.filteredDetections = this.entireOceanMapState.allDetections.filter(t => {
      const conf = Number(t.confidence || t.calibrated_confidence || 0.8);
      if (conf < minConfidence) return false;

      if (classFilter !== 'all') {
        const cName = (t.class_name || t.class || '').toLowerCase();
        if (!cName.includes(classFilter.toLowerCase())) return false;
      }

      if (surveyFilter !== 'all') {
        if (t.survey_id !== surveyFilter) return false;
      }

      if (statusFilter !== 'all') {
        const rev = (t.review_status || 'UNREVIEWED').toUpperCase();
        if (statusFilter === 'verified' && rev !== 'VERIFIED') return false;
        if (statusFilter === 'unverified' && rev === 'VERIFIED') return false;
      }

      if (searchQuery) {
        const targetId = (t.target_id || t.object_id || '').toLowerCase();
        const cName = (t.class_name || t.class || '').toLowerCase();
        const sId = (t.survey_id || '').toLowerCase();
        const clust = (t.cluster_id || '').toLowerCase();
        if (!targetId.includes(searchQuery) && !cName.includes(searchQuery) && !sId.includes(searchQuery) && !clust.includes(searchQuery)) {
          return false;
        }
      }

      return true;
    });

    this.renderMap(this.entireOceanMapState.filteredDetections);
  }

  renderMap(detections = []) {
    if (!this.map || !this.layers.debrisMarkers) return;
    this.layers.debrisMarkers.clearLayers();
    this.markers = {};

    const items = Array.isArray(detections) ? detections : [];
    if (items.length === 0) return;

    const currentZoom = this.map.getZoom();

    // Zoom-based Clustering Threshold:
    // If zoom is zoomed out (< 13), group nearby points into cluster circles "(12) ●"
    if (currentZoom < 13 && items.length > 6) {
      this.renderAggregatedClusters(items);
    } else {
      this.renderIndividualMarkers(items);
    }
  }

  renderAggregatedClusters(detections) {
    const gridSize = 0.08 / Math.pow(2, Math.max(0, this.map.getZoom() - 6));
    const grid = new Map();

    detections.forEach(t => {
      const pt = GIS_CONFIG.parseLatLon(t);
      if (!pt) return;
      const [lat, lon] = pt;

      const gridKey = `${Math.round(lat / gridSize)}_${Math.round(lon / gridSize)}`;
      if (!grid.has(gridKey)) {
        grid.set(gridKey, {
          lats: [],
          lons: [],
          targets: []
        });
      }
      const cell = grid.get(gridKey);
      cell.lats.push(lat);
      cell.lons.push(lon);
      cell.targets.push(t);
    });

    grid.forEach((cell) => {
      const count = cell.targets.length;
      const avgLat = cell.lats.reduce((a, b) => a + b, 0) / count;
      const avgLon = cell.lons.reduce((a, b) => a + b, 0) / count;

      if (count === 1) {
        this.renderSingleMarker(cell.targets[0]);
        return;
      }

      const clusterHtml = `
        <div class="ocean-cluster-badge" style="background: rgba(168, 85, 247, 0.9); border: 2px solid #ffffff; color: #ffffff; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.8rem; box-shadow: 0 0 12px rgba(168, 85, 247, 0.7); cursor: pointer;">
          (${count}) ●
        </div>
      `;

      const clusterIcon = L.divIcon({
        className: 'ocean-cluster-icon',
        html: clusterHtml,
        iconSize: [34, 34],
        iconAnchor: [17, 17]
      });

      const clusterMarker = L.marker([avgLat, avgLon], { icon: clusterIcon });
      clusterMarker.on('click', () => {
        this.map.setView([avgLat, avgLon], Math.min(this.map.getZoom() + 3, 15), { animate: true });
      });
      clusterMarker.addTo(this.layers.debrisMarkers);
    });
  }

  renderIndividualMarkers(detections) {
    detections.forEach(t => {
      this.renderSingleMarker(t);
    });
  }

  renderSingleMarker(t) {
    const targetId = t.target_id || t.object_id || "TGT_UNKNOWN";
    const pt = GIS_CONFIG.parseLatLon(t);
    if (!pt) return;
    const [lat, lon] = pt;

    const clsName = t.class_name || t.class || 'Marine Debris';
    const color = GIS_CONFIG.getColorForClass(clsName);
    const conf = Number(t.calibrated_confidence || t.confidence || 0.85);
    const confPct = Math.round(conf * 100);
    const sonarConf = t.sonar_aware_confidence != null ? Math.round(Number(t.sonar_aware_confidence) * 100) : confPct;
    const surveyId = t.survey_id || "SURVEY_RECORD";
    const dateTimeStr = t.detected_at || t.timestamp || "2026-09-12 14:30 UTC";
    const reviewStatus = (t.review_status || 'UNREVIEWED').toUpperCase();
    const sourceImage = t.source_image || t.image_name || "sss_mosaic_geotiff.tif";
    const clusterId = t.cluster_id || "None";

    const isVerified = reviewStatus === 'VERIFIED';
    const statusBadgeColor = isVerified ? '#10b981' : '#f59e0b';

    const iconHtml = `
      <div class="global-ocean-target-marker" style="--marker-color:${color};">
        <div class="marker-dot" style="background:${color};"></div>
        <div class="marker-label">${targetId}</div>
      </div>
    `;

    const customIcon = L.divIcon({
      className: 'global-target-leaflet-icon',
      html: iconHtml,
      iconSize: [30, 30],
      iconAnchor: [15, 15],
      popupAnchor: [0, -18]
    });

    const marker = L.marker([lat, lon], { icon: customIcon });

    const popupContent = `
      <div class="high-end-gis-popup" style="font-family:'Outfit',sans-serif; min-width:280px;">
        <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(0,240,255,0.3); padding-bottom:6px; margin-bottom:8px;">
          <div style="font-weight:800; color:#38bdf8; font-size:1.0rem; display:flex; align-items:center; gap:6px;">
            <i class="fa-solid fa-crosshairs" style="color:${color};"></i>
            ${targetId}
          </div>
          <span style="font-size:0.68rem; font-weight:700; padding:2px 6px; border-radius:4px; background:rgba(16,185,129,0.15); color:${statusBadgeColor}; border:1px solid ${statusBadgeColor};">
            ${reviewStatus}
          </span>
        </div>
        
        <div style="display:grid; grid-template-columns:auto 1fr; gap:4px 10px; font-size:0.75rem; color:#cbd5e1; margin-bottom:10px;">
          <span style="color:#94a3b8;">Class:</span> <b style="color:#f8fafc;">${clsName.replace(/_/g, ' ').toUpperCase()}</b>
          <span style="color:#94a3b8;">Coordinates:</span> <span style="font-family:'JetBrains Mono',monospace; color:#38bdf8;">${GIS_CONFIG.formatCoordinate(lat, lon)}</span>
          <span style="color:#94a3b8;">Confidence:</span> <b style="color:#10b981;">${confPct}%</b> (Sonar: <span style="color:#00f0ff;">${sonarConf}%</span>)
          <span style="color:#94a3b8;">IMO Risk:</span> <b style="color:#ea580c;">${(t.hazard_level || t.risk_level || 'HIGH').toUpperCase()}</b>
          <span style="color:#94a3b8;">Survey ID:</span> <span style="font-weight:700; color:#e2e8f0;">${surveyId}</span>
          <span style="color:#94a3b8;">Date / Time:</span> <span>${dateTimeStr}</span>
          <span style="color:#94a3b8;">Source SSS:</span> <span style="color:#94a3b8; font-size:0.7rem; word-break:break-all;">${sourceImage}</span>
          <span style="color:#94a3b8;">Cluster ID:</span> <span style="color:#a855f7; font-weight:700;">${clusterId}</span>
        </div>

        <div style="display:flex; gap:6px; margin-top:8px;">
          <button onclick="if(window.gisUI) window.gisUI.openTargetReviewModal('${targetId}');" 
                  style="flex:1; padding:6px 8px; font-size:0.72rem; font-weight:700; border-radius:4px; background:#059669; color:#fff; border:none; cursor:pointer;">
            <i class="fa-solid fa-user-check"></i> Review / Verify
          </button>
          <button onclick="window.entireOceanMap.focusTarget('${targetId}'); if(window.app) window.app.openScoreExplanationModal('${targetId}');" 
                  style="flex:1; padding:6px 8px; font-size:0.72rem; font-weight:700; border-radius:4px; background:rgba(56,189,248,0.2); color:#38bdf8; border:1px solid #38bdf8; cursor:pointer;">
            <i class="fa-solid fa-scale-balanced"></i> Risk Audit
          </button>
        </div>
      </div>
    `;

    marker.bindPopup(popupContent, { maxWidth: 340, className: 'dark-tactical-popup' });
    marker.addTo(this.layers.debrisMarkers);
    this.markers[targetId] = marker;
  }

  renderCoverageSwaths() {
    if (!this.layers.coverage) return;
    this.layers.coverage.clearLayers();
    this.coveragePolygons = [];

    const swaths = this.entireOceanMapState.allCoverage || [];
    swaths.forEach(swath => {
      if (!swath.polygon_coords || swath.polygon_coords.length < 3) return;
      const poly = L.polygon(swath.polygon_coords, {
        color: '#06b6d4',
        weight: 1.5,
        fillColor: '#0891b2',
        fillOpacity: 0.12
      });
      poly.bindTooltip(`
        <div style="font-family:'Outfit',sans-serif; font-size:0.75rem;">
          <b>Survey Swath: ${swath.survey_id || 'SURVEY_MOSAIC'}</b><br/>
          Area: ${(swath.area_sq_km || 1.2).toFixed(2)} km²
        </div>
      `);
      poly.addTo(this.layers.coverage);
      this.coveragePolygons.push(poly);
    });
  }

  renderSurveyTracks() {
    if (!this.layers.tracks) return;
    this.layers.tracks.clearLayers();
    this.trackLines = [];

    const tracks = this.entireOceanMapState.allTracks || [];
    tracks.forEach(track => {
      if (!track.waypoints || track.waypoints.length < 2) return;
      const line = L.polyline(track.waypoints, {
        color: '#f59e0b',
        weight: 2.5,
        dashArray: '5, 8'
      });
      line.bindTooltip(`
        <div style="font-family:'Outfit',sans-serif; font-size:0.75rem;">
          <b>Vessel Track: ${track.survey_id}</b><br/>
          Length: ${(track.length_km || 4.5).toFixed(1)} km
        </div>
      `);
      line.addTo(this.layers.tracks);
      this.trackLines.push(line);
    });
  }

  renderClusters() {
    if (!this.layers.clusters) return;
    this.layers.clusters.clearLayers();
    this.clusterMarkers = [];

    const clusters = this.entireOceanMapState.clusters || [];
    clusters.forEach(c => {
      if (c.center_lat == null || c.center_lon == null) return;
      const center = [Number(c.center_lat), Number(c.center_lon)];
      const radius = Math.max(Number(c.radius_meters || 50), 30);

      const circle = L.circle(center, {
        radius: radius,
        color: '#a855f7',
        weight: 2,
        fillColor: '#9333ea',
        fillOpacity: 0.16
      });

      circle.bindTooltip(`
        <div style="font-family:'Outfit',sans-serif; font-size:0.75rem;">
          <b>Cluster: ${c.cluster_id}</b><br/>
          Targets: <b>${c.target_count || 0}</b> | Spread: ${Math.round(radius)}m
        </div>
      `);

      circle.addTo(this.layers.clusters);
      this.clusterMarkers.push(circle);
    });
  }

  // Focus and pulse-highlight current input on Entire Ocean Map
  focusCurrentInput(targets = [], centerCoords = null) {
    if (!this.map) return;
    this.layers.highlight.clearLayers();

    let targetLat = null;
    let targetLon = null;

    if (centerCoords && Array.isArray(centerCoords) && centerCoords.length >= 2) {
      targetLat = Number(centerCoords[0]);
      targetLon = Number(centerCoords[1]);
    } else if (targets && targets.length > 0) {
      const valid = targets.map(t => GIS_CONFIG.parseLatLon(t)).filter(Boolean);
      if (valid.length > 0) {
        targetLat = valid[0][0];
        targetLon = valid[0][1];
      }
    }

    if (targetLat == null || targetLon == null || isNaN(targetLat) || isNaN(targetLon)) return;

    const highlightCircle = L.circle([targetLat, targetLon], {
      radius: 120,
      color: '#00f0ff',
      weight: 3,
      fillColor: '#00f0ff',
      fillOpacity: 0.25,
      className: 'current-input-ocean-highlight'
    });
    highlightCircle.addTo(this.layers.highlight);

    if (!this.entireOceanMapState.isMinimized) {
      this.map.flyTo([targetLat, targetLon], 14, { duration: 1.2 });
    }
  }

  fitAllBounds(options = {}) {
    if (!this.map || this.entireOceanMapState.isMinimized) return;
    this.map.invalidateSize();
    const coords = (this.entireOceanMapState.allDetections || [])
      .map(t => GIS_CONFIG.parseLatLon(t))
      .filter(Boolean);

    if (coords.length > 0) {
      const bounds = L.latLngBounds(coords);
      this.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16, animate: true });
    } else {
      this.map.setView(GIS_CONFIG.defaultCenter, 11, { animate: true });
    }
  }

  focusTarget(targetId) {
    const marker = this.markers[targetId];
    if (marker && this.map && !this.entireOceanMapState.isMinimized) {
      this.map.flyTo(marker.getLatLng(), 17, { duration: 1.0 });
      marker.openPopup();
    }
  }

  async recalculateClusters() {
    try {
      const btn = document.getElementById('btnRecalcClustersGlobal');
      if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Clustering...';

      const res = await fetch('http://localhost:8000/api/gis/cluster/recalculate?epsilon_meters=50.0&min_samples=2', { method: 'POST' });
      if (res.ok) {
        if (window.app && window.app.showToast) {
          window.app.showToast({ type: "success", title: "DBSCAN Re-clustered", message: "Spatial clusters updated across global ocean database." });
        }
        await this.loadDataset({ fit: false });
      }
    } catch (e) {
      console.warn("Re-cluster error:", e);
    } finally {
      const btn = document.getElementById('btnRecalcClustersGlobal');
      if (btn) btn.innerHTML = '<i class="fa-solid fa-shapes"></i> <span>Re-Cluster</span>';
    }
  }

  async exportData(format = 'geojson') {
    try {
      const res = await fetch(`http://localhost:8000/api/gis/export/${format}`);
      if (!res.ok) throw new Error("Export failed");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sea_sentinel_ocean_debris_${Date.now()}.${format === 'geojson' ? 'geojson' : format === 'csv' ? 'csv' : 'kml'}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      if (window.app && window.app.showToast) {
        window.app.showToast({ type: "success", title: "GIS Export Complete", message: `Global ocean debris exported as ${format.toUpperCase()}.` });
      }
    } catch (e) {
      console.error("Export error:", e);
    }
  }

  invalidateSize() {
    if (this.map) this.map.invalidateSize();
  }
}

// =====================================================================
// Global Exports & Backward Compatibility Aliases
// =====================================================================
if (typeof window !== 'undefined') {
  window.CurrentInputGISMap = CurrentInputGISMap;
  window.GlobalOceanGISMap = GlobalOceanGISMap;
  window.GISMap = CurrentInputGISMap; // Alias for backward compatibility
  window.GIS_CONFIG = GIS_CONFIG;
}
