/**
 * Sea Sentinel: High-End Offline Marine Debris GIS & Geospatial Intelligence System
 * Unified Multi-Image Ocean Map Engine with Physics-Grounded WGS84 Georeferencing,
 * Offline Tactical Nautical Grid, Cumulative Ocean Debris Dataset & Current Image Focus.
 */

class GISMap {
  constructor(containerId) {
    this.containerId = containerId;
    this.map = null;
    this.markers = {};
    this.currentScanMarkers = {};
    this.clusterMarkers = [];
    this.trackLines = [];
    this.coveragePolygons = [];
    this.uncertaintyCircles = [];

    // Current active input image state
    this.currentScanTargets = [];
    this.currentScanMeta = {};
    this.activeViewMode = 'current'; // 'current' | 'ocean'

    // Cumulative Ocean Dataset from persistent spatial DB
    this.allTargets = [];
    this.filteredTargets = [];
    this.allClusters = [];
    this.allTracks = [];
    this.allCoverage = [];

    // Separate toggleable Leaflet Layer Groups
    this.layers = {
      offlineGrid: null,
      coverage: null,
      tracks: null,
      clusters: null,
      cumulativeDebris: null,
      currentScanDebris: null,
      uncertainty: null,
      coral_reefs: null,
      marine_protected_areas: null,
      seagrass_meadows: null,
      underwater_infrastructure: null
    };

    this.lastCenter = [30.175, -87.825]; // Gulf of Mexico / Breton Sound default
    this.lastZoom = 13;
    this.filters = {
      minConfidence: 0.0,
      classFilter: "all",
      timeFilter: "all",
      searchQuery: "",
      verificationFilter: "all"
    };

    this._init();
  }

  _init() {
    if (typeof L === 'undefined') {
      console.warn("[GISMap] Leaflet library not loaded yet. Retrying in 250ms...");
      setTimeout(() => this._init(), 250);
      return;
    }

    const container = document.getElementById(this.containerId);
    if (!container) {
      console.warn("[GISMap] Container element not found:", this.containerId);
      return;
    }

    try {
      // Initialize Layer Groups
      this.layers.offlineGrid = L.layerGroup();
      this.layers.coverage = L.layerGroup();
      this.layers.tracks = L.layerGroup();
      this.layers.clusters = L.layerGroup();
      this.layers.cumulativeDebris = L.layerGroup();
      this.layers.currentScanDebris = L.layerGroup();
      this.layers.uncertainty = L.layerGroup();
      this.layers.coral_reefs = L.layerGroup();
      this.layers.marine_protected_areas = L.layerGroup();
      this.layers.seagrass_meadows = L.layerGroup();
      this.layers.underwater_infrastructure = L.layerGroup();

      this._initMap();
      this._buildOfflineTacticalGrid();
      this.loadLocalGISLayers();
      this.loadGISDataset();
      this._setupResizeObserver();
    } catch (err) {
      console.error("[GISMap] Initialization error:", err);
    }
  }

  _initMap() {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }

    this.map = L.map(this.containerId, {
      center: this.lastCenter,
      zoom: 12,
      zoomControl: false,
      attributionControl: true,
      minZoom: 2,
      maxZoom: 18
    });

    L.control.zoom({ position: 'bottomright' }).addTo(this.map);

    // 1. Primary Default Basemap: OpenStreetMap Standard (100% Free, Truly Open-Source, Zero API Keys, No Watermarks)
    const osmBase = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
      maxNativeZoom: 19
    });

    // 2. High-Resolution Satellite Imagery & Coastal Reefs (ESRI World Imagery - Free, No API Key)
    const satBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, Maxar, Earthstar Geographics',
      maxZoom: 19,
      maxNativeZoom: 18
    });

    // 3. Real Ocean Bathymetry & Seabed Contours (GEBCO / NOAA / ESRI - maxNativeZoom: 10 prevents missing tile placeholders)
    const oceanBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, GEBCO, NOAA, National Geographic',
      maxZoom: 19,
      maxNativeZoom: 10
    });

    // 4. ESRI World Topographic & Physical Ocean Map (Free, No API Key)
    const topoBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, HERE, Garmin, USGS, Intermap, INCREMENT P',
      maxZoom: 19,
      maxNativeZoom: 18
    });

    // 5. Dark Tactical Marine Canvas (ESRI World Dark Canvas - Free, No API Key)
    const darkBase = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
      maxZoom: 19,
      maxNativeZoom: 16
    });

    // 6. OpenStreetMap Humanitarian / Coastal Relief (100% Free, No API Key)
    const hotBase = L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team',
      maxZoom: 19,
      maxNativeZoom: 19
    });

    // Default: Add OpenStreetMap Standard as primary base (Guaranteed zero API keys, no watermarks)
    osmBase.addTo(this.map);
    this.layers.offlineGrid.addTo(this.map);

    // Basemaps selector (All 100% Free, Zero API Keys Required)
    const baseMaps = {
      "<span style='color:#059669; font-weight:700;'>🗺️ OpenStreetMap Marine</span>": osmBase,
      "<span style='color:#0284c7; font-weight:700;'>🛰️ Satellite &amp; Coastal Reefs</span>": satBase,
      "<span style='color:#00f0ff; font-weight:700;'>🌊 Ocean Bathymetry (GEBCO)</span>": oceanBase,
      "<span style='color:#0d9488; font-weight:700;'>🧭 Topographic &amp; Seabed</span>": topoBase,
      "<span style='color:#6366f1; font-weight:700;'>◈ Dark Tactical Marine</span>": darkBase,
      "<span style='color:#f59e0b; font-weight:700;'>🏖️ Humanitarian Coastal</span>": hotBase
    };

    // Overlay layers selector
    const overlayMaps = {
      "<span style='color:#00f0ff; font-weight:700;'>🎯 Current Input Debris</span>": this.layers.currentScanDebris,
      "<span style='color:#38bdf8; font-weight:600;'>🌐 Cumulative Ocean Debris</span>": this.layers.cumulativeDebris,
      "<span style='color:#a855f7; font-weight:600;'>◈ DBSCAN Clusters</span>": this.layers.clusters,
      "<span style='color:#f59e0b; font-weight:600;'>🚢 Survey Vessel Tracks</span>": this.layers.tracks,
      "<span style='color:#06b6d4; font-weight:600;'>📐 Scanned Sonar Swaths</span>": this.layers.coverage,
      "<span style='color:#ef4444; font-weight:600;'>⭕ Uncertainty Radii</span>": this.layers.uncertainty,
      "<span style='color:#64748b; font-weight:600;'>🧭 Nautical Graticule</span>": this.layers.offlineGrid,
      "<span style='color:#ec4899; font-weight:600;'>🪸 Coral Reefs</span>": this.layers.coral_reefs,
      "<span style='color:#10b981; font-weight:600;'>🛡️ MPAs</span>": this.layers.marine_protected_areas,
      "<span style='color:#84cc16; font-weight:600;'>🌿 Seagrass Beds</span>": this.layers.seagrass_meadows,
      "<span style='color:#f97316; font-weight:600;'>⚡ Subsea Cables/Pipes</span>": this.layers.underwater_infrastructure
    };

    // Layer switcher: sleek collapsible floating control
    this.layersControl = L.control.layers(baseMaps, overlayMaps, { position: 'topright', collapsed: true }).addTo(this.map);

    // Add active overlay layers
    this.layers.coverage.addTo(this.map);
    this.layers.tracks.addTo(this.map);
    this.layers.clusters.addTo(this.map);
    this.layers.cumulativeDebris.addTo(this.map);
    this.layers.currentScanDebris.addTo(this.map);
    this.layers.uncertainty.addTo(this.map);

    // Live cursor telemetry HUD
    this.map.on('mousemove', (e) => {
      const hud = document.getElementById('mapCoordsHud');
      if (hud && e.latlng) {
        const latStr = this.formatCoordDeg(e.latlng.lat, 'lat');
        const lonStr = this.formatCoordDeg(e.latlng.lng, 'lon');
        hud.innerHTML = `<i class="fa-solid fa-crosshairs"></i> Cursor: <b>${latStr}, ${lonStr}</b> &nbsp;|&nbsp; Datum: <span style="color:#00f0ff;">WGS84 (EPSG:4326)</span> &nbsp;|&nbsp; Status: <span style="color:#10b981;">● OFFLINE READY</span>`;
      }
    });
  }

  // -----------------------------------------------------------------
  // Offline-Native Bathymetric Nautical Graticule Generator
  // Transparent oceanic graticule lines that enhance real ocean maps
  // -----------------------------------------------------------------
  _buildOfflineTacticalGrid() {
    if (!this.layers.offlineGrid) return;
    this.layers.offlineGrid.clearLayers();

    // Graticule Lat/Lon Grid Lines across global oceans (subtle cyan/blue lines)
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

  _setupResizeObserver() {
    const container = document.getElementById(this.containerId);
    if (!container) return;

    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => {
        if (this.map) {
          this.map.invalidateSize();
        }
      });
      ro.observe(container);
    }
  }

  // -----------------------------------------------------------------
  // Load Unified GIS Dataset from Backend Spatial DB
  // -----------------------------------------------------------------
  async loadGISDataset(options = {}) {
    try {
      const url = `http://localhost:8000/api/gis/map-data?min_confidence=${this.filters.minConfidence}&class_filter=${this.filters.classFilter}`;
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();

      this.allTargets = data.targets || [];
      this.allClusters = data.clusters || [];
      this.allTracks = data.survey_tracks || [];
      this.allCoverage = data.survey_coverage || [];

      // Update Dashboard Statistics HUD
      if (data.statistics) {
        this.updateStatsBar(data.statistics);
      }

      // Render GIS Layers
      this.renderCoverageSwaths();
      this.renderSurveyTracks();
      this.renderClusters();
      this.renderTargets();

      if (options.fit !== false) {
        if (this.currentScanTargets && this.currentScanTargets.length > 0) {
          this.fitCurrentScan();
        } else {
          this.fitEntireOcean({ initialOnly: true });
        }
      }
    } catch (err) {
      console.warn("[GISMap] GIS dataset load warning (offline mode active):", err);
    }
  }

  // -----------------------------------------------------------------
  // Integration Hook: Called whenever an SSS image is analyzed
  // -----------------------------------------------------------------
  setTargets(targets = [], surveyMeta = {}) {
    this.currentScanTargets = targets || [];
    this.currentScanMeta = surveyMeta || {};

    // Reload the cumulative database dataset so that the new survey and all previous surveys appear together
    this.loadGISDataset({ fit: true });
  }

  // -----------------------------------------------------------------
  // Render Scanned Swath Coverage Polygons
  // -----------------------------------------------------------------
  renderCoverageSwaths() {
    if (!this.layers.coverage) return;
    this.layers.coverage.clearLayers();

    this.allCoverage.forEach(cov => {
      const geom = cov.geometry || {};
      if (geom.type === "Polygon" && geom.coordinates) {
        const latLngs = geom.coordinates[0].map(c => [c[1], c[0]]); // [lon, lat] -> [lat, lon]
        const poly = L.polygon(latLngs, {
          color: '#06b6d4',
          weight: 1.5,
          opacity: 0.7,
          fillColor: '#0891b2',
          fillOpacity: 0.20,
          dashArray: '4, 4'
        });

        poly.bindTooltip(`
          <div style="font-family:'Outfit',sans-serif; font-size:0.75rem;">
            <b><i class="fa-solid fa-vector-square"></i> Scanned Sonar Swath</b><br/>
            Survey ID: ${cov.survey_id || '--'}<br/>
            Width: ${cov.coverage_width_m || 100}m | Range: ${cov.range_m || 50}m<br/>
            Area: ${Math.round(cov.area_sq_m || 5000)} m²
          </div>
        `);
        this.layers.coverage.addLayer(poly);
      }
    });
  }

  // -----------------------------------------------------------------
  // Render Survey Vehicle Tracklines (LineStrings)
  // -----------------------------------------------------------------
  renderSurveyTracks() {
    if (!this.layers.tracks) return;
    this.layers.tracks.clearLayers();

    this.allTracks.forEach(trk => {
      const geom = trk.geometry || {};
      if (geom.type === "LineString" && geom.coordinates) {
        const latLngs = geom.coordinates.map(c => [c[1], c[0]]);
        const polyline = L.polyline(latLngs, {
          color: '#f59e0b',
          weight: 3,
          opacity: 0.85,
          dashArray: '6, 6'
        });

        polyline.bindTooltip(`
          <div style="font-family:'Outfit',sans-serif; font-size:0.75rem;">
            <b><i class="fa-solid fa-ship"></i> Survey Track: ${trk.track_id}</b><br/>
            Heading: ${trk.heading || 0}° | Speed: ${trk.speed_knots || 3.0} kts<br/>
            Time: ${trk.timestamp || '--'}
          </div>
        `);
        this.layers.tracks.addLayer(polyline);
      }
    });
  }

  // -----------------------------------------------------------------
  // Render DBSCAN Spatial Clusters
  // -----------------------------------------------------------------
  renderClusters() {
    if (!this.layers.clusters) return;
    this.layers.clusters.clearLayers();

    this.allClusters.forEach(c => {
      const lat = Number(c.center_latitude);
      const lon = Number(c.center_longitude);
      if (isNaN(lat) || isNaN(lon)) return;

      const radiusMeters = Math.max(10, Number(c.radius_m || 20));

      const circle = L.circle([lat, lon], {
        radius: radiusMeters,
        color: '#a855f7',
        weight: 2,
        opacity: 0.75,
        fillColor: '#9333ea',
        fillOpacity: 0.18,
        dashArray: '5, 5'
      });

      const icon = L.divIcon({
        className: 'custom-cluster-badge',
        html: `
          <div style="
            background: linear-gradient(135deg, #7e22ce, #a855f7);
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 12px;
            border: 2px solid #ffffff;
            box-shadow: 0 0 12px rgba(168, 85, 247, 0.8);
            white-space: nowrap;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
          ">
            <i class="fa-solid fa-shapes"></i> ${c.cluster_id} (${c.target_count})
          </div>
        `,
        iconSize: [60, 24],
        iconAnchor: [30, 12]
      });

      const badge = L.marker([lat, lon], { icon });

      badge.bindPopup(`
        <div style="font-family:'Outfit',sans-serif; color:#0f172a; min-width:200px; padding:4px;">
          <div style="font-weight:800; font-size:0.95rem; color:#7e22ce; font-family:'JetBrains Mono',monospace;">
            ${c.cluster_id} &mdash; Debris Cluster
          </div>
          <div style="font-size:0.80rem; margin:4px 0;"><b>Targets:</b> ${c.target_count} Objects</div>
          <div style="font-size:0.80rem; margin:4px 0;"><b>Dominant Class:</b> <span style="text-transform:capitalize; font-weight:700;">${(c.dominant_class||'debris').replace(/_/g,' ')}</span></div>
          <div style="font-size:0.75rem; color:#64748b;"><b>Radius:</b> ±${radiusMeters}m | <b>Density:</b> ${c.density} targets/km²</div>
          <button onclick="if(window.gisMap) window.gisMap.map.flyTo([${lat}, ${lon}], 17);" style="margin-top:6px; width:100%; background:#7e22ce; color:#fff; border:none; border-radius:4px; padding:4px 8px; font-weight:700; font-size:0.72rem; cursor:pointer;">
            Zoom into Cluster
          </button>
        </div>
      `);

      this.layers.clusters.addLayer(circle);
      this.layers.clusters.addLayer(badge);
    });
  }

  // -----------------------------------------------------------------
  // Render Debris Targets (Current Scan Active + Cumulative Ocean)
  // -----------------------------------------------------------------
  renderTargets() {
    if (!this.layers.cumulativeDebris || !this.layers.currentScanDebris || !this.layers.uncertainty) return;
    this.layers.cumulativeDebris.clearLayers();
    this.layers.currentScanDebris.clearLayers();
    this.layers.uncertainty.clearLayers();
    this.markers = {};
    this.currentScanMarkers = {};

    // Determine current scan target IDs
    const currentTargetIds = new Set();
    this.currentScanTargets.forEach(ct => {
      const id = ct.target_id || ct.object_id || ct.id;
      if (id) currentTargetIds.add(String(id));
    });

    // Merge all historical targets from persistent database with latest current scan targets
    const targetsMap = new Map();
    this.allTargets.forEach(t => {
      const id = t.target_id || t.object_id || t.id;
      if (id) targetsMap.set(String(id), t);
    });
    this.currentScanTargets.forEach((ct, idx) => {
      const id = ct.target_id || ct.object_id || ct.id || `TGT_SCAN_${idx}`;
      if (targetsMap.has(String(id))) {
        targetsMap.set(String(id), { ...targetsMap.get(String(id)), ...ct });
      } else {
        targetsMap.set(String(id), ct);
      }
    });

    const targetsToRender = Array.from(targetsMap.values());

    targetsToRender.forEach(t => {
      const lat = Number(t.latitude);
      const lon = Number(t.longitude);
      if (isNaN(lat) || isNaN(lon) || t.latitude === null || t.longitude === null) {
        return; // Strictly omit unreferenced targets to avoid false projection
      }

      const isCurrentScan = currentTargetIds.has(t.target_id) || (this.currentScanTargets.length > 0 && targetsToRender === this.currentScanTargets);
      const className = (t.class_name || t.class || "debris").toLowerCase();
      const conf = Math.round(Number(t.confidence || t.calibrated_confidence || 0.85) * 100);
      const uncert = Number(t.uncertainty_radius || t.uncertainty_radius_m || 4.5);
      const obsCount = Number(t.observation_count || 1);
      const isVerified = t.verification_status === "VERIFIED";

      // Color coding by debris class
      let color = "#00f0ff";
      if (className.includes("ghost_net") || className.includes("net")) color = "#ff0055";
      else if (className.includes("fishing") || className.includes("gear")) color = "#ffaa00";
      else if (className.includes("metal") || className.includes("container") || className.includes("engine")) color = "#00e676";
      else if (className.includes("plastic")) color = "#38bdf8";
      else if (className.includes("shipwreck") || className.includes("wreck")) color = "#f43f5e";
      else if (className.includes("pipe") || className.includes("cable")) color = "#eab308";

      // 1. Uncertainty Area Buffer
      const uncertCircle = L.circle([lat, lon], {
        radius: uncert,
        color: isCurrentScan ? '#00f0ff' : color,
        weight: isCurrentScan ? 2 : 1,
        opacity: isCurrentScan ? 0.7 : 0.45,
        fillColor: color,
        fillOpacity: isCurrentScan ? 0.18 : 0.08,
        dashArray: isCurrentScan ? '2, 4' : '3, 3'
      });
      this.layers.uncertainty.addLayer(uncertCircle);

      // 2. Tactical Marker Icon
      const markerSize = isCurrentScan ? 34 : 26;
      const pulseHtml = isCurrentScan
        ? `<div style="
            position:absolute; width:${markerSize}px; height:${markerSize}px; border-radius:50%;
            background:${color}; opacity:0.6;
            animation:sonar-ping 1.5s cubic-bezier(0,0,0.2,1) infinite;
            border: 2px solid #00f0ff;
          "></div>`
        : `<div style="
            position:absolute; width:${markerSize}px; height:${markerSize}px; border-radius:50%;
            background:${color}; opacity:0.35;
            animation:sonar-ping 2.5s cubic-bezier(0,0,0.2,1) infinite;
          "></div>`;

      const icon = L.divIcon({
        className: `custom-debris-marker ${isCurrentScan ? 'active-scan-marker' : ''}`,
        html: `
          <div style="position:relative; width:${markerSize}px; height:${markerSize}px; display:flex; align-items:center; justify-content:center;">
            ${pulseHtml}
            <div style="
              width:${isCurrentScan ? 18 : 14}px; height:${isCurrentScan ? 18 : 14}px; border-radius:50%;
              background:${color};
              box-shadow:0 0 12px ${color}, 0 0 20px ${isCurrentScan ? '#00f0ff' : color};
              border:2px solid ${isCurrentScan ? '#ffd700' : (isVerified ? '#10b981' : '#ffffff')};
              cursor:pointer; position:relative; z-index:2;
              display:flex; align-items:center; justify-content:center;
            ">
              ${isCurrentScan ? '<span style="font-size:0.55rem; color:#000; font-weight:900;">★</span>' : (obsCount > 1 ? `<span style="font-size:0.52rem; font-weight:900; color:#000;">${obsCount}</span>` : '')}
            </div>
          </div>
        `,
        iconSize: [markerSize, markerSize],
        iconAnchor: [markerSize / 2, markerSize / 2]
      });

      const marker = L.marker([lat, lon], { icon });
      marker._targetId = t.target_id;
      this.markers[t.target_id] = marker;

      if (isCurrentScan) {
        this.currentScanMarkers[t.target_id] = marker;
      }

      // Target Detail Popup
      const statusBadge = isVerified
        ? `<span style="background:rgba(16,185,129,0.2); color:#10b981; border:1px solid #10b981; padding:2px 6px; border-radius:4px; font-size:0.65rem; font-weight:800;">✓ HUMAN VERIFIED</span>`
        : `<span style="background:rgba(245,158,11,0.2); color:#f59e0b; border:1px solid #f59e0b; padding:2px 6px; border-radius:4px; font-size:0.65rem; font-weight:800;">● UNVERIFIED</span>`;

      const activeBadge = isCurrentScan
        ? `<span style="background:rgba(0,240,255,0.2); color:#00f0ff; border:1px solid #00f0ff; padding:2px 6px; border-radius:4px; font-size:0.65rem; font-weight:800;">🎯 CURRENT SCAN</span>`
        : `<span style="background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid #38bdf8; padding:2px 6px; border-radius:4px; font-size:0.65rem; font-weight:700;">🌐 OCEAN DB</span>`;

      const waterBodyName = this._inferWaterBody(lat, lon);

      const popupContent = `
        <div style="font-family:'Outfit',sans-serif; color:#0f172a; min-width:250px; padding:6px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-weight:900; font-size:1rem; color:${color}; font-family:'JetBrains Mono',monospace;">
              ${t.target_id}
            </span>
            <div style="display:flex; gap:4px;">
              ${activeBadge}
              ${statusBadge}
            </div>
          </div>
          
          <div style="font-size:0.86rem; margin-bottom:4px;">
            <b>Classification:</b> <span style="font-weight:700; text-transform:capitalize;">${className.replace(/_/g, ' ')}</span>
          </div>

          <div style="background:#f1f5f9; border-radius:4px; padding:6px; margin-bottom:6px; display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:0.75rem;">
            <div>🎯 <b>AI Conf:</b> <span style="font-weight:700; color:#0284c7;">${conf}%</span></div>
            <div>🔄 <b>Merged Obs:</b> <span style="font-weight:700; color:#7e22ce;">${obsCount}</span></div>
            <div>🌊 <b>Depth:</b> <span style="font-weight:700;">${(t.depth||0).toFixed(1)} m</span></div>
            <div>📏 <b>Accuracy:</b> <span style="font-weight:700; color:#059669;">&plusmn;${uncert.toFixed(1)} m</span></div>
          </div>

          <div style="font-size:0.74rem; color:#0369a1; background:rgba(2,132,199,0.08); border:1px solid rgba(2,132,199,0.2); border-radius:4px; padding:4px 6px; margin-bottom:6px;">
            <i class="fa-solid fa-water"></i> <b>Water Body:</b> ${waterBodyName}
          </div>

          ${t.cluster_id ? `<div style="font-size:0.75rem; color:#7e22ce; margin-bottom:4px;"><b>Cluster:</b> ${t.cluster_id}</div>` : ''}

          <div style="font-size:0.70rem; color:#0284c7; font-family:'JetBrains Mono',monospace; margin-bottom:6px;">
            <i class="fa-solid fa-crosshairs"></i> ${this.formatCoordinate(lat, lon)}
          </div>

          <div style="display:flex; gap:6px; border-top:1px solid #e2e8f0; padding-top:6px;">
            <button onclick="if(window.app) window.app.openTargetReviewModal('${t.target_id}');" style="flex:1; background:#0284c7; color:#fff; border:none; border-radius:4px; padding:5px 6px; font-weight:700; font-size:0.72rem; cursor:pointer;">
              <i class="fa-solid fa-clipboard-check"></i> Review
            </button>
            <button onclick="if(window.app) window.app.inspectTargetDetails('${t.target_id}');" style="flex:1; background:#334155; color:#fff; border:none; border-radius:4px; padding:5px 6px; font-weight:700; font-size:0.72rem; cursor:pointer;">
              <i class="fa-solid fa-magnifying-glass"></i> Details
            </button>
          </div>
        </div>
      `;

      marker.bindPopup(popupContent);

      if (isCurrentScan) {
        this.layers.currentScanDebris.addLayer(marker);
      } else {
        this.layers.cumulativeDebris.addLayer(marker);
      }
    });
  }

  // -----------------------------------------------------------------
  // Statistics Bar Updating
  // -----------------------------------------------------------------
  updateStatsBar(stats) {
    const elTargets = document.getElementById('gisStatTargets') || document.getElementById('kpiTotal');
    if (elTargets) elTargets.innerText = stats.total_targets || 0;

    const elVerified = document.getElementById('gisStatVerified') || document.getElementById('kpiConfirmed');
    if (elVerified) elVerified.innerText = stats.verified_targets || 0;

    const elClusters = document.getElementById('gisStatClusters');
    if (elClusters) elClusters.innerText = stats.total_clusters || 0;

    const elArea = document.getElementById('gisStatArea');
    if (elArea) elArea.innerText = `${stats.surveyed_area_km2 || 0} km²`;

    const elImages = document.getElementById('gisStatImages');
    if (elImages) elImages.innerText = stats.total_images || 0;
  }

  // -----------------------------------------------------------------
  // Map Fit Bounds Utilities (Current Scan vs Entire Ocean)
  // -----------------------------------------------------------------
  fitCurrentScan() {
    if (!this.map) return;
    this.map.invalidateSize();
    const coords = [];

    // Check current scan targets
    const targetsToCheck = (this.currentScanTargets && this.currentScanTargets.length > 0)
      ? this.currentScanTargets
      : this.allTargets;

    targetsToCheck.forEach(t => {
      const lat = Number(t.latitude != null ? t.latitude : t.lat);
      const lon = Number(t.longitude != null ? t.longitude : t.lon);
      if (!isNaN(lat) && !isNaN(lon) && lat !== 0 && lon !== 0 && t.latitude !== null && t.longitude !== null) {
        coords.push([lat, lon]);
      }
    });

    if (coords.length > 0) {
      if (coords.length === 1) {
        this.map.flyTo(coords[0], 16, { duration: 0.8 });
      } else {
        const bounds = L.latLngBounds(coords);
        this.map.fitBounds(bounds, { padding: [60, 60], maxZoom: 16 });
      }
      // Open first current scan popup if available
      const firstTarget = targetsToCheck[0];
      if (firstTarget && firstTarget.target_id && this.markers[firstTarget.target_id]) {
        setTimeout(() => this.markers[firstTarget.target_id].openPopup(), 400);
      }
    } else {
      this.fitEntireOcean();
    }
  }

  fitEntireOcean(opts = {}) {
    if (!this.map) return;
    this.map.invalidateSize();
    const coords = [];

    this.allTargets.forEach(t => {
      const lat = Number(t.latitude != null ? t.latitude : t.lat);
      const lon = Number(t.longitude != null ? t.longitude : t.lon);
      if (!isNaN(lat) && !isNaN(lon) && lat !== 0 && lon !== 0 && t.latitude !== null && t.longitude !== null) {
        coords.push([lat, lon]);
      }
    });

    this.allTracks.forEach(trk => {
      const geom = trk.geometry || {};
      if (geom.coordinates) {
        geom.coordinates.forEach(c => coords.push([c[1], c[0]]));
      }
    });

    if (coords.length > 0) {
      const bounds = L.latLngBounds(coords);
      this.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
    } else {
      this.map.setView(this.lastCenter, this.lastZoom);
    }
  }

  fitAllDebris(opts = {}) {
    this.fitEntireOcean(opts);
  }

  fitSurveyTracks() {
    if (!this.map) return;
    const coords = [];
    this.allTracks.forEach(trk => {
      const geom = trk.geometry || {};
      if (geom.coordinates) {
        geom.coordinates.forEach(c => coords.push([c[1], c[0]]));
      }
    });

    if (coords.length > 0) {
      const bounds = L.latLngBounds(coords);
      this.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16 });
    } else {
      this.fitEntireOcean();
    }
  }

  // -----------------------------------------------------------------
  // Target Selection & Flying
  // -----------------------------------------------------------------
  focusTarget(targetId) {
    if (!this.map) return;
    let marker = this.markers[targetId] || this.currentScanMarkers[targetId];

    if (!marker) {
      // Find by matching target_id or object_id in markers
      const targetObj = this.allTargets.find(t => t.target_id === targetId || t.object_id === targetId || t.id === targetId) ||
                        this.currentScanTargets.find(t => t.target_id === targetId || t.object_id === targetId || t.id === targetId);
      if (targetObj && targetObj.target_id) {
        marker = this.markers[targetObj.target_id];
      }
    }

    if (marker && this.map) {
      this.map.flyTo(marker.getLatLng(), 17, { duration: 1.0 });
      marker.openPopup();
    }
  }

  selectTarget(targetId, options = {}) {
    this.focusTarget(targetId);
  }

  // -----------------------------------------------------------------
  // Offline Marine GIS Layers Ingestion
  // -----------------------------------------------------------------
  async loadLocalGISLayers() {
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
      console.warn("[GISMap] Local GIS habitat layers warning:", e);
    }
  }

  // Water Body & Marine Region Deduction Helper
  _inferWaterBody(lat, lon) {
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
  }

  // Formatting helpers
  formatCoordDeg(val, type) {
    if (val == null || isNaN(val)) return "--";
    const num = Number(val);
    const absVal = Math.abs(num).toFixed(5);
    return type === 'lat' ? `${absVal}° ${num >= 0 ? 'N' : 'S'}` : `${absVal}° ${num >= 0 ? 'E' : 'W'}`;
  }

  formatCoordinate(lat, lon) {
    if (lat == null || lon == null || isNaN(lat) || isNaN(lon)) return "Unreferenced Target";
    return `${this.formatCoordDeg(lat, 'lat')}, ${this.formatCoordDeg(lon, 'lon')}`;
  }
}

if (typeof window !== 'undefined') {
  window.GISMap = GISMap;
}
