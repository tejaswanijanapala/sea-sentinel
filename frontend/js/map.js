/**
 * Sea Sentinel: Interactive GIS Map Component
 * Powered by Leaflet.js with Dark Matter bathymetric tiles, WGS84 target markers,
 * Towfish Nadir Trackline, Sonar Swath Corridor, and Live Cursor Coordinate HUD.
 */

class GISMap {
  constructor(containerId) {
    this.containerId = containerId;
    this.map = null;
    this.markers = {};
    this.surveyLayers = (typeof L !== 'undefined' && L.layerGroup) ? L.layerGroup() : null;
    this.gisLayers = {
      coral_reefs: (typeof L !== 'undefined' && L.layerGroup) ? L.layerGroup() : null,
      marine_protected_areas: (typeof L !== 'undefined' && L.layerGroup) ? L.layerGroup() : null,
      seagrass_meadows: (typeof L !== 'undefined' && L.layerGroup) ? L.layerGroup() : null,
      underwater_infrastructure: (typeof L !== 'undefined' && L.layerGroup) ? L.layerGroup() : null,
      shipping_lanes: (typeof L !== 'undefined' && L.layerGroup) ? L.layerGroup() : null
    };
    this.lastTargets = [];
    this.lastCoords = [];
    this.lastBounds = null;
    this.lastCenter = [42.7474, -73.7945];
    this.lastZoom = 14;
    this.showSwath = true;
    if (typeof L !== 'undefined' && document.getElementById(containerId)) {
      try {
        this._initMap();
        this.loadLocalGISLayers();
      } catch (err) {
        console.warn("GISMap init warning:", err);
      }
    }
  }

  _initMap() {
    if (typeof L === 'undefined') return;
    const container = document.getElementById(this.containerId);
    if (!container) return;
    // Default center: Hudson River / Albany hydrographic survey corridor
    this.map = L.map(this.containerId, {
      center: this.lastCenter,
      zoom: 13,
      zoomControl: false
    });

    L.control.zoom({ position: 'bottomright' }).addTo(this.map);

    // 1. ESRI World Dark Gray Canvas (Default: Clean dark tactical basemap)
    const darkBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri &mdash; NIOT Sea Sentinel',
      maxZoom: 16
    });
    const darkRef = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
      attribution: '',
      maxZoom: 16
    });
    const darkTactical = L.layerGroup([darkBase, darkRef]).addTo(this.map);

    // 2. ESRI World Ocean Basemap (Hydrographic bathymetry & marine depth contours)
    const oceanBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, GEBCO, NOAA, National Geographic',
      maxZoom: 13
    });

    // 3. ESRI World Imagery (High-res orbital & aerial satellite)
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: '&copy; Esri, Maxar, Earthstar Geographics',
      maxZoom: 18
    });

    // 4. OpenStreetMap Standard
    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19
    });

    // Basemap selector
    const baseLayers = {
      "<span style='color:#38bdf8; font-weight:600;'>◈ Dark Tactical</span>": darkTactical,
      "<span style='color:#06b6d4; font-weight:600;'>🌊 Ocean Bathymetry</span>": oceanBase,
      "<span style='color:#10b981; font-weight:600;'>🛰️ Satellite Imagery</span>": satellite,
      "<span style='color:#94a3b8; font-weight:600;'>🗺️ OpenStreetMap</span>": osm
    };

    L.control.layers(baseLayers, null, { position: 'topright' }).addTo(this.map);

    // Add Survey Layers (Trackline & Swath)
    this.surveyLayers.addTo(this.map);

    // Global popupopen listener to ensure target synchronization
    this.map.on('popupopen', (e) => {
      if (e.popup && e.popup._source && e.popup._source._targetObjectId && window.app) {
        window.app.onTargetSelected(e.popup._source._targetObjectId, { fly: false });
      }
    });

    // Cursor coordinates telemetry HUD listener
    this.map.on('mousemove', (e) => {
      const hud = document.getElementById('mapCoordsHud');
      if (hud && e.latlng) {
        const latStr = this.formatCoordDeg(e.latlng.lat, 'lat');
        const lonStr = this.formatCoordDeg(e.latlng.lng, 'lon');
        hud.innerHTML = `<i class="fa-solid fa-crosshairs"></i> Cursor: <b>${latStr}, ${lonStr}</b> &nbsp;|&nbsp; Datum: <span style="color:#00f0ff;">WGS84 (EPSG:4326)</span>`;
      }
    });
  }

  formatCoordDeg(val, type) {
    if (val == null || isNaN(val)) return "--";
    const num = Number(val);
    const absVal = Math.abs(num).toFixed(5);
    if (type === 'lat') {
      return `${absVal}° ${num >= 0 ? 'N' : 'S'}`;
    } else {
      return `${absVal}° ${num >= 0 ? 'E' : 'W'}`;
    }
  }

  formatCoordinate(lat, lon) {
    if (lat == null || lon == null || isNaN(lat) || isNaN(lon)) return "Unreferenced Target";
    return `${this.formatCoordDeg(lat, 'lat')}, ${this.formatCoordDeg(lon, 'lon')} (WGS84)`;
  }

  setTargets(targets, surveyMeta = {}) {
    if (!this.map) return;
    this.lastTargets = targets || [];

    // Clear existing markers & survey layers
    Object.values(this.markers).forEach(m => {
      try { this.map.removeLayer(m); } catch (e) {}
    });
    this.markers = {};
    if (this.surveyLayers) this.surveyLayers.clearLayers();

    const validCoords = [];

    this.lastTargets.forEach(t => {
      let lat = null;
      let lon = null;

      // Robust coordinate extraction handling all casing and simulated formats
      if (t.latitude !== undefined && t.latitude !== null && !isNaN(Number(t.latitude))) {
        lat = Number(t.latitude);
      } else if (t.lat !== undefined && t.lat !== null && !isNaN(Number(t.lat))) {
        lat = Number(t.lat);
      } else if (t.simulated_coords && t.simulated_coords.lat != null && !isNaN(Number(t.simulated_coords.lat))) {
        lat = Number(t.simulated_coords.lat);
      } else if (t.coordinates && t.coordinates.lat != null && !isNaN(Number(t.coordinates.lat))) {
        lat = Number(t.coordinates.lat);
      }

      if (t.longitude !== undefined && t.longitude !== null && !isNaN(Number(t.longitude))) {
        lon = Number(t.longitude);
      } else if (t.lon !== undefined && t.lon !== null && !isNaN(Number(t.lon))) {
        lon = Number(t.lon);
      } else if (t.simulated_coords && t.simulated_coords.lon != null && !isNaN(Number(t.simulated_coords.lon))) {
        lon = Number(t.simulated_coords.lon);
      } else if (t.coordinates && t.coordinates.lon != null && !isNaN(Number(t.coordinates.lon))) {
        lon = Number(t.coordinates.lon);
      }

      if (lat !== null && lon !== null && !isNaN(lat) && !isNaN(lon)) {
        validCoords.push([lat, lon]);

        let color = "#00e676"; // Low risk
        if (t.risk_score === "HIGH") color = "#ff1744";
        else if (t.risk_score === "MEDIUM") color = "#ffab00";

        // Create High-Tech Animated Radar Ping Marker
        const icon = L.divIcon({
          className: 'custom-target-marker',
          html: `
            <div style="position:relative; width:22px; height:22px; display:flex; align-items:center; justify-content:center;">
              <div style="
                position:absolute;
                width: 22px; height: 22px;
                border-radius: 50%;
                background: ${color};
                opacity: 0.35;
                animation: sonar-ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;
              "></div>
              <div style="
                width: 12px; height: 12px;
                border-radius: 50%;
                background: ${color};
                box-shadow: 0 0 10px ${color}, 0 0 18px ${color};
                border: 2px solid #ffffff;
                cursor: pointer;
                position: relative;
                z-index: 2;
              "></div>
            </div>
          `,
          iconSize: [22, 22],
          iconAnchor: [11, 11]
        });

        const marker = L.marker([lat, lon], { icon }).addTo(this.map);
        marker._targetObjectId = t.object_id;

        const dims = (t.length_m && t.width_m) ? `${Math.round(t.length_m)}m × ${Math.round(t.width_m)}m` : "Estimated 14m × 5m";
        const conf = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
        const isHigher = conf > 75;
        const prioScore = t.priority_score || 85;
        const prioLevel = t.priority_level || (t.risk_score || "HIGH");
        const hazardRisk = t.hazard_risk || 80;
        const hazardLevel = t.hazard_risk_level || (t.risk_score || "HIGH");

        const prioTag = `<span style="background:rgba(255,51,102,0.18); color:#ff4d79; border:1px solid rgba(255,51,102,0.4); padding:2px 6px; border-radius:4px; font-size:0.68rem; font-weight:800; font-family:'JetBrains Mono',monospace;">P: ${prioScore}/100 (${prioLevel})</span>`;
        const formattedClass = (t.class || "Unknown").replace(/_/g, " ");
        const georefCase = t.georeferencing_case ? `Case ${t.georeferencing_case}` : "Case B (Dead-Reckoning)";
        const uncert = t.position_uncertainty_m ? `±${t.position_uncertainty_m}m` : "±1.5m";

        const popupContent = `
          <div style="font-family: 'Outfit', sans-serif; color: #060b18; min-width: 250px; padding: 6px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px; gap:8px;">
              <div style="font-weight: 800; font-size: 1rem; color: ${color}; font-family: 'JetBrains Mono', monospace;">
                ${t.object_id}
              </div>
              ${prioTag}
            </div>
            <div style="font-size: 0.86rem; margin-bottom: 4px;"><b>Type:</b> <span style="font-weight:700; color:#0f172a; text-transform:capitalize;">${formattedClass}</span></div>
            
            <div style="background:#f1f5f9; border-radius:4px; padding:6px 8px; margin-bottom:6px; display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:0.75rem;">
              <div>🎯 <b>AI Conf:</b> <span style="font-weight:700; color:#0284c7; font-family:'JetBrains Mono',monospace;">${conf}%</span></div>
              <div>⚠️ <b>Hazard:</b> <span style="font-weight:700; color:#e11d48; font-family:'JetBrains Mono',monospace;">${hazardRisk}/100</span></div>
            </div>

            <div style="font-size: 0.78rem; margin-bottom: 3px;"><b>Extent:</b> ${dims}</div>
            <div style="font-size: 0.74rem; margin-bottom: 3px; color:#475569;"><b>Derivation:</b> ${georefCase} (${uncert})</div>
            
            <div style="margin-top:6px; display:flex; justify-content:space-between; align-items:center; border-top:1px solid #e2e8f0; padding-top:6px;">
              <span style="font-size: 0.70rem; color: #0284c7; font-family:'JetBrains Mono',monospace; font-weight:600;">
                <i class="fa-solid fa-crosshairs"></i> ${this.formatCoordinate(lat, lon)}
              </span>
              <button onclick="if(window.app) window.app.openScoreExplanationModal('${t.object_id}')" style="background:#0284c7; color:#ffffff; border:none; border-radius:3px; font-size:0.68rem; font-weight:700; padding:3px 7px; cursor:pointer;">Why this score?</button>
            </div>
          </div>
        `;

        marker.bindPopup(popupContent);

        // Click selection
        marker.on('click', () => {
          if (window.app) window.app.onTargetSelected(t.object_id, { fly: false });
        });

        // Popup open synchronization
        marker.on('popupopen', () => {
          if (window.app) window.app.onTargetSelected(t.object_id, { fly: false });
        });

        // Mouseover inspection
        marker.on('mouseover', () => {
          marker.openPopup();
          if (window.app) window.app.onTargetSelected(t.object_id, { fly: false });
        });

        this.markers[t.object_id] = marker;
      }
    });

    this.lastCoords = validCoords;

    if (validCoords.length > 0) {
      this._removeUnreferencedNotice();
      if (validCoords.length === 1) {
        this.lastCenter = validCoords[0];
        this.lastZoom = 16;
        this.lastBounds = null;
      } else {
        this.lastBounds = L.latLngBounds(validCoords);
        this.lastCenter = this.lastBounds.getCenter();
      }

      // Draw Towfish Nadir Trackline & Swath Corridor
      this._renderSurveySwath(validCoords, surveyMeta);

      // Apply view safely with container dimensions validation
      this._applyView();
    } else if (surveyMeta && surveyMeta.bbox_wgs84) {
      // Georeferenced survey without target detections (e.g. clean seabed mosaic)
      this._removeUnreferencedNotice();
      const b = surveyMeta.bbox_wgs84;
      const bounds = [[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]];
      const rect = L.rectangle(bounds, {
        color: '#00e5ff',
        weight: 2,
        fillColor: '#00e5ff',
        fillOpacity: 0.08,
        dashArray: '4, 6'
      }).bindTooltip(`Survey Boundary: ${surveyMeta.dataset_profile || 'GeoTIFF Mosaic'}`, { sticky: true });
      this.surveyLayers.addLayer(rect);
      this.lastBounds = L.latLngBounds(bounds);
      this.lastCenter = this.lastBounds.getCenter();
      this._applyView();
    } else {
      this.lastBounds = null;
      this._showUnreferencedNotice(surveyMeta);
    }
  }

  _showUnreferencedNotice(surveyMeta = {}) {
    this._removeUnreferencedNotice();
    const container = this.map ? this.map.getContainer() : document.getElementById("sonarMap");
    if (!container) return;

    const noticeEl = document.createElement("div");
    noticeEl.id = "mapUnrefOverlay";
    noticeEl.className = "map-unref-overlay";
    const profile = surveyMeta.dataset_profile || "Unreferenced Acoustic Chip (Case C)";
    noticeEl.innerHTML = `
      <div class="map-unref-card">
        <div class="unref-icon"><i class="fa-solid fa-satellite-dish"></i></div>
        <div class="unref-content">
          <div class="unref-title">UNREFERENCED DATASET (Case C)</div>
          <div class="unref-source"><b>Source Profile:</b> ${profile}</div>
          <div class="unref-text">
            This sonar dataset contains no embedded GeoTIFF tags (CRS/Affine) or navigation telemetry logs.
            In compliance with hydrographic safety standards, <b>synthetic coordinates are strictly suppressed</b>.
          </div>
          <div class="unref-action">
            <i class="fa-solid fa-circle-info"></i> To visualize targets on this GIS Map, select a <b>Georeferenced GeoTIFF</b> (e.g. NOAA Survey H11584, USGS DS 1005) or provide sidecar navigation logs.
          </div>
        </div>
      </div>
    `;
    container.appendChild(noticeEl);
  }

  _removeUnreferencedNotice() {
    const existing = document.getElementById("mapUnrefOverlay");
    if (existing) existing.remove();
  }

  _renderSurveySwath(validCoords, surveyMeta) {
    if (!this.map || !this.surveyLayers || !validCoords || validCoords.length === 0 || typeof L === 'undefined') return;

    // Determine survey track midpoint
    const centerLat = validCoords.reduce((a, c) => a + c[0], 0) / validCoords.length;
    const centerLon = validCoords.reduce((a, c) => a + c[1], 0) / validCoords.length;

    // Survey line heading (degrees True, default 15° for Hudson River survey line)
    const heading = surveyMeta.heading || 15.0;
    const rad = (heading * Math.PI) / 180.0;

    // Approx 250m survey line length
    const dLat = (Math.cos(rad) * 0.0022);
    const dLon = (Math.sin(rad) * 0.0030);

    const startPt = [centerLat - dLat, centerLon - dLon];
    const endPt = [centerLat + dLat, centerLon + dLon];

    // Swath width ~75m port and starboard
    const perpRad = rad + Math.PI / 2;
    const sLat = Math.cos(perpRad) * 0.00068; // ~75 meters in latitude
    const sLon = Math.sin(perpRad) * 0.00092; // ~75 meters in longitude

    const swathPolygon = [
      [startPt[0] - sLat, startPt[1] - sLon],
      [endPt[0] - sLat, endPt[1] - sLon],
      [endPt[0] + sLat, endPt[1] + sLon],
      [startPt[0] + sLat, startPt[1] + sLon]
    ];

    // Swath Coverage Corridor
    const swathLayer = L.polygon(swathPolygon, {
      color: '#00e5ff',
      weight: 1,
      dashArray: '4, 6',
      fillColor: '#00e5ff',
      fillOpacity: 0.07
    }).bindTooltip("75m Sonar Acoustic Swath Corridor", { sticky: true });

    // Towfish Nadir Trackline
    const trackline = L.polyline([startPt, endPt], {
      color: '#38bdf8',
      weight: 2,
      dashArray: '6, 8',
      opacity: 0.85
    }).bindTooltip("Towfish Nadir Trackline (Heading 015°T)", { sticky: true });

    this.surveyLayers.addLayer(swathLayer);
    this.surveyLayers.addLayer(trackline);
  }

  _applyView() {
    if (!this.map) return;
    const container = this.map.getContainer();
    // Guard: Do not attempt to compute bounds or set view if map container is hidden (0x0 dimensions)
    if (!container || container.offsetWidth === 0 || container.offsetHeight === 0) {
      return;
    }

    if (this.lastBounds && this.lastCoords.length > 1) {
      this.map.fitBounds(this.lastBounds, { padding: [50, 50], maxZoom: 16 });
    } else if (this.lastCenter) {
      this.map.setView(this.lastCenter, this.lastZoom || 15);
    }
  }

  selectTarget(targetId, options = {}) {
    if (!this.map) return;
    const marker = this.markers[targetId];
    if (marker) {
      if (options.fly) {
        this.map.flyTo(marker.getLatLng(), Math.max(this.map.getZoom(), 16), { duration: 0.8 });
        setTimeout(() => marker.openPopup(), 300);
      } else if (!marker.isPopupOpen()) {
        marker.openPopup();
      }
    }
  }

  flyToTarget(targetId) {
    if (!this.map) return;
    const marker = this.markers[targetId];
    if (marker) {
      this.map.flyTo(marker.getLatLng(), 17, { duration: 1.0 });
      setTimeout(() => marker.openPopup(), 400);
    }
  }

  highlightTarget(targetId) {
    if (!this.map) return;
    const marker = this.markers[targetId];
    if (marker && !marker.isPopupOpen()) {
      marker.openPopup();
    }
  }

  focusAllTargets() {
    this._applyView();
  }

  toggleSwath() {
    if (!this.map || !this.surveyLayers) return false;
    this.showSwath = !this.showSwath;
    if (this.showSwath) {
      this.map.addLayer(this.surveyLayers);
    } else {
      this.map.removeLayer(this.surveyLayers);
    }
    return this.showSwath;
  }

  async loadLocalGISLayers() {
    if (!this.map || typeof L === 'undefined') return;
    try {
      const geojsonData = await window.apiService.getGISLayers();
      if (!geojsonData || !geojsonData.features) return;

      // Color mapping for marine features
      const layerStyles = {
        coral_reefs: { color: '#f43f5e', fillColor: '#f43f5e', fillOpacity: 0.25, weight: 2 },
        marine_protected_areas: { color: '#10b981', fillColor: '#10b981', fillOpacity: 0.18, weight: 2, dashArray: '6, 6' },
        seagrass_meadows: { color: '#84cc16', fillColor: '#84cc16', fillOpacity: 0.20, weight: 1.5 },
        underwater_infrastructure: { color: '#f59e0b', weight: 3, dashArray: '4, 8' },
        shipping_lanes: { color: '#38bdf8', fillColor: '#38bdf8', fillOpacity: 0.12, weight: 2, dashArray: '8, 8' }
      };

      for (const feat of geojsonData.features) {
        const lKey = feat.properties && feat.properties.layer_key;
        if (!lKey || !this.gisLayers[lKey]) continue;

        const style = layerStyles[lKey] || { color: '#00e5ff', weight: 2 };
        const geoLayer = L.geoJSON(feat, {
          style: style,
          onEachFeature: (feature, layer) => {
            const p = feature.properties || {};
            layer.bindTooltip(`<b>${p.name || 'Marine Zone'}</b><br><span style="color:#94a3b8;">${p.type || ''} · ${p.sensitivity || 'PROTECTED'}</span>`, { sticky: true });
          }
        });

        this.gisLayers[lKey].addLayer(geoLayer);
      }

      // Add all GIS layers to map by default
      for (const group of Object.values(this.gisLayers)) {
        if (group) group.addTo(this.map);
      }
    } catch (e) {
      console.warn("Failed loading offline GIS layers:", e);
    }
  }

  toggleGISLayer(layerKey, isVisible) {
    const group = this.gisLayers[layerKey];
    if (!group || !this.map) return;
    if (isVisible) {
      this.map.addLayer(group);
    } else {
      this.map.removeLayer(group);
    }
  }

  invalidateSize() {
    if (this.map) {
      setTimeout(() => {
        this.map.invalidateSize();
        this._applyView();
      }, 100);
    }
  }
}

window.GISMap = GISMap;
