/**
 * Sea Sentinel: Interactive GIS Map Component
 * Powered by Leaflet.js with Dark Matter bathymetric tiles and WGS84 target markers.
 */

class GISMap {
  constructor(containerId) {
    this.containerId = containerId;
    this.map = null;
    this.markers = {};
    this._initMap();
  }

  _initMap() {
    // Default center: Hudson River / Albany survey corridor (42.747°N, -73.794°W)
    this.map = L.map(this.containerId, {
      center: [42.7474, -73.7945],
      zoom: 13,
      zoomControl: false
    });

    L.control.zoom({ position: 'bottomright' }).addTo(this.map);

    // 1. ESRI World Dark Gray Canvas (Default: Clean dark tactical basemap, ZERO API key, NO watermark)
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

    // Global popupopen listener to ensure target synchronization
    this.map.on('popupopen', (e) => {
      if (e.popup && e.popup._source && e.popup._source._targetObjectId && window.app) {
        window.app.onTargetSelected(e.popup._source._targetObjectId, { fly: false });
      }
    });
  }

  setTargets(targets) {
    // Clear existing markers
    Object.values(this.markers).forEach(m => this.map.removeLayer(m));
    this.markers = {};

    const validCoords = [];

    targets.forEach(t => {
      const lat = (t.latitude !== undefined && t.latitude !== null) ? Number(t.latitude) : null;
      const lon = (t.longitude !== undefined && t.longitude !== null) ? Number(t.longitude) : null;

      if (lat !== null && lon !== null && !isNaN(lat) && !isNaN(lon)) {
        validCoords.push([lat, lon]);

        let color = "#00e676";
        if (t.risk_score === "HIGH") color = "#ff1744";
        else if (t.risk_score === "MEDIUM") color = "#ffab00";

        // Create Custom SVG Pulse Marker
        const icon = L.divIcon({
          className: 'custom-target-marker',
          html: `
            <div style="
              width: 16px; height: 16px;
              border-radius: 50%;
              background: ${color};
              box-shadow: 0 0 10px ${color}, 0 0 20px ${color};
              border: 2px solid #ffffff;
              cursor: pointer;
            "></div>
          `,
          iconSize: [16, 16],
          iconAnchor: [8, 8]
        });

        const marker = L.marker([lat, lon], { icon }).addTo(this.map);
        marker._targetObjectId = t.object_id;

        const dims = (t.length_m && t.width_m) ? `${t.length_m}m × ${t.width_m}m` : "Unavailable";
        const conf = Math.round((t.calibrated_confidence || t.confidence || 0) * 100);
        const isHigher = conf > 75;
        const prioTag = isHigher 
          ? '<span style="background:rgba(0,240,255,0.18); color:#00f0ff; border:1px solid rgba(0,240,255,0.4); padding:2px 6px; border-radius:4px; font-size:0.68rem; font-weight:700;">▲ HIGHER (&gt;75%)</span>'
          : '<span style="background:rgba(148,163,184,0.18); color:#94a3b8; border:1px solid rgba(148,163,184,0.3); padding:2px 6px; border-radius:4px; font-size:0.68rem; font-weight:700;">▼ LOWER (≤75%)</span>';
        const formattedClass = (t.class || "Unknown").replace(/_/g, " ");

        const popupContent = `
          <div style="font-family: 'Outfit', sans-serif; color: #060b18; min-width: 210px; padding: 4px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px; gap:8px;">
              <div style="font-weight: 700; font-size: 0.95rem; color: ${color}; font-family: 'JetBrains Mono', monospace;">
                ${t.object_id}
              </div>
              ${prioTag}
            </div>
            <div style="font-size: 0.85rem; margin-bottom: 3px;"><b>Class:</b> <span style="font-weight:700; color:#0f172a; text-transform:capitalize;">${formattedClass}</span></div>
            <div style="font-size: 0.8rem; margin-bottom: 3px;"><b>Confidence:</b> <span style="font-weight:600; font-family:'JetBrains Mono',monospace;">${conf}%</span></div>
            <div style="font-size: 0.8rem; margin-bottom: 3px;"><b>Dimensions:</b> ${dims}</div>
            <div style="font-size: 0.8rem; margin-bottom: 3px;"><b>Risk Level:</b> <span style="font-weight:700; color:${color}; font-family:'JetBrains Mono',monospace;">${t.risk_score || 'LOW'}</span></div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px; border-top:1px solid #e2e8f0; padding-top:4px;">
              <i class="fa-solid fa-location-dot"></i> ${lat.toFixed(5)}°N, ${lon.toFixed(5)}°W (WGS84)
            </div>
          </div>
        `;

        marker.bindPopup(popupContent);

        // 1. Click selection
        marker.on('click', () => {
          if (window.app) window.app.onTargetSelected(t.object_id, { fly: false });
        });

        // 2. Popup open synchronization
        marker.on('popupopen', () => {
          if (window.app) window.app.onTargetSelected(t.object_id, { fly: false });
        });

        // 3. Mouseover / Pointing out synchronization
        marker.on('mouseover', () => {
          marker.openPopup();
          if (window.app) window.app.onTargetSelected(t.object_id, { fly: false });
        });

        this.markers[t.object_id] = marker;
      }
    });

    if (validCoords.length > 0) {
      if (validCoords.length === 1) {
        this.map.setView(validCoords[0], 15);
      } else {
        this.map.fitBounds(L.latLngBounds(validCoords), { padding: [40, 40], maxZoom: 16 });
      }
    }
  }

  flyToTarget(targetId) {
    const marker = this.markers[targetId];
    if (marker) {
      this.map.flyTo(marker.getLatLng(), 17, { duration: 1.2 });
      marker.openPopup();
    }
  }

  highlightTarget(targetId) {
    const marker = this.markers[targetId];
    if (marker && !marker.isPopupOpen()) {
      marker.openPopup();
    }
  }

  invalidateSize() {
    if (this.map) {
      setTimeout(() => this.map.invalidateSize(), 150);
    }
  }
}

window.GISMap = GISMap;
