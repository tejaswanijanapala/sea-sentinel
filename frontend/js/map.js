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
    // Default center: Hudson River / Albany survey area (42.747°N, -73.794°W)
    this.map = L.map(this.containerId, {
      center: [42.7474, -73.7945],
      zoom: 15,
      zoomControl: false
    });

    L.control.zoom({ position: 'bottomright' }).addTo(this.map);

    // Dark Matter Nautical Basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap & CartoDB | NIOT Sea Sentinel',
      subdomains: 'abcd',
      maxZoom: 19
    }).addTo(this.map);
  }

  setTargets(targets) {
    // Clear existing markers
    Object.values(this.markers).forEach(m => this.map.removeLayer(m));
    this.markers = {};

    const validCoords = [];

    targets.forEach(t => {
      if (t.latitude && t.longitude) {
        const lat = t.latitude;
        const lon = t.longitude;
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
            "></div>
          `,
          iconSize: [16, 16],
          iconAnchor: [8, 8]
        });

        const marker = L.marker([lat, lon], { icon }).addTo(this.map);

        const dims = (t.length_m && t.width_m) ? `${t.length_m}m × ${t.width_m}m` : "Unavailable";
        const popupContent = `
          <div style="font-family: 'Outfit', sans-serif; color: #060b18; min-width: 180px;">
            <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 4px; color: ${color};">
              ${t.object_id}
            </div>
            <div style="font-size: 0.8rem; margin-bottom: 2px;"><b>Class:</b> ${t.class}</div>
            <div style="font-size: 0.8rem; margin-bottom: 2px;"><b>Dimensions:</b> ${dims}</div>
            <div style="font-size: 0.8rem; margin-bottom: 2px;"><b>Risk:</b> <span style="font-weight:600; color:${color};">${t.risk_score}</span></div>
            <div style="font-size: 0.72rem; color: #555; margin-top: 4px;">(${lat.toFixed(5)}°N, ${lon.toFixed(5)}°W)</div>
          </div>
        `;

        marker.bindPopup(popupContent);
        marker.on('click', () => {
          if (window.app) window.app.onTargetSelected(t.object_id);
        });

        this.markers[t.object_id] = marker;
      }
    });

    if (validCoords.length > 0) {
      this.map.fitBounds(L.latLngBounds(validCoords), { padding: [40, 40] });
    }
  }

  flyToTarget(targetId) {
    const marker = this.markers[targetId];
    if (marker) {
      this.map.flyTo(marker.getLatLng(), 17, { duration: 1.2 });
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
