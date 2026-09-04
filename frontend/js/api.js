/**
 * Sea Sentinel: API & Data Service
 * Connects to FastAPI backend (/api/...) with seamless offline/mock data fallback.
 */

const API_BASE_URL = "http://localhost:8000";

// Benchmark test dataset for immediate demonstration
const BENCHMARK_TARGETS = [
  {
    object_id: "TGT_001",
    class: "fishing_net",
    raw_confidence: 0.88,
    calibrated_confidence: 0.81,
    anomaly_status: "confirmed_debris",
    risk_score: "HIGH",
    latitude: 42.747402,
    longitude: -73.794567,
    length_m: 14.2,
    width_m: 5.8,
    area_sq_m: 82.36,
    reconstruction_error: 0.1245,
    shadow_verified: true,
    is_rock_cluster: false,
    position_uncertainty_m: 1.5,
    georeferencing_case: "A",
    pixel_bbox: { x1: 280, y1: 140, x2: 430, y2: 260 },
    explanation: {
      morphology_note: "Dispersed irregular acoustic backscatter mesh typical of synthetic polymer netting.",
      action_recommendation: "PRIORITY INTERVENTION: Schedule targeted ROV/AUV optical inspection and recovery planning to prevent wildlife entanglement.",
      executive_narrative: "Target TGT_001 categorized as 'fishing_net' with 81.0% calibrated confidence. Pronounced acoustic shadow confirms elevated benthic relief. Autoencoder error (0.1245) confirms man-made synthetic anomaly. Assigned HIGH ecological hazard."
    }
  },
  {
    object_id: "TGT_002",
    class: "pipeline_or_cable",
    raw_confidence: 0.82,
    calibrated_confidence: 0.77,
    anomaly_status: "confirmed_debris",
    risk_score: "HIGH",
    latitude: 42.748950,
    longitude: -73.792840,
    length_m: 38.6,
    width_m: 2.1,
    area_sq_m: 81.06,
    reconstruction_error: 0.1082,
    shadow_verified: true,
    is_rock_cluster: false,
    position_uncertainty_m: 1.5,
    georeferencing_case: "A",
    pixel_bbox: { x1: 680, y1: 220, x2: 1040, y2: 270 },
    explanation: {
      morphology_note: "Continuous linear/tubular acoustic signature with high aspect ratio.",
      action_recommendation: "ASSET MONITORING: Log pipeline corridor coordinate; inspect for bottom-trawling anchor drag damage.",
      executive_narrative: "Target TGT_002 categorized as 'pipeline_or_cable' with 77.0% calibrated confidence. Continuous linear backscatter with trailing shadow. Assigned HIGH navigation hazard."
    }
  },
  {
    object_id: "TGT_003_ROCK",
    class: "riprap_debris",
    raw_confidence: 0.58,
    calibrated_confidence: 0.04,
    anomaly_status: "noise_rejected",
    risk_score: "LOW",
    latitude: 42.746120,
    longitude: -73.796100,
    length_m: 3.2,
    width_m: 2.8,
    area_sq_m: 8.96,
    reconstruction_error: 0.0612,
    shadow_verified: false,
    is_rock_cluster: true,
    position_uncertainty_m: 1.5,
    georeferencing_case: "A",
    pixel_bbox: { x1: 150, y1: 300, x2: 190, y2: 340 },
    explanation: {
      morphology_note: "Dense clustered point highlights characteristic of natural rock moraines.",
      action_recommendation: "NATURAL GEOLOGY: Filtered by DBSCAN spatial cluster suppression; no action required.",
      executive_narrative: "Target TGT_003_ROCK identified as natural geological rock field; suppressed by DBSCAN density filter (confidence penalized to 4.0%)."
    }
  },
  {
    object_id: "TGT_004",
    class: "shipwreck_fragment",
    raw_confidence: 0.74,
    calibrated_confidence: 0.69,
    anomaly_status: "suspicious_anomaly",
    risk_score: "MEDIUM",
    latitude: 42.745500,
    longitude: -73.791500,
    length_m: 11.5,
    width_m: 7.2,
    area_sq_m: 82.80,
    reconstruction_error: 0.0965,
    shadow_verified: true,
    is_rock_cluster: false,
    position_uncertainty_m: 1.5,
    georeferencing_case: "A",
    pixel_bbox: { x1: 520, y1: 80, x2: 640, y2: 160 },
    explanation: {
      morphology_note: "Rectilinear geometric acoustic highlight with distinct relief shadow.",
      action_recommendation: "SUBSEA HAZARD: Log target for subsequent multi-beam verification pass.",
      executive_narrative: "Target TGT_004 categorized as 'shipwreck_fragment' (69.0% calibrated). Autoencoder MSE (0.0965) confirms anomaly. Assigned MEDIUM risk."
    }
  }
];

class SeaSentinelAPI {
  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async checkHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return await res.json();
    } catch (e) {
      // Backend offline
    }
    return { status: "offline", fallback_mode: true };
  }

  async fetchSamples() {
    try {
      const res = await fetch(`${this.baseUrl}/api/samples`, { signal: AbortSignal.timeout(2500) });
      if (res.ok) {
        const data = await res.json();
        if (data && data.samples && data.samples.length > 0) {
          return data.samples;
        }
      }
    } catch (e) {
      console.warn("Backend /api/samples unreachable, using fallback sample catalog.");
    }

    return [
      {
        id: "ghost_net_01",
        name: "Ghost Fishing Net Mesh",
        category: "fishing_net",
        risk_hint: "HIGH",
        filename: "quanzhou_HN_004.jpg",
        description: "Dispersed synthetic polymer netting with high acoustic backscatter highlight and acoustic void shadow.",
        georef_case: "A",
        simulated_coords: { lat: 42.747402, lon: -73.794567 }
      },
      {
        id: "pipeline_cable_01",
        name: "Subsea Pipeline / Cable",
        category: "pipeline_or_cable",
        risk_hint: "HIGH",
        filename: "dongying_POC_017.jpg",
        description: "Continuous linear acoustic signature with prominent relief shadow across seabed corridor.",
        georef_case: "A",
        simulated_coords: { lat: 42.748950, lon: -73.792840 }
      },
      {
        id: "rock_cluster_01",
        name: "Natural Seabed Moraine / Riprap",
        category: "riprap_debris",
        risk_hint: "LOW",
        filename: "quanzhou_RP_002.jpg",
        description: "Dense clustered geological rock formation; filtered and suppressed by DBSCAN spatial clustering.",
        georef_case: "A",
        simulated_coords: { lat: 42.746120, lon: -73.796100 }
      },
      {
        id: "engine_part_01",
        name: "Heavy Metallic Engine Debris",
        category: "engine_debris",
        risk_hint: "HIGH",
        filename: "dongying_EP_008.jpg",
        description: "High-density specular acoustic reflector with sharp boundary and distinct acoustic shadow trailing down-range.",
        georef_case: "A",
        simulated_coords: { lat: 42.745500, lon: -73.791500 }
      }
    ];
  }

  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${this.baseUrl}/api/upload`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }

    return await res.json();
  }

  async analyzeImage(imagePath, rasterMeta = null, navLog = null) {
    const payload = {
      image_path: imagePath,
      raster_meta: rasterMeta,
      nav_log: navLog
    };

    const res = await fetch(`${this.baseUrl}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Analysis failed" }));
      throw new Error(err.detail || `Analysis failed with status ${res.status}`);
    }

    return await res.json();
  }

  async getSurveyTargets() {
    try {
      const res = await fetch(`${this.baseUrl}/api/geospatial`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json();
        if (data && data.targets && data.targets.length > 0) {
          return data.targets;
        }
      }
    } catch (e) {
      console.warn("Geospatial targets API not reachable:", e);
    }
    return [];
  }
}

window.apiService = new SeaSentinelAPI();
