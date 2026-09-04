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
  async checkHealth() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) return await res.json();
    } catch (e) {
      // Backend offline
    }
    return { status: "offline", fallback_mode: true };
  }

  async getSurveyTargets() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/geospatial`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json();
        if (data && data.targets && data.targets.length > 0) {
          return data.targets;
        }
      }
    } catch (e) {
      // Use benchmark dataset
    }
    return BENCHMARK_TARGETS;
  }
}

window.apiService = new SeaSentinelAPI();
