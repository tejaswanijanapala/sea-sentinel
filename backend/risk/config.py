"""
Sea Sentinel: IMO-Aligned Marine Debris Hazard Risk Assessment Engine
Configuration, Weights & Threshold Profiles

Scientific & Compliance Note:
This configuration provides the project-specific parameter weights, threshold boundaries,
and correlation profiles for Sea Sentinel's marine debris hazard model.
These are Sea Sentinel project-specific modeling choices aligned with IMO FSA principles
(MSC/Circ.1023 / MEPC/Circ.392). They are stored declaratively to allow future calibration
using empirical maritime incident archives and hydrodynamic field observations.
"""

from typing import Dict, Any

RISK_MODEL_VERSION = "SS-RISK-v1.0"
FRAMEWORK_DESIGNATION = "IMO-aligned project-specific marine debris hazard risk assessment framework based on the principles of IMO Formal Safety Assessment (FSA)"

# ---------------------------------------------------------------------
# Risk Level Thresholds (Sea Sentinel Project Thresholds, 0-100)
# ---------------------------------------------------------------------
RISK_THRESHOLDS = {
    "LOW": {"min": 0.0, "max": 39.99, "color": "#10b981", "badge": "LOW RISK"},
    "MODERATE": {"min": 40.0, "max": 59.99, "color": "#eab308", "badge": "MODERATE RISK"},
    "HIGH": {"min": 60.0, "max": 79.99, "color": "#f97316", "badge": "HIGH RISK"},
    "CRITICAL": {"min": 80.0, "max": 100.0, "color": "#ef4444", "badge": "CRITICAL RISK"}
}

# ---------------------------------------------------------------------
# Layer A: Hazard Severity Score (HSS) Weights
# ---------------------------------------------------------------------
HAZARD_SEVERITY_WEIGHTS = {
    "debris_type": 0.25,
    "size_volume": 0.20,
    "water_column_position": 0.10,
    "entanglement_potential": 0.20,
    "persistence": 0.10,
    "visibility_detectability": 0.15
}

# ---------------------------------------------------------------------
# Layer B: Likelihood / Exposure Score (LS) Weights
# ---------------------------------------------------------------------
LIKELIHOOD_WEIGHTS = {
    "navigation_exposure": 0.30,
    "depth_interaction": 0.15,
    "size_encounter_footprint": 0.15,
    "water_column_position": 0.10,
    "mobility": 0.10,
    "debris_density": 0.10,
    "recurrence": 0.05,
    "visibility_detectability": 0.05
}

# ---------------------------------------------------------------------
# Layer C: Consequence Context Profiles (Weight distribution per operational domain)
# ---------------------------------------------------------------------
CONSEQUENCE_CONTEXT_PROFILES = {
    "STANDARD_BALANCED": {
        "navigation": 0.35,
        "ecological": 0.35,
        "operational_economic": 0.15,
        "human_safety": 0.15
    },
    "COMMERCIAL_SHIPPING_CORRIDOR": {
        "navigation": 0.55,
        "ecological": 0.15,
        "operational_economic": 0.20,
        "human_safety": 0.10
    },
    "MARINE_PROTECTED_AREA": {
        "navigation": 0.10,
        "ecological": 0.65,
        "operational_economic": 0.15,
        "human_safety": 0.10
    },
    "PORT_APPROACH_HARBOR": {
        "navigation": 0.45,
        "ecological": 0.15,
        "operational_economic": 0.25,
        "human_safety": 0.15
    },
    "OFFSHORE_ENERGY_SUBSEA": {
        "navigation": 0.20,
        "ecological": 0.20,
        "operational_economic": 0.50,
        "human_safety": 0.10
    }
}

# ---------------------------------------------------------------------
# Context Weights for Final Overall Multi-Dimensional Risk
# ---------------------------------------------------------------------
MULTI_CONTEXT_RISK_WEIGHTS = {
    "navigation_risk": 0.35,
    "ecological_risk": 0.35,
    "operational_economic_risk": 0.15,
    "human_safety_risk": 0.15
}

# ---------------------------------------------------------------------
# Debris Type Baseline Severity & Inherent Traits
# ---------------------------------------------------------------------
DEBRIS_TYPE_PROFILES: Dict[str, Dict[str, Any]] = {
    "fishing_net": {
        "base_severity": 92.0,
        "entanglement_potential": 96.0,
        "persistence_score": 88.0,
        "mobility_class": "SUSPENDED_DRIFTING",
        "ecological_impact_factor": 1.5,
        "navigation_impact_factor": 1.4,
        "typical_water_column": "SUBSURFACE_MIDWATER"
    },
    "ghost_net": {
        "base_severity": 94.0,
        "entanglement_potential": 98.0,
        "persistence_score": 90.0,
        "mobility_class": "SUSPENDED_DRIFTING",
        "ecological_impact_factor": 1.6,
        "navigation_impact_factor": 1.4,
        "typical_water_column": "SUBSURFACE_MIDWATER"
    },
    "ghost_gear": {
        "base_severity": 94.0,
        "entanglement_potential": 98.0,
        "persistence_score": 90.0,
        "mobility_class": "SUSPENDED_DRIFTING",
        "ecological_impact_factor": 1.6,
        "navigation_impact_factor": 1.4,
        "typical_water_column": "SUBSURFACE_MIDWATER"
    },
    "container": {
        "base_severity": 95.0,
        "entanglement_potential": 30.0,
        "persistence_score": 92.0,
        "mobility_class": "SURFACE_FLOATING",
        "ecological_impact_factor": 1.2,
        "navigation_impact_factor": 1.9,
        "typical_water_column": "NEAR_SURFACE_FLOATING"
    },
    "cargo_container": {
        "base_severity": 95.0,
        "entanglement_potential": 30.0,
        "persistence_score": 92.0,
        "mobility_class": "SURFACE_FLOATING",
        "ecological_impact_factor": 1.2,
        "navigation_impact_factor": 1.9,
        "typical_water_column": "NEAR_SURFACE_FLOATING"
    },
    "pipeline_or_cable": {
        "base_severity": 82.0,
        "entanglement_potential": 75.0,
        "persistence_score": 95.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 1.1,
        "navigation_impact_factor": 1.4,
        "typical_water_column": "SEABED"
    },
    "pipe_cable": {
        "base_severity": 82.0,
        "entanglement_potential": 75.0,
        "persistence_score": 95.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 1.1,
        "navigation_impact_factor": 1.4,
        "typical_water_column": "SEABED"
    },
    "engine_debris": {
        "base_severity": 85.0,
        "entanglement_potential": 35.0,
        "persistence_score": 92.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 1.3,
        "navigation_impact_factor": 1.3,
        "typical_water_column": "SEABED"
    },
    "engine_or_machinery": {
        "base_severity": 85.0,
        "entanglement_potential": 35.0,
        "persistence_score": 92.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 1.3,
        "navigation_impact_factor": 1.3,
        "typical_water_column": "SEABED"
    },
    "shipwreck_fragment": {
        "base_severity": 90.0,
        "entanglement_potential": 65.0,
        "persistence_score": 96.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 1.2,
        "navigation_impact_factor": 1.7,
        "typical_water_column": "SEABED"
    },
    "shipwreck_or_hull": {
        "base_severity": 90.0,
        "entanglement_potential": 65.0,
        "persistence_score": 96.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 1.2,
        "navigation_impact_factor": 1.7,
        "typical_water_column": "SEABED"
    },
    "plastic_fragment": {
        "base_severity": 55.0,
        "entanglement_potential": 45.0,
        "persistence_score": 82.0,
        "mobility_class": "SUSPENDED_DRIFTING",
        "ecological_impact_factor": 1.3,
        "navigation_impact_factor": 0.5,
        "typical_water_column": "SUBSURFACE_MIDWATER"
    },
    "plastic_debris": {
        "base_severity": 55.0,
        "entanglement_potential": 45.0,
        "persistence_score": 82.0,
        "mobility_class": "SUSPENDED_DRIFTING",
        "ecological_impact_factor": 1.3,
        "navigation_impact_factor": 0.5,
        "typical_water_column": "SUBSURFACE_MIDWATER"
    },
    "tire_or_rubber": {
        "base_severity": 52.0,
        "entanglement_potential": 35.0,
        "persistence_score": 85.0,
        "mobility_class": "SLOW_BEDLOAD",
        "ecological_impact_factor": 1.1,
        "navigation_impact_factor": 0.6,
        "typical_water_column": "SEABED"
    },
    "riprap_debris": {
        "base_severity": 50.0,
        "entanglement_potential": 20.0,
        "persistence_score": 90.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 0.9,
        "navigation_impact_factor": 0.7,
        "typical_water_column": "SEABED"
    },
    "drum_or_barrel": {
        "base_severity": 82.0,
        "entanglement_potential": 25.0,
        "persistence_score": 80.0,
        "mobility_class": "SURFACE_FLOATING",
        "ecological_impact_factor": 1.6,
        "navigation_impact_factor": 1.3,
        "typical_water_column": "NEAR_SURFACE_FLOATING"
    },
    "marine_plastic_drum": {
        "base_severity": 82.0,
        "entanglement_potential": 25.0,
        "persistence_score": 80.0,
        "mobility_class": "SURFACE_FLOATING",
        "ecological_impact_factor": 1.6,
        "navigation_impact_factor": 1.3,
        "typical_water_column": "NEAR_SURFACE_FLOATING"
    },
    "munitions_or_uxo": {
        "base_severity": 99.0,
        "entanglement_potential": 10.0,
        "persistence_score": 98.0,
        "mobility_class": "STATIONARY_SEABED",
        "ecological_impact_factor": 1.8,
        "navigation_impact_factor": 2.0,
        "typical_water_column": "SEABED"
    },
    "marine_debris": {
        "base_severity": 65.0,
        "entanglement_potential": 50.0,
        "persistence_score": 75.0,
        "mobility_class": "SLOW_BEDLOAD",
        "ecological_impact_factor": 1.0,
        "navigation_impact_factor": 1.0,
        "typical_water_column": "SEABED"
    }
}

# ---------------------------------------------------------------------
# Water Column Position Multipliers
# ---------------------------------------------------------------------
WATER_COLUMN_FACTORS = {
    "NEAR_SURFACE_FLOATING": {
        "score": 95.0,
        "nav_exposure_multiplier": 1.8,
        "eco_exposure_multiplier": 1.1,
        "human_safety_multiplier": 1.5
    },
    "SUBSURFACE_MIDWATER": {
        "score": 75.0,
        "nav_exposure_multiplier": 1.4,
        "eco_exposure_multiplier": 1.4,
        "human_safety_multiplier": 1.2
    },
    "SEABED": {
        "score": 35.0,
        "nav_exposure_multiplier": 0.6,
        "eco_exposure_multiplier": 1.0,
        "human_safety_multiplier": 0.7
    }
}

# ---------------------------------------------------------------------
# Mobility Multipliers
# ---------------------------------------------------------------------
MOBILITY_PROFILES = {
    "SURFACE_FLOATING": {"score": 95.0, "drift_factor": 1.0},
    "SUSPENDED_DRIFTING": {"score": 80.0, "drift_factor": 0.75},
    "SLOW_BEDLOAD": {"score": 35.0, "drift_factor": 0.15},
    "STATIONARY_SEABED": {"score": 10.0, "drift_factor": 0.0}
}

# ---------------------------------------------------------------------
# Sensitive Habitat Proximity Impact Profile
# ---------------------------------------------------------------------
HABITAT_SENSITIVITIES = {
    "CORAL_REEF": {"base_sensitivity": 95.0, "critical_radius_m": 500.0},
    "MARINE_PROTECTED_AREA": {"base_sensitivity": 90.0, "critical_radius_m": 1000.0},
    "SEAGRASS": {"base_sensitivity": 80.0, "critical_radius_m": 400.0},
    "MANGROVE_ESTUARY": {"base_sensitivity": 85.0, "critical_radius_m": 600.0},
    "PSSA": {"base_sensitivity": 92.0, "critical_radius_m": 1500.0},
    "NONE": {"base_sensitivity": 20.0, "critical_radius_m": 100.0}
}
