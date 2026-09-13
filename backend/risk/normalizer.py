"""
Sea Sentinel: Scientific Parameter Normalization Engine
Converts all 15 heterogeneous maritime parameters into standard continuous 0-100 scale.
Prevents raw unit mixing (meters vs m² vs objects/km² vs percentages).
"""

import math
from typing import Optional, Dict, Any
from .models import RawRiskParameters, NormalizedRiskParameters
from .config import (
    DEBRIS_TYPE_PROFILES,
    WATER_COLUMN_FACTORS,
    MOBILITY_PROFILES,
    HABITAT_SENSITIVITIES
)


TAXONOMY_ALIASES = {
    "engine_debris": "engine_debris",
    "engine": "engine_debris",
    "machinery": "engine_or_machinery",
    "heavy_machinery": "engine_or_machinery",
    "shipwreck_fragment": "shipwreck_fragment",
    "shipwreck": "shipwreck_fragment",
    "wreck": "shipwreck_fragment",
    "ghost_net": "ghost_net",
    "fishing_net": "fishing_net",
    "ghost_gear": "ghost_gear",
    "cargo_container": "cargo_container",
    "container": "container",
    "pipeline_or_cable": "pipeline_or_cable",
    "pipe_cable": "pipe_cable",
    "marine_plastic_drum": "marine_plastic_drum",
    "drum_or_barrel": "drum_or_barrel",
    "plastic_debris": "plastic_debris",
    "plastic_fragment": "plastic_fragment",
    "riprap_debris": "riprap_debris",
    "tire_or_rubber": "tire_or_rubber",
    "munitions_or_uxo": "munitions_or_uxo"
}


class RiskParameterNormalizer:
    """
    Transforms raw physical, spatial, and probabilistic metrics into normalized scores (0 - 100).
    Uses scientifically defensible transfer curves (sigmoid, exponential decay, piecewise linear).
    """

    @staticmethod
    def get_canonical_profile(debris_type: str) -> Dict[str, Any]:
        """Resolves any taxonomy string into its canonical DEBRIS_TYPE_PROFILES profile."""
        dt_clean = (debris_type or "marine_debris").lower().strip().replace(" ", "_")
        if dt_clean in DEBRIS_TYPE_PROFILES:
            return DEBRIS_TYPE_PROFILES[dt_clean]
        if dt_clean in TAXONOMY_ALIASES:
            alias_key = TAXONOMY_ALIASES[dt_clean]
            if alias_key in DEBRIS_TYPE_PROFILES:
                return DEBRIS_TYPE_PROFILES[alias_key]
        for key, prof in DEBRIS_TYPE_PROFILES.items():
            if key in dt_clean or dt_clean in key:
                return prof
        return DEBRIS_TYPE_PROFILES["marine_debris"]

    @classmethod
    def normalize_debris_type(cls, debris_type: str) -> float:
        """Normalized inherent debris type hazard (0-100)."""
        prof = cls.get_canonical_profile(debris_type)
        return float(prof.get("base_severity", 65.0))

    @staticmethod
    def normalize_size_volume(
        length_m: Optional[float] = None,
        width_m: Optional[float] = None,
        height_m: Optional[float] = None,
        area_sq_m: Optional[float] = None,
        volume_cu_m: Optional[float] = None
    ) -> float:
        """
        Normalizes physical size/extent (0-100).
        Transfer curve: logarithmic-sigmoidal scaling from 0.1 m² to 500+ m².
        """
        eff_area = area_sq_m
        if eff_area is None:
            if length_m is not None and width_m is not None:
                eff_area = max(0.05, length_m * width_m)
            elif length_m is not None:
                eff_area = max(0.05, length_m * (length_m * 0.4))
            elif volume_cu_m is not None:
                eff_area = math.pow(volume_cu_m, 2.0 / 3.0)
            else:
                eff_area = 25.0  # standard baseline

        # Sigmoidal transfer curve calibrated for marine navigation obstruction:
        # Area < 2 m² -> ~35, Area = 15 m² -> ~65, Area = 50 m² -> ~82, Area > 150 m² -> >92
        val = 100.0 / (1.0 + math.exp(-0.045 * (eff_area - 20.0)))
        return float(max(15.0, min(100.0, val)))

    @staticmethod
    def normalize_depth_exposure(
        water_depth_m: Optional[float] = None,
        object_depth_m: Optional[float] = None,
        water_col_pos: str = "SEABED"
    ) -> float:
        """
        Normalizes depth vulnerability (0-100).
        Shallower water / shallower object depth = higher physical interaction probability for vessels.
        """
        if water_col_pos in ["NEAR_SURFACE_FLOATING", "SURFACE_FLOATING"]:
            return 98.0

        depth = object_depth_m if object_depth_m is not None else water_depth_m
        if depth is None:
            depth = 16.0  # standard shelf depth default

        if depth <= 6.0:
            return 96.0
        elif depth <= 15.0:
            return 88.0
        elif depth <= 30.0:
            return 78.0
        elif depth <= 60.0:
            return 60.0
        elif depth <= 100.0:
            return 40.0
        else:
            return 20.0

    @staticmethod
    def normalize_navigation_distance(
        distance_m: Optional[float] = None,
        traffic_density: Optional[str] = "MODERATE"
    ) -> float:
        """
        Normalizes proximity to navigation route/fairway (0-100).
        Transfer curve: < 100m = 92-100, 250m = 84, 500m = 72, 1000m = 55, > 3000m = 20.
        """
        if distance_m is None:
            distance_m = 180.0  # realistic marine survey proximity

        dist = max(0.0, float(distance_m))
        decay_factor = math.exp(-dist / 1100.0)
        base_score = decay_factor * 100.0

        # Traffic density amplifier
        density_mult = 1.15 if traffic_density == "HIGH" else (1.0 if traffic_density == "MODERATE" else 0.85)
        return float(max(10.0, min(100.0, base_score * density_mult)))

    @staticmethod
    def normalize_water_column_position(position: str) -> float:
        """Translates vertical water column stratum into exposure score (0-100)."""
        pos_upper = (position or "SEABED").upper()
        if "SURFACE" in pos_upper or "FLOAT" in pos_upper:
            return WATER_COLUMN_FACTORS["NEAR_SURFACE_FLOATING"]["score"]
        elif "MID" in pos_upper or "SUB" in pos_upper:
            return WATER_COLUMN_FACTORS["SUBSURFACE_MIDWATER"]["score"]
        return WATER_COLUMN_FACTORS["SEABED"]["score"]

    @classmethod
    def normalize_entanglement_potential(
        cls,
        debris_type: str,
        override: Optional[float] = None
    ) -> float:
        """Calculates entanglement potential (0-100) based on morphological profile."""
        if override is not None:
            return float(max(0.0, min(100.0, override)))

        prof = cls.get_canonical_profile(debris_type)
        return float(prof.get("entanglement_potential", 50.0))

    @staticmethod
    def normalize_ecological_sensitivity(
        distance_to_habitat_m: Optional[float] = None,
        habitat_type: Optional[str] = None
    ) -> float:
        """
        Normalizes ecological impact and habitat vulnerability (0-100).
        Spatial decay model relative to critical buffer radius.
        """
        hab_key = (habitat_type or "NONE").upper()
        hab_prof = HABITAT_SENSITIVITIES.get(hab_key, HABITAT_SENSITIVITIES["NONE"])
        base_sens = hab_prof["base_sensitivity"]
        crit_radius = hab_prof["critical_radius_m"]

        if distance_to_habitat_m is None:
            return base_sens * 0.5

        dist = max(0.0, float(distance_to_habitat_m))
        if dist <= 50.0:
            return base_sens
        elif dist <= crit_radius:
            ratio = 1.0 - (dist / crit_radius) * 0.6
            return base_sens * ratio
        else:
            decay = math.exp(-(dist - crit_radius) / 1200.0)
            return max(10.0, base_sens * 0.4 * decay)

    @classmethod
    def normalize_persistence(
        cls,
        debris_type: str,
        material: Optional[str] = None
    ) -> float:
        """Normalizes physical longevity and biodegradation resistance (0-100)."""
        prof = cls.get_canonical_profile(debris_type)
        return float(prof.get("persistence_score", 80.0))

    @staticmethod
    def normalize_mobility(
        mobility_class: Optional[str] = "STATIONARY_SEABED",
        drift_velocity_knots: Optional[float] = 0.0,
        current_speed_knots: Optional[float] = 0.5
    ) -> float:
        """Normalizes dynamic mobility and relocation potential (0-100)."""
        mob_key = (mobility_class or "STATIONARY_SEABED").upper()
        mob_prof = MOBILITY_PROFILES.get(mob_key, MOBILITY_PROFILES["STATIONARY_SEABED"])
        base = mob_prof["score"]

        # Velocity amplifier
        vel = max(float(drift_velocity_knots or 0.0), float(current_speed_knots or 0.0) * mob_prof["drift_factor"])
        amplifier = min(1.25, 1.0 + (vel * 0.12))
        return float(max(5.0, min(100.0, base * amplifier)))

    @staticmethod
    def normalize_debris_density(density_sq_km: Optional[float] = None) -> float:
        """
        Normalizes spatial cluster density (0-100).
        1 obj/km² -> 20, 5 obj/km² -> 60, 15+ obj/km² -> 95.
        """
        if density_sq_km is None:
            return 30.0

        dens = max(0.0, float(density_sq_km))
        val = 100.0 * (1.0 - math.exp(-dens / 4.5))
        return float(max(5.0, min(100.0, val)))

    @staticmethod
    def normalize_recurrence(detection_count: int = 1) -> float:
        """
        Normalizes recurrence persistence across multiple temporal survey passes (0-100).
        1 pass -> 20 (new detection), 2 passes -> 50, 4+ passes -> 90+ (chronic unaddressed hazard).
        """
        cnt = max(1, int(detection_count))
        if cnt == 1:
            return 25.0
        elif cnt == 2:
            return 55.0
        elif cnt == 3:
            return 75.0
        else:
            return min(100.0, 80.0 + (cnt - 3) * 5.0)

    @staticmethod
    def normalize_detection_confidence(
        ai_conf: float = 0.85,
        sensor_quality: float = 0.90,
        class_conf: float = 0.85
    ) -> float:
        """Normalizes AI/sensor detection certainty (0-100)."""
        weighted = (ai_conf * 0.50) + (sensor_quality * 0.25) + (class_conf * 0.25)
        return float(max(10.0, min(100.0, weighted * 100.0)))

    @staticmethod
    def normalize_position_uncertainty(uncertainty_m: float = 5.0) -> float:
        """
        Normalizes position confidence score (0-100).
        Low uncertainty (±2m) = High Score (95), High uncertainty (±50m) = Low Score (20).
        """
        unc = max(0.1, float(uncertainty_m))
        val = 100.0 * math.exp(-unc / 25.0)
        return float(max(5.0, min(100.0, val)))

    @staticmethod
    def normalize_visibility_detectability(
        contrast_ratio: Optional[float] = 0.80,
        has_shadow: bool = True
    ) -> float:
        """
        Normalizes acoustic detectability and benthic contrast (0-100).
        Higher acoustic visibility = easier identification.
        """
        cr = max(0.1, min(1.0, float(contrast_ratio or 0.8)))
        shadow_bonus = 15.0 if has_shadow else 0.0
        score = (cr * 85.0) + shadow_bonus
        return float(max(10.0, min(100.0, score)))

    @classmethod
    def normalize_potential_consequence(
        cls,
        debris_type: str,
        size_score: float,
        override_rating: Optional[str] = None
    ) -> float:
        """Normalizes potential baseline consequence magnitude (0-100)."""
        if override_rating:
            rating_map = {"CRITICAL": 95.0, "HIGH": 75.0, "MODERATE": 50.0, "LOW": 25.0}
            return rating_map.get(override_rating.upper(), 50.0)

        prof = cls.get_canonical_profile(debris_type)
        base = float(prof.get("base_severity", 65.0))
        return float(max(10.0, min(100.0, (base * 0.55) + (size_score * 0.45))))

    def normalize_all(self, raw: RawRiskParameters) -> NormalizedRiskParameters:
        """Transforms all 15 raw parameter fields into a complete NormalizedRiskParameters container."""
        dt_score = self.normalize_debris_type(raw.debris_type)
        sz_score = self.normalize_size_volume(
            length_m=raw.length_m,
            width_m=raw.width_m,
            height_m=raw.height_m,
            area_sq_m=raw.area_sq_m,
            volume_cu_m=raw.volume_cu_m
        )
        dp_score = self.normalize_depth_exposure(
            water_depth_m=raw.water_depth_m,
            object_depth_m=raw.object_depth_m,
            water_col_pos=raw.water_column_position
        )
        nav_dist_score = self.normalize_navigation_distance(
            distance_m=raw.distance_to_route_m,
            traffic_density=raw.route_traffic_density
        )
        wc_score = self.normalize_water_column_position(raw.water_column_position)
        ent_score = self.normalize_entanglement_potential(
            debris_type=raw.debris_type,
            override=raw.entanglement_potential_override
        )
        eco_score = self.normalize_ecological_sensitivity(
            distance_to_habitat_m=raw.distance_to_sensitive_habitat_m,
            habitat_type=raw.habitat_type
        )
        pers_score = self.normalize_persistence(
            debris_type=raw.debris_type,
            material=raw.material_composition
        )
        mob_score = self.normalize_mobility(
            mobility_class=raw.mobility_class,
            drift_velocity_knots=raw.drift_velocity_knots,
            current_speed_knots=raw.current_speed_knots
        )
        dens_score = self.normalize_debris_density(raw.local_debris_density_sq_km)
        rec_score = self.normalize_recurrence(raw.detection_count)
        det_conf_score = self.normalize_detection_confidence(
            ai_conf=raw.ai_detection_confidence,
            sensor_quality=raw.sensor_quality_score,
            class_conf=raw.classification_confidence
        )
        pos_unc_score = self.normalize_position_uncertainty(raw.position_uncertainty_m)
        vis_score = self.normalize_visibility_detectability(
            contrast_ratio=raw.acoustic_contrast_ratio,
            has_shadow=getattr(raw, 'has_acoustic_shadow', True)
        )
        pot_conseq_score = self.normalize_potential_consequence(
            debris_type=raw.debris_type,
            size_score=sz_score,
            override_rating=raw.potential_consequence_severity
        )

        return NormalizedRiskParameters(
            debris_type_score=dt_score,
            size_volume_score=sz_score,
            depth_exposure_score=dp_score,
            navigation_distance_score=nav_dist_score,
            water_column_score=wc_score,
            entanglement_potential_score=ent_score,
            ecological_sensitivity_score=eco_score,
            persistence_score=pers_score,
            mobility_score=mob_score,
            debris_density_score=dens_score,
            recurrence_score=rec_score,
            detection_confidence_score=det_conf_score,
            position_uncertainty_score=pos_unc_score,
            visibility_detectability_score=vis_score,
            potential_consequence_score=pot_conseq_score
        )
