"""
Sea Sentinel: Dynamic Mobility & Drift Prediction Engine
Forecasts future debris trajectory and dispersion uncertainty cones at 6h, 12h, 24h, and 48h.
Evaluates dynamic future navigation and ecological exposure.
"""

import math
from typing import List, Optional, Tuple, Dict, Any
from .models import RawRiskParameters, DriftProjection
from .config import MOBILITY_PROFILES


class DriftPredictor:
    """
    Simulates hydrodynamic and wind-driven particle drift for mobile marine debris.
    Forecasts positions at +6h, +12h, +24h, and +48h with expanding spatial uncertainty ellipses.
    """

    EARTH_RADIUS_M = 6371000.0  # WGS84 mean radius

    @classmethod
    def offset_lat_lon(cls, lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
        """Calculates destination point given distance (m) and bearing (deg) using great-circle approximation."""
        rad_lat = math.radians(lat)
        rad_lon = math.radians(lon)
        rad_bearing = math.radians(bearing_deg)
        angular_dist = distance_m / cls.EARTH_RADIUS_M

        dest_lat = math.asin(
            math.sin(rad_lat) * math.cos(angular_dist) +
            math.cos(rad_lat) * math.sin(angular_dist) * math.cos(rad_bearing)
        )
        dest_lon = rad_lon + math.atan2(
            math.sin(rad_bearing) * math.sin(angular_dist) * math.cos(rad_lat),
            math.cos(angular_dist) - math.sin(rad_lat) * math.sin(dest_lat)
        )
        return math.degrees(dest_lat), math.degrees(dest_lon)

    def forecast_drift(
        self,
        raw: RawRiskParameters,
        current_lat: Optional[float] = None,
        current_lon: Optional[float] = None
    ) -> List[DriftProjection]:
        """
        Generates 4 temporal drift horizons: 6h, 12h, 24h, and 48h.
        """
        lat = current_lat if current_lat is not None else raw.latitude
        lon = current_lon if current_lon is not None else raw.longitude

        # If stationary seabed object or no coordinates, return minimal stationary projections
        mob_key = (raw.mobility_class or "STATIONARY_SEABED").upper()
        is_stationary = "STATIONARY" in mob_key or (raw.water_column_position == "SEABED" and "DRIFT" not in mob_key)

        if lat == null_safe(None) or lon == null_safe(None) or is_stationary:
            return []

        mob_prof = MOBILITY_PROFILES.get(mob_key, MOBILITY_PROFILES["STATIONARY_SEABED"])
        drift_factor = mob_prof.get("drift_factor", 0.0)
        if drift_factor <= 0.01:
            return []

        # Hydrodynamic & Leeway Drift Vector Calculation
        # V_drift = Current_velocity * drag_coeff + Wind_velocity * leeway_factor
        curr_knots = float(raw.current_speed_knots or 0.5)
        curr_bearing = float(raw.current_direction_deg or 90.0)
        wind_knots = float(raw.wind_speed_knots or 10.0)
        wind_bearing = float(raw.wind_direction_deg or 90.0)

        # Leeway factor: surface floating items catch ~3% of wind, subsurface catch ~0.5%
        leeway_coeff = 0.03 if raw.water_column_position == "NEAR_SURFACE_FLOATING" else 0.005

        # Vector addition in nautical meters/sec
        # 1 knot = 0.514444 m/s
        u_curr = (curr_knots * 0.514444) * math.sin(math.radians(curr_bearing))
        v_curr = (curr_knots * 0.514444) * math.cos(math.radians(curr_bearing))

        u_wind = (wind_knots * 0.514444 * leeway_coeff) * math.sin(math.radians(wind_bearing))
        v_wind = (wind_knots * 0.514444 * leeway_coeff) * math.cos(math.radians(wind_bearing))

        u_total = (u_curr * drift_factor) + u_wind
        v_total = (v_curr * drift_factor) + v_wind

        total_drift_speed_mps = math.sqrt(u_total**2 + v_total**2)
        drift_bearing_deg = (math.degrees(math.atan2(u_total, v_total)) + 360.0) % 360.0

        base_unc_m = max(5.0, float(raw.position_uncertainty_m or 5.0))
        dist_to_route = float(raw.distance_to_route_m) if raw.distance_to_route_m is not None else 1000.0
        dist_to_habitat = float(raw.distance_to_sensitive_habitat_m) if raw.distance_to_sensitive_habitat_m is not None else 1500.0

        projections = []
        horizons = [6, 12, 24, 48]

        for h in horizons:
            elapsed_sec = h * 3600.0
            distance_travelled_m = total_drift_speed_mps * elapsed_sec

            # Compute projected coordinate
            proj_lat, proj_lon = self.offset_lat_lon(lat, lon, distance_travelled_m, drift_bearing_deg)

            # Expanding Gaussian uncertainty cone (diffusion + velocity uncertainty)
            expanded_unc_m = base_unc_m + (distance_travelled_m * 0.15)

            # Simulated future exposure dynamics
            # If drifting toward route (e.g. initial distance decreases)
            future_nav_dist = max(0.0, dist_to_route - (distance_travelled_m * 0.5))
            future_nav_exp = min(100.0, max(10.0, math.exp(-future_nav_dist / 650.0) * 100.0))

            future_eco_dist = max(0.0, dist_to_habitat - (distance_travelled_m * 0.3))
            future_eco_exp = min(100.0, max(10.0, math.exp(-future_eco_dist / 1000.0) * 100.0))

            intersects_route = (future_nav_dist <= expanded_unc_m + 30.0)
            intersects_habitat = (future_eco_dist <= expanded_unc_m + 50.0)

            projections.append(
                DriftProjection(
                    horizon_hours=h,
                    projected_latitude=round(proj_lat, 6),
                    projected_longitude=round(proj_lon, 6),
                    uncertainty_radius_m=round(expanded_unc_m, 1),
                    future_navigation_exposure=round(future_nav_exp, 1),
                    future_ecological_exposure=round(future_eco_exp, 1),
                    intersects_route=intersects_route,
                    intersects_habitat=intersects_habitat
                )
            )

        return projections


def null_safe(val):
    return val
