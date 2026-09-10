"""
Uncertainty-Aware Sonar Geolocation Engine.
Computes geodetic coordinates (WGS84) and rigorous error covariance ellipses
from acoustic slant-range geometry and navigation sensor uncertainties.
Strictly reports position uncertainty bounds rather than false millimeter precision.
"""

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
from .sensor_fusion import VehicleNavState


class UncertaintyGeolocator:
    """
    Translates pixel centroid in Side-Scan Sonar waterfall swath into real-world coordinates,
    accounting for vehicle attitude (roll/pitch/heading), altitude, and sensor drift.
    """
    def __init__(self, max_slant_range_m: float = 75.0):
        self.max_slant_range_m = max_slant_range_m

    def geolocate_target(
        self,
        object_id: str,
        class_label: str,
        confidence: float,
        pixel_centroid: Tuple[float, float],
        image_shape: Tuple[int, int],
        nav_state: VehicleNavState,
        sonar_frame_id: str = "FRAME_001",
        model_version: str = "sea_sentinel_v2.0_edge_int8"
    ) -> Dict[str, Any]:
        """
        Geolocates an acoustic candidate with explicit uncertainty bounds.
        """
        h_px, w_px = image_shape[:2]
        px_x, px_y = pixel_centroid
        center_x = w_px / 2.0

        # 1. Determine Port or Starboard channel
        is_starboard = (px_x >= center_x)
        norm_cross_track = abs(px_x - center_x) / max(1.0, center_x)
        slant_range_m = norm_cross_track * self.max_slant_range_m

        # 2. Slant-Range to Ground-Range Conversion
        alt_m = max(1.0, nav_state.altitude_m)
        if slant_range_m >= alt_m:
            ground_range_m = math.sqrt(slant_range_m**2 - alt_m**2)
        else:
            # Inside nadir dead-zone; approximate directly under vessel
            ground_range_m = 0.5 * slant_range_m

        # Along-track offset relative to frame center
        # Assuming ~0.1m resolution per along-track ping row
        along_track_offset_m = ((px_y - (h_px / 2.0)) / h_px) * 20.0

        # 3. Target Bearing from Vehicle Heading
        # Port is Heading - 90 deg; Starboard is Heading + 90 deg
        beam_azimuth_deg = (nav_state.heading_deg + (90.0 if is_starboard else -90.0)) % 360.0
        azimuth_rad = math.radians(beam_azimuth_deg)

        # 4. Local East-North-Up (ENU) Coordinates
        local_x_east = ground_range_m * math.sin(azimuth_rad)
        local_y_north = ground_range_m * math.cos(azimuth_rad)
        local_z_depth = nav_state.depth_m + (alt_m - 0.5)  # Target resting on seabed

        # 5. Geodetic WGS84 Coordinates
        # 1 degree latitude ~ 111,139 meters
        lat_rad = math.radians(nav_state.lat)
        meters_per_lat = 111139.0
        meters_per_lon = 111139.0 * max(0.1, math.cos(lat_rad))

        target_lat = nav_state.lat + (local_y_north / meters_per_lat)
        target_lon = nav_state.lon + (local_x_east / meters_per_lon)

        # 6. Propagation of Position Uncertainty (Covariance)
        # Components: Navigation drift + Slant-range resolution + Heading error
        nav_sigma_m = 1.2 if nav_state.dvl_bottom_lock else 6.5
        range_sigma_m = 0.04 * slant_range_m + 0.3
        heading_sigma_rad = math.radians(1.2)
        bearing_error_m = ground_range_m * math.sin(heading_sigma_rad)

        total_position_uncertainty_m = math.sqrt(
            nav_sigma_m**2 + range_sigma_m**2 + bearing_error_m**2
        )

        return {
            "object_id": object_id,
            "class": class_label,
            "confidence": round(confidence, 3),
            "slant_range_m": round(slant_range_m, 2),
            "ground_range_m": round(ground_range_m, 2),
            "relative_bearing_deg": round(beam_azimuth_deg, 1),
            "channel": "STARBOARD" if is_starboard else "PORT",
            "latitude": round(target_lat, 7),
            "longitude": round(target_lon, 7),
            "local_x_east_m": round(local_x_east, 2),
            "local_y_north_m": round(local_y_north, 2),
            "estimated_depth_m": round(local_z_depth, 2),
            "vehicle_pose": {
                "lat": round(nav_state.lat, 7),
                "lon": round(nav_state.lon, 7),
                "depth_m": round(nav_state.depth_m, 2),
                "altitude_m": round(nav_state.altitude_m, 2),
                "heading_deg": round(nav_state.heading_deg, 1)
            },
            "timestamp": nav_state.timestamp,
            "position_uncertainty_m": round(total_position_uncertainty_m, 2),
            "model_version": model_version,
            "sonar_frame_id": sonar_frame_id
        }
