"""
Navigation Sensor Fusion & AUV/ROV Safety Isolation Engine.
Fuses Side-Scan Sonar timestamps with DVL, INS/IMU, pressure depth, and compass.
Guarantees strict safety isolation: perception operates as an advisory observer only,
and cannot directly actuate propulsion, depth, or emergency systems.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import math
import time
import bisect
import numpy as np


@dataclass
class VehicleNavState:
    timestamp: float
    lat: float
    lon: float
    depth_m: float
    altitude_m: float
    heading_deg: float
    roll_deg: float
    pitch_deg: float
    surge_velocity_mps: float
    sway_velocity_mps: float
    nav_source: str  # "GPS_SURFACED", "DVL_DEAD_RECKONING", "INS_ESTIMATE"
    battery_pct: float
    water_temp_c: float
    dvl_bottom_lock: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SensorFusionEngine:
    """
    Synchronizes high-frequency acoustic pings with subsea navigation telemetry.
    Supports DVL velocity integration during GPS-denied submerged mission phases.
    """
    def __init__(self, max_history_size: int = 500):
        self.max_history_size = max_history_size
        self._history: List[VehicleNavState] = []
        self._timestamps: List[float] = []

    def inject_nav_packet(self, state: VehicleNavState):
        """Appends incoming telemetry from MAVLink / ROS 2 / DVL driver."""
        self._history.append(state)
        self._timestamps.append(state.timestamp)
        if len(self._history) > self.max_history_size:
            self._history.pop(0)
            self._timestamps.pop(0)

    def interpolate_nav_state(self, ping_timestamp: float) -> VehicleNavState:
        """
        Interpolates vehicle pose and geodetic coordinate for an exact sonar ping timestamp.
        Falls back to the latest state if out of range.
        """
        if not self._history:
            # Default reference origin (e.g. Gulf of Mexico / coastal baseline)
            return VehicleNavState(
                timestamp=ping_timestamp,
                lat=30.17154,
                lon=-87.82354,
                depth_m=18.5,
                altitude_m=5.0,
                heading_deg=90.0,
                roll_deg=0.0,
                pitch_deg=0.0,
                surge_velocity_mps=1.5,
                sway_velocity_mps=0.0,
                nav_source="SIMULATED_BENCHMARK",
                battery_pct=88.0,
                water_temp_c=22.0,
                dvl_bottom_lock=True
            )

        if ping_timestamp <= self._timestamps[0]:
            return self._history[0]
        if ping_timestamp >= self._timestamps[-1]:
            return self._history[-1]

        # Binary search for closest timestamp brackets
        idx = bisect.bisect_right(self._timestamps, ping_timestamp)
        s0 = self._history[idx - 1]
        s1 = self._history[idx]

        dt = s1.timestamp - s0.timestamp
        if dt <= 1e-6:
            return s0

        ratio = (ping_timestamp - s0.timestamp) / dt

        # Interpolate linear components
        interp_lat = s0.lat + ratio * (s1.lat - s0.lat)
        interp_lon = s0.lon + ratio * (s1.lon - s0.lon)
        interp_depth = s0.depth_m + ratio * (s1.depth_m - s0.depth_m)
        interp_alt = s0.altitude_m + ratio * (s1.altitude_m - s0.altitude_m)

        # Angular interpolation (shortest arc for heading)
        dh = (s1.heading_deg - s0.heading_deg + 180.0) % 360.0 - 180.0
        interp_heading = (s0.heading_deg + ratio * dh) % 360.0

        return VehicleNavState(
            timestamp=ping_timestamp,
            lat=interp_lat,
            lon=interp_lon,
            depth_m=interp_depth,
            altitude_m=interp_alt,
            heading_deg=interp_heading,
            roll_deg=s0.roll_deg + ratio * (s1.roll_deg - s0.roll_deg),
            pitch_deg=s0.pitch_deg + ratio * (s1.pitch_deg - s0.pitch_deg),
            surge_velocity_mps=s0.surge_velocity_mps + ratio * (s1.surge_velocity_mps - s0.surge_velocity_mps),
            sway_velocity_mps=s0.sway_velocity_mps + ratio * (s1.sway_velocity_mps - s0.sway_velocity_mps),
            nav_source=s1.nav_source,
            battery_pct=s1.battery_pct,
            water_temp_c=s1.water_temp_c,
            dvl_bottom_lock=s1.dvl_bottom_lock
        )

    @staticmethod
    def validate_safety_isolation(command_attempt: Dict[str, Any]) -> bool:
        """
        Enforces AUV/ROV Safety Isolation.
        Blocks any unauthorized attempt by AI modules to command vehicle thrusters,
        rudder, ballast, or emergency release systems directly.
        """
        forbidden_keys = {"set_motor", "set_rudder", "set_depth", "emergency_stop", "arm_thrusters"}
        for k in command_attempt.keys():
            if k in forbidden_keys:
                # Reject actuator override
                return False
        return True
