"""
MAVLink Telemetry Bridge & Pixhawk Autopilot Safety Isolation.
Interfaces with ArduSub / Pixhawk over serial or UDP (port 14550) to ingest vehicle pose.
Injects STATUSTEXT advisory notifications into QGroundControl HUD upon debris confirmation,
while strictly isolating AI perception from flight-critical actuator commands.
"""

from typing import Dict, Any, Optional, Callable
import time
from edge.sensor_fusion import VehicleNavState, SensorFusionEngine


class MAVLinkSafetyBridge:
    """
    Subsea MAVLink listener for Pixhawk / ArduSub.
    Maintains autonomous perception boundary: reads telemetry, never sends flight commands.
    """
    def __init__(self, connection_str: str = "udpin:0.0.0.0:14550", fusion_engine: Optional[SensorFusionEngine] = None):
        self.connection_str = connection_str
        self.fusion = fusion_engine or SensorFusionEngine()
        self.is_connected = False
        self.last_heartbeat = 0.0

    def parse_mavlink_message(self, msg_type: str, msg_data: Dict[str, Any]):
        """
        Parses incoming MAVLink packets and routes them into the Sensor Fusion engine.
        """
        now = time.time()
        self.last_heartbeat = now
        self.is_connected = True

        if msg_type == "GLOBAL_POSITION_INT":
            # Scale: lat/lon in 1e7 degrees, relative_alt in mm
            lat = msg_data.get("lat", 301715400) / 1e7
            lon = msg_data.get("lon", -878235400) / 1e7
            depth_m = max(0.0, -msg_data.get("relative_alt", -18500) / 1000.0)
            hdg = msg_data.get("hdg", 9000) / 100.0

            state = VehicleNavState(
                timestamp=now,
                lat=lat,
                lon=lon,
                depth_m=depth_m,
                altitude_m=msg_data.get("alt_above_bottom", 5.0),
                heading_deg=hdg,
                roll_deg=msg_data.get("roll", 0.0),
                pitch_deg=msg_data.get("pitch", 0.0),
                surge_velocity_mps=msg_data.get("vx", 150) / 100.0,
                sway_velocity_mps=msg_data.get("vy", 0) / 100.0,
                nav_source="MAVLINK_DVL_DR",
                battery_pct=msg_data.get("battery_remaining", 85.0),
                water_temp_c=22.0,
                dvl_bottom_lock=True
            )
            self.fusion.inject_nav_packet(state)

    def generate_qgroundcontrol_alert(self, target_id: str, class_name: str, confidence: float, range_m: float) -> str:
        """
        Formats an advisory MAVLink STATUSTEXT message for the operator's QGroundControl console.
        """
        severity = "EMERGENCY" if "shipwreck" in class_name or "ordnance" in class_name else "WARNING"
        text = f"[SEA-SENTINEL {severity}] {target_id}: {class_name.upper()} ({confidence*100:.0f}%) at {range_m:.1f}m"
        # Truncate to standard MAVLink STATUSTEXT 50-character limit
        return text[:50]

    def enforce_safety_isolation(self, outgoing_packet: Dict[str, Any]) -> bool:
        """
        Strictly drops any packet attempting to command vehicle velocity, depth, or actuators.
        Returns True if packet is authorized (advisory only), False if rejected.
        """
        return SensorFusionEngine.validate_safety_isolation(outgoing_packet)
