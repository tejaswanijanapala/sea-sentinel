"""
Low-Bandwidth Acoustic Telemetry Encoder for Underwater AUV/ROV Modems.
Compresses confirmed marine-debris detection events into ultra-compact 24-byte binary packets
(with CRC-8 integrity) suitable for subsea acoustic modems (EvoLogics, Subnero, WHOI Micro-Modem).
"""

from typing import Dict, Any, Optional, Tuple
import struct
import time


class AcousticTelemetryEncoder:
    """
    Encodes mission-critical debris alerts into an ultra-compact 24-byte binary payload.
    Schema:
      [Sync: 1B] [Type: 1B] [Target_ID: 2B] [Class_ID: 1B] [Confidence: 1B]
      [Range: 2B] [Bearing: 2B] [Rel_X: 3B] [Rel_Y: 3B] [Depth: 2B]
      [Timestamp: 4B] [Uncertainty: 1B] [CRC8: 1B]
      Total = 24 Bytes (strictly within 16-32B underwater modem envelope)
    """
    SYNC_BYTE = 0xA5
    PACKET_TYPE_DEBRIS = 0x01

    CLASS_MAP = {
        "ghost_fishing_net": 1,
        "fishing_net": 1,
        "metal_container": 2,
        "container": 2,
        "pipeline_cable": 3,
        "pipeline": 3,
        "sunken_tire": 4,
        "tire": 4,
        "plastic_debris": 5,
        "shipwreck": 6,
        "unknown_anomaly": 7,
        "unknown": 7
    }
    REV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

    @staticmethod
    def _crc8(data: bytes) -> int:
        """Computes Dallas/Maxim CRC-8 (polynomial 0x31)."""
        crc = 0x00
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x31) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    def encode(
        self,
        target_id_num: int,
        class_name: str,
        confidence: float,
        slant_range_m: float,
        bearing_deg: float,
        local_x_m: float,
        local_y_m: float,
        depth_m: float,
        timestamp_s: Optional[float] = None,
        uncertainty_m: float = 1.5
    ) -> bytes:
        """
        Packs a detection event into exactly 24 bytes.
        """
        ts = int(timestamp_s or time.time()) & 0xFFFFFFFF
        class_id = self.CLASS_MAP.get(class_name.lower().replace(" ", "_"), 7)
        conf_byte = int(round(confidence * 100)) & 0xFF
        range_short = int(round(slant_range_m * 10)) & 0xFFFF
        bearing_short = int(round(bearing_deg * 10)) & 0xFFFF

        # Quantize relative coordinates (0.1m precision, stored in 24-bit signed int)
        x_quant = int(round(local_x_m * 10))
        y_quant = int(round(local_y_m * 10))
        x_bytes = (x_quant).to_bytes(3, byteorder="big", signed=True)
        y_bytes = (y_quant).to_bytes(3, byteorder="big", signed=True)

        depth_short = int(round(depth_m * 10)) & 0xFFFF
        unc_byte = min(255, int(round(uncertainty_m * 10))) & 0xFF

        # Pack header and standard fields
        # >BBHB BHH (1 + 1 + 2 + 1 + 1 + 2 + 2 = 10 bytes)
        prefix = struct.pack(
            ">BBHBBHH",
            self.SYNC_BYTE,
            self.PACKET_TYPE_DEBRIS,
            target_id_num & 0xFFFF,
            class_id,
            conf_byte,
            range_short,
            bearing_short
        )

        # >HI B (2 + 4 + 1 = 7 bytes)
        suffix = struct.pack(
            ">HIB",
            depth_short,
            ts,
            unc_byte
        )

        # Body: 10 + 3 + 3 + 7 = 23 bytes
        payload_body = prefix + x_bytes + y_bytes + suffix

        # Compute CRC-8 over the 23 bytes
        crc = self._crc8(payload_body)
        packet = payload_body + struct.pack("B", crc)

        assert len(packet) == 24, f"Packet size must be 24 bytes, got {len(packet)}"
        return packet

    def to_hex(self, packet: bytes) -> str:
        """Returns uppercase hexadecimal string for acoustic modem logging."""
        return packet.hex().upper()

    def decode(self, packet: bytes) -> Dict[str, Any]:
        """
        Unpacks a 24-byte acoustic packet and validates the CRC-8 checksum.
        """
        if len(packet) != 24:
            raise ValueError(f"Expected 24-byte packet, got {len(packet)} bytes")

        # Validate CRC-8
        computed_crc = self._crc8(packet[:23])
        packet_crc = packet[23]
        crc_valid = (computed_crc == packet_crc)

        prefix = struct.unpack(">BBHBBHH", packet[:10])
        sync, p_type, tid, class_id, conf_pct, range_x10, bear_x10 = prefix

        rel_x_m = int.from_bytes(packet[10:13], byteorder="big", signed=True) / 10.0
        rel_y_m = int.from_bytes(packet[13:16], byteorder="big", signed=True) / 10.0

        depth_x10, ts, unc_x10 = struct.unpack(">HIB", packet[16:23])

        return {
            "sync_valid": sync == self.SYNC_BYTE,
            "packet_type": p_type,
            "target_id": f"TGT_{tid:04d}",
            "class": self.REV_CLASS_MAP.get(class_id, "unknown"),
            "confidence": round(conf_pct / 100.0, 2),
            "slant_range_m": round(range_x10 / 10.0, 1),
            "relative_bearing_deg": round(bear_x10 / 10.0, 1),
            "local_x_east_m": round(rel_x_m, 1),
            "local_y_north_m": round(rel_y_m, 1),
            "depth_m": round(depth_x10 / 10.0, 1),
            "timestamp": ts,
            "uncertainty_m": round(unc_x10 / 10.0, 1),
            "crc_valid": crc_valid,
            "packet_bytes": len(packet)
        }
