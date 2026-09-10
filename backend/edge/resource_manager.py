"""
Edge Resource, Power & Thermal Manager with Graceful Degradation Levels (0-4).
Detects hardware architecture (NVIDIA Jetson Orin AGX/NX/Nano, Raspberry Pi 5, ARM),
monitors battery discharge and core temperatures, and dynamically throttles AI workload
to prevent thermal shutdown or vehicle power starvation.
"""

from typing import Dict, Any, Optional, Tuple
import os
import gc
import platform
import psutil
import torch


class PowerState:
    PERFORMANCE = "PERFORMANCE"   # Battery > 60%, Temp Normal
    BALANCED = "BALANCED"         # Battery 30-60%, Temp Normal
    POWER_SAVE = "POWER_SAVE"     # Battery 15-30%, Temp Elevated
    CRITICAL = "CRITICAL"         # Battery < 15%, Temp High


class DegradationLevel:
    LEVEL_0 = 0  # Full Parallel YOLO + U-Net + High Resolution + Dense Tiling
    LEVEL_1 = 1  # Standard YOLO + U-Net, Adaptive Resolution (640x640)
    LEVEL_2 = 2  # Lightweight YOLO + Sparse U-Net Verification
    LEVEL_3 = 3  # Detection Only (YOLO Nano single-pass, U-Net bypassed)
    LEVEL_4 = 4  # Minimalist Telemetry & Quality Logging Only (AI perception disabled)


class EdgeResourceManager:
    """
    Supervises system health, power draw, thermals, and memory pressure.
    Enforces deterministic safety boundaries for AUV/ROV edge computers.
    """
    def __init__(self, override_device: Optional[str] = None):
        self.device_profile = self._detect_hardware(override_device)
        self.current_degradation_level = DegradationLevel.LEVEL_1
        self.current_power_state = PowerState.BALANCED
        self.thermal_warning_temp_c = 72.0
        self.thermal_critical_temp_c = 82.0

    def _detect_hardware(self, override_device: Optional[str] = None) -> Dict[str, Any]:
        """Auto-detects edge processor architecture and accelerators."""
        if override_device:
            dev_type = override_device
        else:
            arch = platform.machine().lower()
            system_desc = platform.platform().lower()
            
            # Check for NVIDIA Jetson Tegra platform
            is_jetson = os.path.exists("/etc/nv_tegra_release") or os.path.exists("/sys/devices/soc0/family")
            
            if is_jetson:
                # Distinguish Orin series based on RAM or device-tree
                total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
                if total_ram_gb >= 30.0:
                    dev_type = "JETSON_ORIN_AGX"
                elif total_ram_gb >= 15.0:
                    dev_type = "JETSON_ORIN_NX"
                else:
                    dev_type = "JETSON_ORIN_NANO"
            elif "arm" in arch or "aarch64" in arch:
                if os.path.exists("/proc/device-tree/model"):
                    try:
                        with open("/proc/device-tree/model", "r") as f:
                            model_str = f.read().lower()
                        if "raspberry pi 5" in model_str:
                            dev_type = "RASPBERRY_PI_5"
                        else:
                            dev_type = "ARM_GENERIC"
                    except Exception:
                        dev_type = "ARM_GENERIC"
                else:
                    dev_type = "ARM_GENERIC"
            else:
                dev_type = "X86_64_DESKTOP_DEV"

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        total_ram = round(psutil.virtual_memory().total / (1024 ** 3), 2)

        # Precision & Backend selection
        if dev_type.startswith("JETSON"):
            recommended_backend = "TENSORRT_INT8"
            optimal_precision = "int8"
        elif cuda_available:
            recommended_backend = "CUDA_FP16"
            optimal_precision = "fp16"
        elif dev_type == "RASPBERRY_PI_5":
            recommended_backend = "ONNX_INT8_NCNN"
            optimal_precision = "int8"
        else:
            recommended_backend = "CPU_INT8"
            optimal_precision = "int8"

        return {
            "device_class": dev_type,
            "recommended_backend": recommended_backend,
            "optimal_precision": optimal_precision,
            "total_ram_gb": total_ram,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "cpu_cores": psutil.cpu_count(logical=False) or 4
        }

    def evaluate_operating_state(
        self,
        battery_pct: float = 85.0,
        battery_discharge_amps: float = 4.2,
        temperature_c: Optional[float] = None,
        mem_used_pct: Optional[float] = None
    ) -> Tuple[int, str, Dict[str, Any]]:
        """
        Evaluates system vitals and selects appropriate Degradation Level and Power State.
        Returns: (degradation_level, power_state, execution_policy)
        """
        # Read temperature if not injected
        if temperature_c is None:
            temperature_c = 48.0
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for k, entries in temps.items():
                        if entries:
                            temperature_c = max(temperature_c, entries[0].current)
            except Exception:
                pass

        # Check Memory Pressure if not overridden
        if mem_used_pct is None:
            mem = psutil.virtual_memory()
            mem_used_pct = mem.percent

        # 1. Determine Power State
        if battery_pct < 15.0 or temperature_c >= self.thermal_critical_temp_c:
            power_state = PowerState.CRITICAL
        elif battery_pct < 30.0 or temperature_c >= self.thermal_warning_temp_c:
            power_state = PowerState.POWER_SAVE
        elif battery_pct < 60.0:
            power_state = PowerState.BALANCED
        else:
            power_state = PowerState.PERFORMANCE

        # 2. Determine Graceful Degradation Level
        if power_state == PowerState.CRITICAL:
            level = DegradationLevel.LEVEL_4  # Logging only, zero heavy AI
            self.emergency_memory_cleanup()
        elif power_state == PowerState.POWER_SAVE or mem_used_pct > 88.0:
            level = DegradationLevel.LEVEL_3  # Detection only (nano YOLO)
            self.emergency_memory_cleanup()
        elif power_state == PowerState.BALANCED or mem_used_pct > 75.0:
            level = DegradationLevel.LEVEL_2  # Lightweight YOLO + sparse U-Net
        elif self.device_profile["device_class"] in ["JETSON_ORIN_AGX", "X86_64_DESKTOP_DEV"]:
            level = DegradationLevel.LEVEL_0  # Maximum perception quality
        else:
            level = DegradationLevel.LEVEL_1  # Standard Orin NX / Nano

        self.current_degradation_level = level
        self.current_power_state = power_state

        # 3. Derive Execution Policy
        policy = {
            "degradation_level": level,
            "power_state": power_state,
            "temperature_c": round(temperature_c, 1),
            "battery_pct": round(battery_pct, 1),
            "memory_used_pct": round(mem_used_pct, 1),
            "input_resolution": (640, 640) if level <= 1 else ((480, 480) if level == 2 else (320, 320)),
            "enable_unet": level <= 2,
            "enable_tiling": level == 0,
            "enable_unknown_detector": level <= 2,
            "target_inference_fps": 30 if level == 0 else (15 if level <= 2 else 5)
        }
        return level, power_state, policy

    @staticmethod
    def emergency_memory_cleanup():
        """Frees unreferenced memory pools and clears CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
