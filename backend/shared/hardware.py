"""
Sea Sentinel: Hardware-Aware Acceleration & Pipeline Profiling Utility
Inspects available hardware (CUDA, TensorRT, OpenVINO, CPU threads, FP16),
manages model pre-warming, and provides high-resolution stage profiler.
"""

from typing import Dict, Any, List, Optional
import os
import time
import torch
import numpy as np
from shared.utils.logger import get_logger
logger = get_logger(__name__)




class HardwareDetector:
    """
    Detects compute hardware, active accelerators, precision support,
    and configures optimal execution parameters.
    """
    @staticmethod
    def get_hardware_profile() -> Dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        gpu_name = None
        gpu_memory_gb = 0.0
        fp16_supported = False
        device_type = "cpu"

        if cuda_available:
            try:
                device_type = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                # FP16 is supported on Pascal (compute capability 6.0+) and newer
                cap = torch.cuda.get_device_capability(0)
                fp16_supported = (cap[0] >= 6)
            except Exception:
                pass
        
        cpu_count = os.cpu_count() or 4
        # Configure torch thread count for optimal parallel CPU throughput
        if not cuda_available:
            try:
                optimal_threads = max(1, min(cpu_count, 8))
                torch.set_num_threads(optimal_threads)
            except Exception:
                pass

        return {
            "device": device_type,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "fp16_supported": fp16_supported,
            "cpu_cores": cpu_count,
            "torch_threads": torch.get_num_threads(),
            "recommended_batch_size": 16 if cuda_available else 4
        }


class PipelineProfiler:
    """
    High-resolution end-to-end stage profiler measuring execution times,
    enforcing <20s budget, and identifying latency bottlenecks.
    """
    def __init__(self, target_seconds: float = 20.0):
        self.target_seconds = target_seconds
        self.start_time = time.perf_counter()
        self.stages: Dict[str, float] = {}
        self._current_stage_name: Optional[str] = None
        self._current_stage_start: Optional[float] = None

    def start_stage(self, name: str):
        if self._current_stage_name and self._current_stage_start:
            self.end_stage(self._current_stage_name)
        self._current_stage_name = name
        self._current_stage_start = time.perf_counter()

    def end_stage(self, name: str) -> float:
        if self._current_stage_start is not None and self._current_stage_name == name:
            dur_ms = round((time.perf_counter() - self._current_stage_start) * 1000, 2)
            self.stages[name] = dur_ms
            self._current_stage_name = None
            self._current_stage_start = None
            return dur_ms
        return 0.0

    def record_stage(self, name: str, duration_ms: float):
        self.stages[name] = round(duration_ms, 2)

    def get_summary(self, mode: str = "balanced") -> Dict[str, Any]:
        total_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        total_sec = round(total_ms / 1000.0, 3)
        budget_pass = (total_sec <= self.target_seconds)
        headroom_sec = round(self.target_seconds - total_sec, 3)

        # Identify slowest component
        bottleneck_stage = "None"
        bottleneck_ms = 0.0
        if self.stages:
            sorted_stages = sorted(self.stages.items(), key=lambda x: x[1], reverse=True)
            bottleneck_stage, bottleneck_ms = sorted_stages[0]

        bottleneck_pct = round((bottleneck_ms / max(1.0, total_ms)) * 100, 1)

        return {
            "mode": mode,
            "target_seconds": self.target_seconds,
            "total_duration_ms": total_ms,
            "total_duration_seconds": total_sec,
            "budget_status": "PASS" if budget_pass else "EXCEEDED",
            "headroom_seconds": headroom_sec,
            "stages_ms": self.stages,
            "bottleneck": {
                "stage": bottleneck_stage,
                "duration_ms": bottleneck_ms,
                "duration_seconds": round(bottleneck_ms / 1000.0, 3),
                "percentage": bottleneck_pct
            },
            "hardware": HardwareDetector.get_hardware_profile()
        }


def warmup_ai_models(agent_instance: Any) -> Dict[str, Any]:
    """
    Executes a fast synthetic warmup forward pass through YOLO and Attention U-Net
    to eliminate cold-start compilation overhead for first user image.
    """
    t0 = time.perf_counter()
    warmup_status = {"yolo": False, "unet": False, "duration_ms": 0.0}

    dummy_img = np.zeros((640, 640), dtype=np.uint8)
    dummy_img[200:260, 200:260] = 180

    try:
        if hasattr(agent_instance, "detector") and agent_instance.detector.is_model_loaded:
            agent_instance.detector.detect(dummy_img)
            warmup_status["yolo"] = True
    except Exception as e:
        logger.info(f"[Warmup] YOLO warmup note: {e}")

    try:
        if hasattr(agent_instance, "segmenter") and agent_instance.segmenter.is_model_loaded:
            agent_instance.segmenter.segment_roi(dummy_img[200:300, 200:300])
            warmup_status["unet"] = True
    except Exception as e:
        logger.info(f"[Warmup] U-Net warmup note: {e}")

    warmup_status["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return warmup_status
