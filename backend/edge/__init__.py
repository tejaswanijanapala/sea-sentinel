"""
Sea Sentinel: Edge AI & AUV/ROV Production Runtime
Provides hard real-time, offline-native, safety-isolated perception modules
for NVIDIA Jetson, Raspberry Pi 5, and ARM edge computers.
"""

from .frame_buffer import BoundedFrameBuffer, PreallocatedBufferPool
from .quality_gate import SonarQualityGate, QualityReport
from .adaptive_preprocessor import AdaptiveSonarPreprocessor
from .unknown_detector import UnknownObjectDetector
from .uncertainty_engine import UncertaintyCalibrationEngine
from .temporal_tracker import SonarKalmanTracker
from .sensor_fusion import SensorFusionEngine, VehicleNavState
from .uncertainty_geolocator import UncertaintyGeolocator
from .resource_manager import EdgeResourceManager, DegradationLevel, PowerState
from .watchdog import EdgeWatchdogSupervisor
from .telemetry_modem import AcousticTelemetryEncoder
from .active_learning import ActiveLearningSelector, ContinualLearningGuardian
from .edge_perception import EdgePerceptionPipeline

__all__ = [
    "BoundedFrameBuffer",
    "PreallocatedBufferPool",
    "SonarQualityGate",
    "QualityReport",
    "AdaptiveSonarPreprocessor",
    "UnknownObjectDetector",
    "UncertaintyCalibrationEngine",
    "SonarKalmanTracker",
    "SensorFusionEngine",
    "VehicleNavState",
    "UncertaintyGeolocator",
    "EdgeResourceManager",
    "DegradationLevel",
    "PowerState",
    "EdgeWatchdogSupervisor",
    "AcousticTelemetryEncoder",
    "ActiveLearningSelector",
    "ContinualLearningGuardian",
    "EdgePerceptionPipeline"
]
