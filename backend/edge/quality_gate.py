"""
Sonar Data Quality Gate for Autonomous Edge Perception.
Inspects raw acoustic frames in <=2ms prior to invoking expensive AI inference.
Rejects corrupted frames, dead pings, severe speckle noise, acoustic saturation,
and motion distortions to eliminate catastrophic false positives.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np


@dataclass
class QualityReport:
    passed: bool
    status: str  # "PASS" or "LOW_QUALITY_FRAME"
    reasons: List[str]
    metrics: Dict[str, float]
    evaluation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SonarQualityGate:
    """
    Sub-millisecond acoustic quality validation gate.
    Ensures corrupted or uninterpretable sonar frames are flagged immediately,
    saving compute power and preventing hallucinated detections on dead pings.
    """
    def __init__(
        self,
        min_contrast_range: float = 18.0,
        max_saturation_pct: float = 22.0,
        min_active_signal_pct: float = 8.0,
        max_speckle_index: float = 1.35,
        min_mean_intensity: float = 10.0,
        max_nan_inf_tolerance: int = 0
    ):
        self.min_contrast_range = min_contrast_range
        self.max_saturation_pct = max_saturation_pct
        self.min_active_signal_pct = min_active_signal_pct
        self.max_speckle_index = max_speckle_index
        self.min_mean_intensity = min_mean_intensity
        self.max_nan_inf_tolerance = max_nan_inf_tolerance

    def evaluate(self, image: np.ndarray) -> QualityReport:
        """
        Evaluates the input sonar array against physical and acoustic criteria.
        Returns a QualityReport indicating PASS or LOW_QUALITY_FRAME.
        """
        t0 = time.perf_counter()
        reasons: List[str] = []
        metrics: Dict[str, float] = {}

        # 1. Integrity check: Valid array structure
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return QualityReport(
                passed=False,
                status="LOW_QUALITY_FRAME",
                reasons=["CORRUPTED_FRAME: Null or empty numpy array"],
                metrics={"size": 0.0},
                evaluation_time_ms=round((time.perf_counter() - t0) * 1000, 3)
            )

        # Convert to 2D grayscale if multichannel
        if image.ndim == 3:
            gray = image[:, :, 0]
        else:
            gray = image

        # 2. Check for NaN or Inf corruption
        nan_inf_count = int(np.isnan(gray).sum() + np.isinf(gray).sum())
        if nan_inf_count > self.max_nan_inf_tolerance:
            reasons.append(f"NUMERICAL_CORRUPTION: Found {nan_inf_count} NaN/Inf pixels")

        # Fast subsampling for ultra-low latency (<1.5ms) on large frames
        h, w = gray.shape
        if h > 512 or w > 512:
            sampled = gray[::2, ::2].astype(np.float32)
        else:
            sampled = gray.astype(np.float32)

        total_pixels = sampled.size

        # 3. Intensity statistics
        mean_val = float(np.mean(sampled))
        std_val = float(np.std(sampled))
        p05, p95 = np.percentile(sampled, [5, 95])
        dynamic_range = float(p95 - p05)

        # 4. Saturation & Signal Starvation
        saturated_pixels = int((sampled >= 250).sum())
        sat_pct = (saturated_pixels / total_pixels) * 100.0

        starved_pixels = int((sampled <= 5).sum())
        starved_pct = (starved_pixels / total_pixels) * 100.0

        # Active acoustic signal ratio
        active_signal_pct = 100.0 - starved_pct

        # 5. Speckle Index (Cu = std / mean) & Equivalent Number of Looks (ENL = (mean / std)^2)
        if mean_val > 1.0:
            speckle_index = float(std_val / mean_val)
            enl = float((mean_val / max(std_val, 1e-4)) ** 2)
        else:
            speckle_index = 999.0
            enl = 0.0

        # 6. Motion / ping drop artifact check (Cross-row variance discontinuity)
        # Compute row-wise means along along-track direction
        row_means = np.mean(sampled, axis=1)
        row_diffs = np.abs(np.diff(row_means))
        max_row_jump = float(np.max(row_diffs)) if len(row_diffs) > 0 else 0.0

        metrics["mean_intensity"] = round(mean_val, 2)
        metrics["std_intensity"] = round(std_val, 2)
        metrics["dynamic_range"] = round(dynamic_range, 2)
        metrics["saturation_pct"] = round(sat_pct, 2)
        metrics["active_signal_pct"] = round(active_signal_pct, 2)
        metrics["speckle_index"] = round(speckle_index, 2)
        metrics["enl"] = round(enl, 2)
        metrics["max_row_jump"] = round(max_row_jump, 2)

        # Apply Hard Thresholds
        if mean_val < self.min_mean_intensity:
            reasons.append(f"INSUFFICIENT_SIGNAL: Mean intensity {mean_val:.1f} < {self.min_mean_intensity}")

        if active_signal_pct < self.min_active_signal_pct:
            reasons.append(f"DEAD_PING_STARVATION: Active signal {active_signal_pct:.1f}% < {self.min_active_signal_pct}%")

        if sat_pct > self.max_saturation_pct:
            reasons.append(f"EXCESSIVE_SATURATION: Acoustic clipping {sat_pct:.1f}% > {self.max_saturation_pct}%")

        if dynamic_range < self.min_contrast_range:
            reasons.append(f"POOR_CONTRAST: Dynamic range {dynamic_range:.1f} < {self.min_contrast_range}")

        if speckle_index > self.max_speckle_index and mean_val > 20:
            reasons.append(f"EXCESSIVE_SPECKLE_NOISE: Speckle index {speckle_index:.2f} > {self.max_speckle_index}")

        if max_row_jump > 70.0:
            reasons.append(f"MOTION_OR_TRANSMIT_ARTIFACT: Severe inter-ping jump {max_row_jump:.1f}")

        eval_ms = round((time.perf_counter() - t0) * 1000, 3)
        passed = len(reasons) == 0

        return QualityReport(
            passed=passed,
            status="PASS" if passed else "LOW_QUALITY_FRAME",
            reasons=reasons,
            metrics=metrics,
            evaluation_time_ms=eval_ms
        )
