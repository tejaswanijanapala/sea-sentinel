"""
Deterministic Real-Time Edge Perception Pipeline for Sea Sentinel.
Integrates Sonar Ingestion, Data Quality Gate, Adaptive Preprocessing,
Parallel YOLO + U-Net Inference, Unknown Object Branch, Temporal Tracking,
Navigation Sensor Fusion, Geolocation with Uncertainty, and Acoustic Telemetry.
Provides granular stage-by-stage latency telemetry and strict bounded execution.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import os
import concurrent.futures
import numpy as np
import cv2

from .frame_buffer import BoundedFrameBuffer, PreallocatedBufferPool
from .quality_gate import SonarQualityGate, QualityReport
from .adaptive_preprocessor import AdaptiveSonarPreprocessor
from .unknown_detector import UnknownObjectDetector
from .uncertainty_engine import UncertaintyCalibrationEngine
from .temporal_tracker import SonarKalmanTracker
from .sensor_fusion import SensorFusionEngine, VehicleNavState
from .uncertainty_geolocator import UncertaintyGeolocator
from .resource_manager import EdgeResourceManager, DegradationLevel
from .telemetry_modem import AcousticTelemetryEncoder
from .active_learning import ActiveLearningSelector

from ai.detection.yolo_detector import YOLODetector
from ai.segmentation.unet_segmenter import UNetSegmenter


class EdgePerceptionPipeline:
    """
    Deterministic, production-grade Edge AI perception loop for underwater AUV/ROV platforms.
    """
    def __init__(
        self,
        yolo_detector: Optional[YOLODetector] = None,
        unet_segmenter: Optional[UNetSegmenter] = None,
        high_recall_mode: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.high_recall_mode = high_recall_mode

        # Core Edge Modules
        self.buffer_pool = PreallocatedBufferPool(pool_size=4, shape=(640, 640))
        self.frame_buffer = BoundedFrameBuffer(max_capacity=4)
        self.quality_gate = SonarQualityGate()
        self.preprocessor = AdaptiveSonarPreprocessor(target_size=(640, 640))
        self.unknown_detector = UnknownObjectDetector()
        self.uncertainty_engine = UncertaintyCalibrationEngine(high_recall_mode=high_recall_mode)
        self.tracker = SonarKalmanTracker()
        self.sensor_fusion = SensorFusionEngine()
        self.geolocator = UncertaintyGeolocator()
        self.resource_manager = EdgeResourceManager()
        self.telemetry_encoder = AcousticTelemetryEncoder()
        self.active_learner = ActiveLearningSelector()

        # Neural Perception Models (Independent instances)
        self.yolo = yolo_detector or YOLODetector()
        self.unet = unet_segmenter or UNetSegmenter()

        # Thread pool for parallel YOLO + U-Net inference
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="edge_parallel_ai"
        )

    def set_high_recall_mode(self, enabled: bool):
        """Switches pipeline into High-Recall mode (minimizing false negatives)."""
        self.high_recall_mode = enabled
        self.uncertainty_engine.set_high_recall_mode(enabled)

    def _infer_yolo_branch(self, image: np.ndarray, conf_thresh: float) -> List[Dict[str, Any]]:
        """Independent YOLO candidate detection branch."""
        if not self.yolo.is_model_loaded:
            return []
        res = self.yolo.detect(image, conf_override=conf_thresh)
        return res.get("detections", [])

    def _infer_unet_branch(self, image: np.ndarray) -> Dict[str, Any]:
        """Independent U-Net pixel-level semantic segmentation branch."""
        if not self.unet.is_model_loaded:
            h, w = image.shape[:2]
            return {
                "mask": np.zeros((h, w), dtype=np.uint8),
                "objects": []
            }
        return self.unet.segment_full_image(image)

    def process_frame(
        self,
        raw_image: np.ndarray,
        frame_id: str = "FRAME_001",
        timestamp: Optional[float] = None,
        altitude_m: Optional[float] = 5.0,
        battery_pct: float = 85.0
    ) -> Dict[str, Any]:
        """
        Executes the full deterministic edge perception pipeline with separate latency profiling.
        """
        t_pipeline_start = time.perf_counter()
        timing: Dict[str, float] = {}
        ping_time = timestamp or time.time()

        # 1. Evaluate Edge Resource Manager & Operating State
        t0 = time.perf_counter()
        deg_level, power_state, policy = self.resource_manager.evaluate_operating_state(
            battery_pct=battery_pct
        )
        timing["resource_eval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Ingestion & Preallocated Memory Buffer Acquisition
        t0 = time.perf_counter()
        buf_id, img_buf = self.buffer_pool.copy_into_pool(raw_image)
        timing["ingestion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Sonar Data Quality Gate
        t0 = time.perf_counter()
        quality_report = self.quality_gate.evaluate(raw_image)
        timing["quality_gate_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # If data quality is severely degraded, reject frame early
        if not quality_report.passed:
            self.buffer_pool.release(buf_id)
            total_ms = round((time.perf_counter() - t_pipeline_start) * 1000, 2)
            timing["total_pipeline_ms"] = total_ms
            return {
                "frame_id": frame_id,
                "status": "LOW_QUALITY_FRAME",
                "quality_report": quality_report.to_dict(),
                "detections": [],
                "telemetry_packets": [],
                "degradation_level": deg_level,
                "power_state": power_state,
                "timing_breakdown": timing
            }

        # 4. Adaptive Sonar Preprocessing
        t0 = time.perf_counter()
        alt_px = int(altitude_m * 10.0) if altitude_m else None
        enhanced_image, prep_meta = self.preprocessor.process(
            img_buf,
            quality_metrics=quality_report.metrics,
            altitude_px=alt_px
        )
        timing["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # In Emergency Degradation Level 4: skip heavy inference entirely
        if deg_level == DegradationLevel.LEVEL_4:
            self.buffer_pool.release(buf_id)
            total_ms = round((time.perf_counter() - t_pipeline_start) * 1000, 2)
            timing["total_pipeline_ms"] = total_ms
            return {
                "frame_id": frame_id,
                "status": "HEALTH_LOGGING_ONLY_LEVEL_4",
                "quality_report": quality_report.to_dict(),
                "detections": [],
                "telemetry_packets": [],
                "degradation_level": deg_level,
                "power_state": power_state,
                "timing_breakdown": timing
            }

        # 5. Parallel YOLO + U-Net Inference
        # In High Recall Mode: lower initial detection threshold to capture faint echoes
        yolo_conf_thresh = 0.18 if self.high_recall_mode else 0.25

        t_infer_start = time.perf_counter()
        future_yolo = self._thread_pool.submit(self._infer_yolo_branch, enhanced_image, yolo_conf_thresh)

        if policy["enable_unet"]:
            future_unet = self._thread_pool.submit(self._infer_unet_branch, enhanced_image)
        else:
            future_unet = None

        yolo_dets = future_yolo.result()
        timing["yolo_ms"] = round((time.perf_counter() - t_infer_start) * 1000, 2)

        if future_unet:
            unet_res = future_unet.result()
            timing["unet_ms"] = round((time.perf_counter() - t_infer_start) * 1000, 2)
        else:
            unet_res = {"mask": np.zeros(enhanced_image.shape[:2], dtype=np.uint8), "objects": []}
            timing["unet_ms"] = 0.0

        timing["parallel_model_inference_ms"] = round((time.perf_counter() - t_infer_start) * 1000, 2)

        # 6. Candidate Merging & Multi-Model Fusion
        t0 = time.perf_counter()
        fused_candidates: List[Dict[str, Any]] = []
        h_f, w_f = enhanced_image.shape[:2]

        # Extract YOLO candidate objects
        for det in yolo_dets:
            bbox = det.get("bbox", {})
            fused_candidates.append({
                "class": det.get("class", "marine_debris"),
                "yolo_conf": float(det.get("confidence", 0.5)),
                "unet_score": 0.40,  # default until verified with mask
                "bbox": bbox,
                "centroid": det.get("centroid", [(bbox.get("x1", 0) + bbox.get("x2", 0))/2,
                                                  (bbox.get("y1", 0) + bbox.get("y2", 0))/2]),
                "detection_source": "yolo"
            })

        # Add independent U-Net candidates that YOLO missed (Crucial for high recall)
        unet_objs = unet_res.get("objects", [])
        for u_obj in unet_objs:
            u_cent = u_obj.get("centroid", [0, 0])
            # Check overlap with existing YOLO candidates
            matched = False
            for cand in fused_candidates:
                dist = np.hypot(cand["centroid"][0] - u_cent[0], cand["centroid"][1] - u_cent[1])
                if dist < 45.0:
                    cand["unet_score"] = float(u_obj.get("mean_probability", 0.7))
                    cand["detection_source"] = "both"
                    matched = True
                    break
            if not matched and policy["enable_unet"]:
                # Novel candidate proposed solely by U-Net
                u_box = u_obj.get("bbox", {})
                fused_candidates.append({
                    "class": "marine_debris_candidate",
                    "yolo_conf": 0.20,
                    "unet_score": float(u_obj.get("mean_probability", 0.65)),
                    "bbox": u_box,
                    "centroid": u_cent,
                    "detection_source": "unet"
                })
        timing["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 7. Third Perception Branch: Unknown Object & Anomaly Analysis
        t0 = time.perf_counter()
        analyzed_candidates: List[Dict[str, Any]] = []
        for cand in fused_candidates:
            bbox = cand.get("bbox", {})
            bx1 = max(0, min(w_f - 1, int(bbox.get("x1", 0))))
            by1 = max(0, min(h_f - 1, int(bbox.get("y1", 0))))
            bx2 = max(bx1 + 4, min(w_f, int(bbox.get("x2", w_f))))
            by2 = max(by1 + 4, min(h_f, int(bbox.get("y2", h_f))))
            patch = enhanced_image[by1:by2, bx1:bx2]

            anomaly_eval = self.unknown_detector.evaluate_candidate(
                patch=patch,
                yolo_conf=cand["yolo_conf"],
                unet_score=cand["unet_score"],
                class_name=cand["class"]
            )
            cand["anomaly_eval"] = anomaly_eval
            cand["is_unknown"] = anomaly_eval["is_unknown"]
            cand["anomaly_score"] = anomaly_eval["anomaly_score"]

            if anomaly_eval["category"] != "FALSE_POSITIVE":
                analyzed_candidates.append(cand)
        timing["unknown_detector_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 8. Temporal Tracking (Kalman Filter across pings)
        t0 = time.perf_counter()
        tracked_candidates = self.tracker.update(analyzed_candidates, dt=1.0)
        timing["tracking_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 9. Confidence + Uncertainty Calibration
        t0 = time.perf_counter()
        sonar_q_score = 1.0 if quality_report.passed else 0.4
        verified_detections: List[Dict[str, Any]] = []

        for cand in tracked_candidates:
            calib = self.uncertainty_engine.compute_calibrated_confidence(
                yolo_conf=cand["yolo_conf"],
                unet_score=cand["unet_score"],
                temporal_hits=cand.get("temporal_hits", 1),
                sonar_quality_score=sonar_q_score,
                relief_score=cand["anomaly_eval"].get("relief_score", 0.5),
                anomaly_score=cand["anomaly_score"],
                is_unknown=cand["is_unknown"]
            )
            cand["calibrated_confidence"] = calib["calibrated_confidence"]
            cand["epistemic_uncertainty"] = calib["epistemic_uncertainty"]
            cand["confidence_tier"] = calib["tier"]
            cand["action"] = calib["action"]

            # In high-recall mode, keep high-recall candidates as well
            if calib["tier"] in ["HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "HIGH_RECALL_CANDIDATE", "UNKNOWN"]:
                verified_detections.append(cand)
        timing["uncertainty_eval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 10. Sensor Fusion & Navigation State Interpolation
        t0 = time.perf_counter()
        nav_state = self.sensor_fusion.interpolate_nav_state(ping_time)
        timing["sensor_fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 11. Uncertainty-Aware Geolocation & Acoustic Telemetry Encoding
        t0 = time.perf_counter()
        final_detections: List[Dict[str, Any]] = []
        telemetry_packets: List[Dict[str, Any]] = []

        for idx, det in enumerate(verified_detections):
            obj_id = det.get("track_id", f"TGT_{idx+1:04d}")
            geo_info = self.geolocator.geolocate_target(
                object_id=obj_id,
                class_label=det["class"],
                confidence=det["calibrated_confidence"],
                pixel_centroid=tuple(det["centroid"]),
                image_shape=(h_f, w_f),
                nav_state=nav_state,
                sonar_frame_id=frame_id
            )
            merged = {**det, **geo_info}
            final_detections.append(merged)

            # 12. Encode confirmed / high-priority alerts for acoustic modem
            if det["confidence_tier"] in ["HIGH_CONFIDENCE", "UNKNOWN", "HIGH_RECALL_CANDIDATE"]:
                try:
                    raw_bytes = self.telemetry_encoder.encode(
                        target_id_num=int(obj_id.replace("TRK_", "").replace("TGT_", "") or "1"),
                        class_name=det["class"],
                        confidence=det["calibrated_confidence"],
                        slant_range_m=geo_info["slant_range_m"],
                        bearing_deg=geo_info["relative_bearing_deg"],
                        local_x_m=geo_info["local_x_east_m"],
                        local_y_m=geo_info["local_y_north_m"],
                        depth_m=geo_info["estimated_depth_m"],
                        timestamp_s=ping_time,
                        uncertainty_m=geo_info["position_uncertainty_m"]
                    )
                    telemetry_packets.append({
                        "object_id": obj_id,
                        "hex_packet": self.telemetry_encoder.to_hex(raw_bytes),
                        "packet_bytes": len(raw_bytes),
                        "crc8_valid": True
                    })
                except Exception:
                    pass

        timing["geolocation_and_telemetry_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Release acquired preallocated memory buffer back to pool
        self.buffer_pool.release(buf_id)

        # Total Pipeline Latency
        total_pipeline_ms = round((time.perf_counter() - t_pipeline_start) * 1000, 2)
        timing["total_pipeline_ms"] = total_pipeline_ms

        return {
            "frame_id": frame_id,
            "status": "PROCESSED",
            "quality_report": quality_report.to_dict(),
            "preprocessing_meta": prep_meta,
            "degradation_level": deg_level,
            "power_state": power_state,
            "high_recall_mode": self.high_recall_mode,
            "detections": final_detections,
            "total_detections": len(final_detections),
            "telemetry_packets": telemetry_packets,
            "nav_state": nav_state.to_dict(),
            "timing_breakdown": timing
        }
