"""
AI Orchestrator / Pipeline Agent
Coordinates the complete SIH57 pipeline across specialized CV, ML, and Geospatial tools:
  - Input Validation & Sonar Preprocessing (Lee Filter, CLAHE)
  - YOLOv11 Candidate Object Detection
  - DBSCAN Geological Rock Cluster Suppression
  - U-Net / Attention U-Net Pixel-Level Segmentation & Contour Extraction
  - CNN Autoencoder Anomaly Detection & Acoustic Shadow Verification
  - Multi-Factor Confidence Calibration (Confirmed Debris / Suspicious / Rejected)
  - Module 5 Geotagging Engine (Case A Affine / Case B Sonar Geometry)
  - Physical Metric Dimension Estimation (meters & square meters)
  - Multi-Factor Hazard Risk Assessment (HIGH / MEDIUM / LOW)
  - Natural Language Hydrographic Explainability & Audit Logging (SQLite)
Strictly adheres to modular tool boundaries; orchestrates specialized engines without replacing them.
"""

from typing import Dict, Any, List, Optional
import os
import uuid
import time
import cv2
import numpy as np

from ai.preprocessing.pipeline import SonarPreprocessor
from ai.detection.yolo_detector import YOLODetector
from ai.segmentation.unet_segmenter import UNetSegmenter
from ai.anomaly_detection.autoencoder import AnomalyDetector
from ai.anomaly_detection.rock_cluster_filter import DBSCANRockFilter
from ai.measurement.estimator import DimensionEstimator
from ai.geospatial.geotagger import GeospatialEngine
from agent.explainability import ExplainabilitySynthesizer
from agent.audit_logger import SurveyAuditLogger


class SIHPipelineAgent:
    """
    Production AI Agent coordinating end-to-end underwater debris detection,
    anomaly filtering, geotagging, and explainability synthesis.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Specialized Core Tools
        self.preprocessor = SonarPreprocessor()
        self.detector = YOLODetector(
            model_path=self.config.get("yolo_checkpoint")
        )
        self.segmenter = UNetSegmenter(
            checkpoint_path=self.config.get("unet_checkpoint")
        )
        self.anomaly_detector = AnomalyDetector(
            checkpoint_path=self.config.get("autoencoder_checkpoint",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "checkpoints", "autoencoder", "baseline_autoencoder.pt"))
        )
        self.rock_filter = DBSCANRockFilter(eps=70.0, min_samples=4)
        self.measurer = DimensionEstimator()
        self.geotagger = GeospatialEngine()
        self.explainer = ExplainabilitySynthesizer()
        self.audit_logger = SurveyAuditLogger()

    def analyze_image(
        self,
        image_path: str,
        raster_meta_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end coordinated pipeline with full audit logging and explainability.
        """
        start_time = time.perf_counter()
        analysis_id = f"SURVEY_{str(uuid.uuid4())[:8].upper()}"
        execution_trace = []

        # -------------------------------------------------------------
        # Stage 1: Input Validation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        val_res = self.preprocessor.validate_image(image_path)
        t_val = round((time.perf_counter() - t0) * 1000, 2)

        if not val_res.get("valid"):
            execution_trace.append({
                "stage": "input_validation",
                "status": "failed",
                "duration_ms": t_val,
                "error": val_res.get("error")
            })
            return {
                "analysis_id": analysis_id,
                "status": "rejected",
                "error": val_res.get("error"),
                "execution_trace": execution_trace
            }

        execution_trace.append({
            "stage": "input_validation",
            "status": "completed",
            "duration_ms": t_val,
            "dimensions": val_res.get("dimensions")
        })

        # Load raw image
        raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # -------------------------------------------------------------
        # Stage 2: Sonar Preprocessing (Lee filter, CLAHE)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        prep_res = self.preprocessor.preprocess(image_path)
        t_prep = round((time.perf_counter() - t0) * 1000, 2)
        execution_trace.append({
            "stage": "preprocessing",
            "status": "completed",
            "duration_ms": t_prep,
            "filters_applied": ["min_max_normalization", "lee_speckle_filter", "clahe_contrast_boost"]
        })

        # -------------------------------------------------------------
        # Stage 3: Candidate Object Detection (YOLO)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        det_res = self.detector.detect(prep_res.get("preprocessed_image", image_path))
        t_det = round((time.perf_counter() - t0) * 1000, 2)
        raw_detections = det_res.get("detections", [])

        execution_trace.append({
            "stage": "candidate_detection",
            "status": "completed",
            "duration_ms": t_det,
            "model_status": det_res.get("status"),
            "candidates_found": len(raw_detections)
        })

        # -------------------------------------------------------------
        # Stage 4: Geological Rock Cluster Filtering (DBSCAN)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        clustered_detections = self.rock_filter.filter_detections(raw_detections)
        t_rock = round((time.perf_counter() - t0) * 1000, 2)
        rock_clusters_count = sum(1 for d in clustered_detections if d.get("is_rock_cluster"))

        execution_trace.append({
            "stage": "rock_cluster_filtering",
            "status": "completed",
            "duration_ms": t_rock,
            "rock_clusters_suppressed": rock_clusters_count
        })

        # -------------------------------------------------------------
        # Stage 5: Raster Metadata & Georeferencing Check (Module 5)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        raster_meta = raster_meta_override or self.geotagger.read_raster_metadata(image_path)
        georef_case = self.geotagger.classify_georef_case(raster_meta)
        t_geo = round((time.perf_counter() - t0) * 1000, 2)

        execution_trace.append({
            "stage": "georeference_check",
            "status": "completed",
            "duration_ms": t_geo,
            "case": georef_case,
            "crs": raster_meta.get("crs")
        })

        # -------------------------------------------------------------
        # Stage 6: Synthesis, Segmentation, Anomaly Calibration & Risk
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        final_objects = []

        for det in clustered_detections:
            bbox = det.get("bbox", {})
            
            # Crop ROI patch for high-res segmentation & anomaly analysis
            h_raw, w_raw = raw_img.shape[:2]
            x1 = max(0, min(w_raw - 1, int(bbox.get("x1", 0))))
            y1 = max(0, min(h_raw - 1, int(bbox.get("y1", 0))))
            x2 = max(x1 + 1, min(w_raw, int(bbox.get("x2", w_raw))))
            y2 = max(y1 + 1, min(h_raw, int(bbox.get("y2", h_raw))))
            patch_crop = raw_img[y1:y2, x1:x2]

            # 6a: U-Net Region Segmentation
            seg_res = self.segmenter.segment_roi(patch_crop)
            has_mask = seg_res.get("mask_available", False)
            pixel_area = seg_res.get("total_area_px") if has_mask else None

            # 6b: Autoencoder Anomaly & Shadow Verification
            anomaly_res = self.anomaly_detector.evaluate_detection(det, image_context=raw_img)

            # 6c: Physical Metric Dimension Estimation
            dims = self.measurer.estimate_dimensions(bbox, raster_meta.get("res"))

            # 6d: Real-World Coordinates (Case A Affine or Case B Dead Reckoning)
            lat, lon = None, None
            if georef_case == "A":
                center = self.geotagger.get_object_center(bbox)
                x_map, y_map = self.geotagger.locate_case_a(center, raster_meta)
                lat, lon = self.geotagger.to_lat_lon(x_map, y_map, raster_meta.get("crs"))

            # 6e: Multi-Factor Risk Assessment
            risk_score = self._calculate_risk(
                debris_class=det.get("class"),
                area_sq_m=dims.get("area_sq_m"),
                confidence=anomaly_res.get("calibrated_confidence", det.get("confidence", 0.5))
            )

            # 6f: Explainability Synthesis
            explanation = self.explainer.explain_target(
                detection=det,
                calibrated_status=anomaly_res.get("status", "unknown"),
                calibrated_conf=anomaly_res.get("calibrated_confidence", det.get("confidence", 0.5)),
                reconstruction_error=anomaly_res.get("reconstruction_error", 0.0),
                shadow_verified=anomaly_res.get("shadow_verified", True),
                is_rock_cluster=det.get("is_rock_cluster", False),
                risk_level=risk_score,
                dimensions=dims,
                coordinates={"lat": lat, "lon": lon}
            )

            # Assemble clean object record
            rec = self.geotagger.create_object_record(
                detection=det,
                lat=lat,
                lon=lon,
                length_m=dims.get("length_m"),
                width_m=dims.get("width_m"),
                case=georef_case,
                uncertainty_m=1.5 if georef_case == "A" else 8.0
            )
            rec["calibrated_confidence"] = anomaly_res.get("calibrated_confidence", det.get("confidence", 0.5))
            rec["anomaly_status"] = anomaly_res.get("status")
            rec["is_anomaly"] = anomaly_res.get("is_anomaly")
            rec["reconstruction_error"] = anomaly_res.get("reconstruction_error")
            rec["shadow_verified"] = anomaly_res.get("shadow_verified")
            rec["is_rock_cluster"] = det.get("is_rock_cluster", False)
            rec["risk_score"] = risk_score
            rec["segmentation_mask_available"] = has_mask
            rec["pixel_area"] = pixel_area
            rec["explanation"] = explanation
            final_objects.append(rec)

        t_synth = round((time.perf_counter() - t0) * 1000, 2)
        execution_trace.append({
            "stage": "synthesis_and_calibration",
            "status": "completed",
            "duration_ms": t_synth,
            "objects_analyzed": len(final_objects)
        })

        total_duration = round((time.perf_counter() - start_time) * 1000, 2)

        # -------------------------------------------------------------
        # Stage 7: SQLite Audit Logging
        # -------------------------------------------------------------
        self.audit_logger.log_survey(
            session_id=analysis_id,
            image_path=image_path,
            georef_case=georef_case,
            execution_trace=execution_trace,
            detections=final_objects,
            yolo_loaded=self.detector.is_model_loaded,
            unet_loaded=self.segmenter.is_model_loaded,
            autoencoder_loaded=self.anomaly_detector.is_model_loaded,
            total_duration_ms=total_duration
        )

        # Summary statistics
        stats = {
            "total_candidates": len(final_objects),
            "confirmed_debris": sum(1 for d in final_objects if d.get("anomaly_status") == "confirmed_debris"),
            "suspicious_anomaly": sum(1 for d in final_objects if d.get("anomaly_status") == "suspicious_anomaly"),
            "noise_rejected": sum(1 for d in final_objects if d.get("anomaly_status") == "noise_rejected"),
            "high_risk_count": sum(1 for d in final_objects if d.get("risk_score") == "HIGH")
        }

        return {
            "analysis_id": analysis_id,
            "status": "success",
            "image_path": image_path,
            "georeferencing_case": georef_case,
            "total_detections": len(final_objects),
            "summary_statistics": stats,
            "detections": final_objects,
            "execution_trace": execution_trace,
            "total_duration_ms": total_duration,
            "yolo_model_loaded": self.detector.is_model_loaded,
            "unet_model_loaded": self.segmenter.is_model_loaded,
            "autoencoder_model_loaded": self.anomaly_detector.is_model_loaded,
            "audit_database": self.audit_logger.db_path
        }

    def _calculate_risk(self, debris_class: Optional[str], area_sq_m: Optional[float], confidence: float) -> str:
        """
        Computes multi-factor risk assessment based on debris type, scale, and calibrated confidence.
        """
        high_risk_classes = ["fishing_net", "shipwreck_fragment"]
        if debris_class in high_risk_classes and confidence >= 0.60:
            return "HIGH"
        if area_sq_m and area_sq_m > 25.0 and confidence >= 0.50:
            return "HIGH"
        if confidence >= 0.70:
            return "MEDIUM"
        return "LOW"
