"""
AI Orchestrator / Pipeline Agent
Coordinates the complete SIH57 pipeline across specialized CV, ML, and Geospatial tools.
The agent acts as an orchestrator and does NOT replace underlying computer vision models.
"""
from typing import Dict, Any, List, Optional
import os
import uuid

from ai.preprocessing.pipeline import SonarPreprocessor
from ai.detection.yolo_detector import YOLODetector
from ai.segmentation.unet_segmenter import UNetSegmenter
from ai.anomaly_detection.autoencoder import AnomalyDetector
from ai.measurement.estimator import DimensionEstimator
from ai.geospatial.geotagger import GeospatialEngine

class SIHPipelineAgent:
    """
    AI Agent that manages execution flow, logs intermediate steps,
    and returns structured, explainable analysis results.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.preprocessor = SonarPreprocessor()
        self.detector = YOLODetector()
        self.segmenter = UNetSegmenter()
        self.anomaly_detector = AnomalyDetector()
        self.measurer = DimensionEstimator()
        self.geotagger = GeospatialEngine()

    def analyze_image(self, image_path: str, raster_meta_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes end-to-end coordinated pipeline with full audit logging.
        """
        analysis_id = str(uuid.uuid4())[:8]
        execution_trace = []

        # 1. Validation
        val_res = self.preprocessor.validate_image(image_path)
        execution_trace.append({"stage": "input_validation", "status": val_res.get("status", "failed")})
        if not val_res.get("valid"):
            return {
                "analysis_id": analysis_id,
                "status": "rejected",
                "error": val_res.get("error"),
                "execution_trace": execution_trace
            }

        # 2. Preprocessing
        prep_res = self.preprocessor.preprocess(image_path)
        execution_trace.append({"stage": "preprocessing", "status": "completed"})

        # 3. Candidate Detection (YOLO)
        det_res = self.detector.detect(prep_res)
        execution_trace.append({
            "stage": "detection",
            "model_status": det_res.get("status"),
            "candidates_found": len(det_res.get("detections", []))
        })

        # 4. Raster Metadata / Georeferencing
        raster_meta = raster_meta_override or self.geotagger.read_raster_metadata(image_path)
        georef_case = self.geotagger.classify_georef_case(raster_meta)
        execution_trace.append({"stage": "georeference_check", "case": georef_case})

        # 5. Process detections
        final_objects = []
        for det in det_res.get("detections", []):
            # Anomaly Filtering & Calibration
            anomaly_res = self.anomaly_detector.evaluate_detection(det)
            
            # Dimension Estimation
            dims = self.measurer.estimate_dimensions(det.get("bbox", {}), raster_meta.get("res"))
            
            # Geospatial Coordinates
            lat, lon = None, None
            if georef_case == "A":
                center = self.geotagger.get_object_center(det.get("bbox", {}))
                x_map, y_map = self.geotagger.locate_case_a(center, raster_meta)
                lat, lon = self.geotagger.to_lat_lon(x_map, y_map, raster_meta.get("crs"))
            
            # Risk Scoring
            risk_score = self._calculate_risk(det.get("class"), dims.get("area_sq_m"), det.get("confidence", 0.5))

            rec = self.geotagger.create_object_record(
                detection=det,
                lat=lat,
                lon=lon,
                length_m=dims.get("length_m"),
                width_m=dims.get("width_m"),
                case=georef_case,
                uncertainty_m=1.5 if georef_case == "A" else 8.0
            )
            rec["risk_score"] = risk_score
            rec["anomaly_status"] = anomaly_res.get("status")
            final_objects.append(rec)

        execution_trace.append({"stage": "final_synthesis", "objects_analyzed": len(final_objects)})

        return {
            "analysis_id": analysis_id,
            "status": "success",
            "image_path": image_path,
            "georeferencing_case": georef_case,
            "total_detections": len(final_objects),
            "detections": final_objects,
            "execution_trace": execution_trace,
            "yolo_model_loaded": self.detector.is_model_loaded,
            "unet_model_loaded": self.segmenter.is_model_loaded
        }

    def _calculate_risk(self, debris_class: Optional[str], area_sq_m: Optional[float], confidence: float) -> str:
        """
        Computes multi-factor risk assessment based on debris type and scale.
        """
        high_risk_classes = ["fishing_net", "shipwreck_fragment"]
        if debris_class in high_risk_classes:
            return "HIGH"
        if area_sq_m and area_sq_m > 20.0:
            return "HIGH"
        if confidence > 0.70:
            return "MEDIUM"
        return "LOW"
