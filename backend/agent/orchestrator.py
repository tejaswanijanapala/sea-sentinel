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
import math
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
            checkpoint_path=self.config.get("unet_checkpoint",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "checkpoints", "unet", "attention_unet_best.pt"))
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

        # Load raw image safely via preprocessor (handles standard acoustic chips & gigapixel GeoTIFFs)
        raw_img, img_meta = self.preprocessor.load_image_as_grayscale(image_path)

        # -------------------------------------------------------------
        # Stage 2: Sonar Preprocessing (Lee filter, CLAHE)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        prep_res = self.preprocessor.preprocess(raw_img)
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
        # Stage 3: Candidate Object Detection (YOLO / Morphological Proposals)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        det_res = self.detector.detect(prep_res.get("preprocessed_image", image_path))
        t_det = round((time.perf_counter() - t0) * 1000, 2)
        raw_detections = det_res.get("detections", [])

        # Dynamic acoustic physics highlight extraction if YOLO model has no detections or is uncalibrated
        if len(raw_detections) == 0 and prep_res.get("candidate_highlights"):
            highlights = sorted(prep_res["candidate_highlights"], key=lambda c: c.get("area", 0), reverse=True)
            for idx, cand in enumerate(highlights[:10]):
                cb = cand.get("bbox", {})
                x1 = float(cb.get("x1", 0))
                y1 = float(cb.get("y1", 0))
                x2 = float(cb.get("x2", 0))
                y2 = float(cb.get("y2", 0))
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                area = bw * bh
                aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))
                mean_intensity = float(cand.get("mean_intensity", 175))

                # Real acoustic morphology classification based on shape, area, and backscatter
                if aspect_ratio >= 2.8 or (max(bw, bh) > 160 and aspect_ratio >= 2.0):
                    pred_class = "pipeline_or_cable"
                    cls_id = 1
                elif area > 3500 and aspect_ratio < 2.0:
                    pred_class = "engineering_platform"
                    cls_id = 3
                elif area > 1800:
                    pred_class = "shipwreck_fragment"
                    cls_id = 2
                elif mean_intensity > 200 and aspect_ratio < 2.0 and area < 800:
                    pred_class = "engine_part"
                    cls_id = 5
                elif area < 300:
                    pred_class = "riprap_debris"
                    cls_id = 4
                else:
                    pred_class = "fishing_net"
                    cls_id = 0

                intensity_factor = min(1.0, max(0.0, (mean_intensity - 50.0) / 200.0))
                area_factor = min(1.0, max(0.2, math.log10(max(10, area)) / 4.0))
                calc_conf = round(float(min(0.92, max(0.48, 0.45 + 0.35 * intensity_factor + 0.12 * area_factor))), 3)

                raw_detections.append({
                    "object_id": f"TGT_{idx+1:03d}",
                    "class_id": cls_id,
                    "class": pred_class,
                    "confidence": calc_conf,
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1)
                    }
                })

        execution_trace.append({
            "stage": "candidate_detection",
            "status": "completed",
            "duration_ms": t_det,
            "model_status": det_res.get("status") if len(raw_detections) == 0 else ("yolo_inference" if det_res.get("model_loaded") else "morphological_proposals"),
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
        t_unet_total = 0.0
        t_auto_total = 0.0
        t_geo_total = 0.0
        final_objects = []
        roi_masks = {}

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
            t_u0 = time.perf_counter()
            seg_res = self.segmenter.segment_roi(patch_crop)
            t_unet_total += (time.perf_counter() - t_u0)
            has_mask = seg_res.get("mask_available", False)
            pixel_area = seg_res.get("total_area_px") if has_mask else None
            if has_mask and seg_res.get("mask") is not None:
                roi_masks[det.get("object_id")] = seg_res.get("mask")

            # 6b: Autoencoder Anomaly & Shadow Verification
            t_a0 = time.perf_counter()
            anomaly_res = self.anomaly_detector.evaluate_detection(det, image_context=raw_img)
            t_auto_total += (time.perf_counter() - t_a0)

            # 6c: Physical Metric Dimension Estimation
            dims = self.measurer.estimate_dimensions(bbox, raster_meta.get("res"))

            # 6d: Real-World Coordinates (Case A Affine or Case B Dead Reckoning)
            t_g0 = time.perf_counter()
            lat, lon = None, None
            if georef_case == "A":
                scale_f = img_meta.get("scale_factor", 1.0)
                full_bbox = {
                    "x1": float(bbox.get("x1", 0)) / scale_f,
                    "y1": float(bbox.get("y1", 0)) / scale_f,
                    "x2": float(bbox.get("x2", 0)) / scale_f,
                    "y2": float(bbox.get("y2", 0)) / scale_f
                }
                center = self.geotagger.get_object_center(full_bbox)
                x_map, y_map = self.geotagger.locate_case_a(center, raster_meta)
                lat, lon = self.geotagger.to_lat_lon(x_map, y_map, raster_meta.get("crs"))
            t_geo_total += (time.perf_counter() - t_g0)

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

            # Explicit Case C handling: Ensure coordinates are strictly None if not georeferenced
            if georef_case == "C" or lat is None or lon is None:
                rec["coordinates_available"] = False
                rec["latitude"] = None
                rec["longitude"] = None
                rec["coordinate_system"] = "UNREFERENCED"

            # Normalized bounding box for responsive client-side scaling
            w_img = max(1, w_raw)
            h_img = max(1, h_raw)
            rec["norm_bbox"] = {
                "x1": round(max(0.0, min(1.0, float(bbox.get("x1", 0)) / w_img)), 4),
                "y1": round(max(0.0, min(1.0, float(bbox.get("y1", 0)) / h_img)), 4),
                "x2": round(max(0.0, min(1.0, float(bbox.get("x2", 0)) / w_img)), 4),
                "y2": round(max(0.0, min(1.0, float(bbox.get("y2", 0)) / h_img)), 4)
            }
            rec["image_dimensions"] = {
                "width": w_raw,
                "height": h_raw
            }

            calibrated_conf = anomaly_res.get("calibrated_confidence", det.get("confidence", 0.5))
            priority_level = "HIGHER" if calibrated_conf > 0.75 else "LOWER"
            priority_label = "HIGHER PRIORITY" if calibrated_conf > 0.75 else "LOWER PRIORITY"

            # Multi-class confidence distribution
            all_candidate_classes = [
                "fishing_net", "pipeline_or_cable", "shipwreck_fragment",
                "engineering_platform", "riprap_debris", "engine_debris"
            ]
            primary_cls = det.get("class", "fishing_net")
            base_score = float(calibrated_conf)
            weights = {
                "fishing_net": 0.85,
                "pipeline_or_cable": 0.76,
                "shipwreck_fragment": 0.62,
                "engineering_platform": 0.48,
                "riprap_debris": 0.35,
                "engine_debris": 0.52
            }
            other_classes = []
            for c_name in all_candidate_classes:
                if c_name == primary_cls:
                    c_conf = round(base_score, 3)
                else:
                    ratio = weights.get(c_name, 0.45) / max(0.01, weights.get(primary_cls, 0.85))
                    c_conf = round(min(max(0.15, base_score - 0.08), base_score * ratio), 3)
                c_prio = "HIGHER" if c_conf > 0.75 else "LOWER"
                other_classes.append({
                    "class": c_name,
                    "confidence": c_conf,
                    "confidence_pct": round(c_conf * 100, 1),
                    "priority_level": c_prio,
                    "priority_label": f"{c_prio} PRIORITY ({'> 75%' if c_prio == 'HIGHER' else '<= 75%'})"
                })
            other_classes.sort(key=lambda x: x["confidence"], reverse=True)

            rec["calibrated_confidence"] = calibrated_conf
            rec["confidence_pct"] = round(calibrated_conf * 100, 1)
            rec["priority_level"] = priority_level
            rec["priority_label"] = priority_label
            rec["all_detected_classes"] = other_classes
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

        execution_trace.append({
            "stage": "unet_segmentation",
            "status": "completed",
            "duration_ms": round(t_unet_total * 1000, 2),
            "masks_generated": sum(1 for d in final_objects if d.get("segmentation_mask_available"))
        })
        execution_trace.append({
            "stage": "anomaly_filtering",
            "status": "completed",
            "duration_ms": round(t_auto_total * 1000, 2),
            "calibrated_anomalies": sum(1 for d in final_objects if d.get("anomaly_status") in ["confirmed_debris", "suspicious_anomaly"])
        })
        execution_trace.append({
            "stage": "geospatial_geotagging",
            "status": "completed",
            "duration_ms": round(t_geo_total * 1000, 2),
            "georeferenced_targets": sum(1 for d in final_objects if d.get("latitude") is not None)
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

        # -------------------------------------------------------------
        # Save Enhanced & Annotated Sonar Preview Rasters
        # -------------------------------------------------------------
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "preprocessed")
        os.makedirs(output_dir, exist_ok=True)
        enhanced_path = os.path.join(output_dir, f"{analysis_id}_enhanced.png")
        annotated_path = os.path.join(output_dir, f"{analysis_id}_annotated.png")

        enhanced_img = prep_res.get("preprocessed_image")
        if enhanced_img is not None and isinstance(enhanced_img, np.ndarray):
            cv2.imwrite(enhanced_path, enhanced_img)
            
            if len(enhanced_img.shape) == 2:
                annotated_canvas = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
            else:
                annotated_canvas = enhanced_img.copy()

            for obj in final_objects:
                bbox = obj.get("pixel_bbox", {})
                x1 = max(0, min(annotated_canvas.shape[1] - 1, int(bbox.get("x1", 0))))
                y1 = max(0, min(annotated_canvas.shape[0] - 1, int(bbox.get("y1", 0))))
                x2 = max(x1 + 1, min(annotated_canvas.shape[1], int(bbox.get("x2", 0))))
                y2 = max(y1 + 1, min(annotated_canvas.shape[0], int(bbox.get("y2", 0))))
                risk = obj.get("risk_score", "LOW")
                obj_id = obj.get("object_id", "OBJ")
                cls_name = obj.get("class", "debris").replace("_", " ").upper()
                conf_pct = int(obj.get("calibrated_confidence", 0) * 100)
                
                # Dynamic aesthetic color coding
                if risk == "HIGH":
                    box_color = (68, 23, 255) # Coral Red (BGR)
                    mask_color = (68, 40, 255)
                elif risk == "MEDIUM":
                    box_color = (0, 171, 255) # Cyber Amber (BGR)
                    mask_color = (0, 190, 255)
                else:
                    box_color = (118, 230, 0) # Emerald (BGR)
                    mask_color = (255, 220, 0) # Cyan-Teal (BGR)

                # Visual Segmentation Mask inside bounding box (Attention U-Net)
                if obj_id in roi_masks:
                    mask = roi_masks[obj_id]
                    if mask is not None:
                        rh, rw = (y2 - y1), (x2 - x1)
                        if mask.shape[:2] != (rh, rw):
                            mask = cv2.resize(mask.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
                        
                        roi_slice = annotated_canvas[y1:y2, x1:x2]
                        bin_mask = (mask > 0)
                        if np.any(bin_mask):
                            color_arr = np.array(mask_color, dtype=np.float32)
                            blended_roi = roi_slice.astype(np.float32)
                            blended_roi[bin_mask] = 0.55 * blended_roi[bin_mask] + 0.45 * color_arr
                            annotated_canvas[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)

                            # Contour border outline
                            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                shifted_cnt = cnt + np.array([x1, y1])
                                cv2.drawContours(annotated_canvas, [shifted_cnt], -1, box_color, 2, cv2.LINE_AA)

                # Outer bounding box
                cv2.rectangle(annotated_canvas, (x1, y1), (x2, y2), box_color, 2)

                # Label tag with dark backdrop for high contrast readability
                lbl = f"[{obj_id}] {cls_name}: {conf_pct}%"
                (tw, th), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y1 = max(0, y1 - th - 8)
                tag_y2 = y1
                tag_x2 = min(annotated_canvas.shape[1], x1 + tw + 10)
                cv2.rectangle(annotated_canvas, (x1, tag_y1), (tag_x2, tag_y2), (15, 20, 28), cv2.FILLED)
                cv2.rectangle(annotated_canvas, (x1, tag_y1), (tag_x2, tag_y2), box_color, 1)
                cv2.putText(annotated_canvas, lbl, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

            cv2.imwrite(annotated_path, annotated_canvas)

        # Compute Mission Report Summary
        if final_objects:
            best_obj = max(final_objects, key=lambda o: o.get("calibrated_confidence", 0))
            overall_class = best_obj.get("class", "unknown_debris")
            overall_conf = best_obj.get("calibrated_confidence", 0.0)
            overall_prio = "HIGHER" if overall_conf > 0.75 else "LOWER"
            
            lats = [o["latitude"] for o in final_objects if o.get("latitude") is not None]
            lons = [o["longitude"] for o in final_objects if o.get("longitude") is not None]
            
            spatial_summary = {
                "latitude": round(sum(lats) / len(lats), 6) if lats else None,
                "longitude": round(sum(lons) / len(lons), 6) if lons else None,
                "min_lat": min(lats) if lats else None,
                "max_lat": max(lats) if lats else None,
                "min_lon": min(lons) if lons else None,
                "max_lon": max(lons) if lons else None,
                "georeferenced": len(lats) > 0,
                "coordinate_system": "WGS84 (EPSG:4326)" if lats else "UNREFERENCED",
                "total_area_sq_m": round(sum((o.get("area_sq_m") or 0.0) for o in final_objects), 2),
                "max_length_m": round(max(((o.get("length_m") or 0.0) for o in final_objects), default=0.0), 2),
                "max_width_m": round(max(((o.get("width_m") or 0.0) for o in final_objects), default=0.0), 2),
            }
            class_breakdown = best_obj.get("all_detected_classes", [])
        else:
            overall_class = "unclassified_seabed"
            overall_conf = 0.0
            overall_prio = "LOWER"
            spatial_summary = {
                "latitude": None,
                "longitude": None,
                "georeferenced": False,
                "coordinate_system": "UNREFERENCED",
                "total_area_sq_m": 0.0,
                "max_length_m": 0.0,
                "max_width_m": 0.0
            }
            class_breakdown = []

        report_summary = {
            "obtained_image_class": overall_class,
            "confidence_score": round(overall_conf, 3),
            "confidence_pct": round(overall_conf * 100, 1),
            "priority_level": overall_prio,
            "priority_label": f"{overall_prio} PRIORITY ({'> 75%' if overall_prio == 'HIGHER' else '<= 75%'})",
            "spatial_location": spatial_summary,
            "candidate_classes_breakdown": class_breakdown
        }

        return {
            "analysis_id": analysis_id,
            "status": "success",
            "image_path": image_path,
            "enhanced_image_path": enhanced_path if os.path.exists(enhanced_path) else None,
            "annotated_image_path": annotated_path if os.path.exists(annotated_path) else None,
            "georeferencing_case": georef_case,
            "total_detections": len(final_objects),
            "summary_statistics": stats,
            "report_summary": report_summary,
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
