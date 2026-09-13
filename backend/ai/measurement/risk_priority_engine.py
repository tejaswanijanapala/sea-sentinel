"""
Debris Risk and Inspection Priority Scoring Engine
Implements the explainable 3-tier scoring framework:
  1. AI Detection Confidence (0-100%): Deep learning detection salience & classification certainty.
  2. Environmental / Hazard Risk (0-100): Intrinsic environmental damage & maritime operational hazard.
  3. Inspection Priority Score (0-100): Operational decision metric for inspection/intervention sequencing.

Maintains strict separation between AI confidence, Hazard Risk, and Inspection Priority.
Sonar quality solely modulates evidence reliability, never intrinsic hazard.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


class RiskPriorityEngine:
    """
    Production scoring engine for underwater debris hazard assessment
    and transparent inspection prioritization.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        risk_cfg = self.config.get("risk_priority", {})

        # 1. Configurable Hazard Weights by Debris Type
        raw_hw = risk_cfg.get("hazard_weights", {
            "fishing_net": 92,        # Critical (wildlife entanglement, ghost fishing)
            "ghost_net": 94,          # Critical (wildlife entanglement, ghost fishing)
            "ghost_gear": 94,         # Critical
            "shipwreck_fragment": 88, # High-Critical (navigation obstacle, structural hazard)
            "shipwreck": 90,
            "cargo_container": 92,    # High-Critical (subsurface hull collision risk)
            "container": 92,
            "engine_debris": 85,      # High (pollutant leaching, heavy snag hazard)
            "engine": 85,
            "heavy_machinery": 85,
            "pipeline_or_cable": 80,  # High (infrastructure asset, anchor snag)
            "pipe_cable": 80,
            "marine_plastic_drum": 78,
            "drum_or_barrel": 78,
            "plastic_debris": 58,     # Moderate (macroplastic, ecosystem degradation)
            "plastic_fragment": 58,
            "riprap_debris": 48,      # Moderate (quarry stone, localized obstruction)
            "tire_or_rubber": 52,
            "munitions_or_uxo": 99,
            "default": 65
        })
        self.hazard_weights = {}
        for k, v in raw_hw.items():
            try:
                # Handle possible string representations or inline comment strings
                clean_v = str(v).split("#")[0].strip()
                self.hazard_weights[k] = float(clean_v)
            except Exception:
                self.hazard_weights[k] = 65.0

        # 2. Configurable Priority Formula Weights
        self.priority_weights = risk_cfg.get("priority_formula_weights", {
            "hazard_risk": 0.45,
            "detection_evidence": 0.25,
            "object_extent": 0.15,
            "location_sensitivity": 0.15
        })

        # 3. Sonar Quality Reliability Modulation
        self.reliability_modulation = risk_cfg.get("reliability_modulation", {
            "optimal": 1.00,
            "good": 0.98,
            "moderate": 0.92,
            "noisy": 0.85
        })

        # 4. Area thresholds (sq meters)
        self.area_thresholds = risk_cfg.get("area_thresholds", {
            "small_max_sq_m": 5.0,
            "medium_max_sq_m": 25.0
        })

    def categorize_score(self, score: float) -> str:
        """Categorizes 0-100 score into standard 4-tier risk categories."""
        if score >= 80.0:
            return "CRITICAL"
        elif score >= 60.0:
            return "HIGH"
        elif score >= 40.0:
            return "MODERATE"
        else:
            return "LOW"

    def estimate_sonar_quality(
        self,
        image_context: Optional[np.ndarray] = None,
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """
        Assesses acoustic sonar quality (SNR, contrast, speckle).
        Quality directly informs evidence reliability, NOT hazard severity.
        """
        if quality_metrics:
            contrast = quality_metrics.get("contrast_score", 0.6)
            shadow = quality_metrics.get("shadow_score", 0.5)
            avg_q = (contrast + shadow) / 2.0
            if avg_q >= 0.70:
                return "optimal", 1.00
            elif avg_q >= 0.50:
                return "good", 0.95
            elif avg_q >= 0.35:
                return "moderate", 0.85
            else:
                return "noisy", 0.75

        if image_context is not None and isinstance(image_context, np.ndarray) and image_context.size > 0:
            std_val = float(np.std(image_context))
            if std_val > 45.0:
                return "good", 0.95
            elif std_val > 25.0:
                return "moderate", 0.85
            else:
                return "noisy", 0.75

        return "good", 0.95

    def evaluate_object_size(
        self,
        target: Dict[str, Any],
        dimensions: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, float, str]:
        """
        Evaluates physical object extent.
        Prefers U-Net segmented polygon area over bounding box area when available.
        Returns: (size_category, area_sq_m_or_px, extent_score_0_100, source_type)
        """
        area_sq_m = None
        source_type = "bounding_box"

        # Check for segmented polygon area
        poly = target.get("polygon")
        if poly and len(poly) >= 3:
            # Calculate polygon area via Shoelace formula
            pts = np.array(poly, dtype=np.float32)
            x = pts[:, 0]
            y = pts[:, 1]
            poly_px_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            source_type = "unet_segmentation"
        else:
            bbox = target.get("bbox") or target.get("pixel_bbox") or {}
            w = max(1.0, float(bbox.get("x2", 0)) - float(bbox.get("x1", 0)))
            h = max(1.0, float(bbox.get("y2", 0)) - float(bbox.get("y1", 0)))
            poly_px_area = w * h

        if dimensions and dimensions.get("area_sq_m") is not None:
            area_sq_m = float(dimensions["area_sq_m"])
        elif target.get("area_sq_m") is not None:
            area_sq_m = float(target["area_sq_m"])

        # Determine size category and normalized extent contribution (0-100)
        if area_sq_m is not None and area_sq_m > 0:
            if area_sq_m > self.area_thresholds["medium_max_sq_m"]:
                size_cat = "large"
                extent_score = min(100.0, 75.0 + (area_sq_m / 100.0) * 25.0)
            elif area_sq_m >= self.area_thresholds["small_max_sq_m"]:
                size_cat = "medium"
                extent_score = 45.0 + ((area_sq_m - 5.0) / 20.0) * 30.0
            else:
                size_cat = "small"
                extent_score = max(15.0, (area_sq_m / 5.0) * 45.0)
            eff_area = round(area_sq_m, 2)
        else:
            # Fallback based on pixel area
            if poly_px_area > 3500:
                size_cat = "large"
                extent_score = 85.0
            elif poly_px_area > 800:
                size_cat = "medium"
                extent_score = 55.0
            else:
                size_cat = "small"
                extent_score = 25.0
            eff_area = round(poly_px_area, 1)

        return size_cat, eff_area, round(extent_score, 1), source_type

    def evaluate_location_sensitivity(
        self,
        target: Dict[str, Any],
        raster_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, str]:
        """
        Evaluates geographic context sensitivity (coral reef corridors, protected zones, shallow navigation channels).
        Returns: (sensitivity_level, sensitivity_score_0_100, description)
        """
        lat = target.get("latitude") or target.get("lat")
        lon = target.get("longitude") or target.get("lon")
        profile = (raster_meta.get("dataset_profile") if raster_meta else "") or ""
        profile_lower = profile.lower()

        # Check for known sensitive environments or coastal profiles
        if "breton" in profile_lower or "sanctuary" in profile_lower or "reef" in profile_lower or "coral" in profile_lower:
            return "high", 88.0, "Ecologically sensitive marine sanctuary / coral corridor"
        elif "gulf" in profile_lower or "quanzhou" in profile_lower or "dongying" in profile_lower or "shipping" in profile_lower:
            return "high", 80.0, "High-density coastal fishing & maritime transport corridor"
        elif lat is not None and lon is not None:
            # Georeferenced target in active waters
            return "medium", 60.0, "Active hydrographic survey zone"
        else:
            # Unreferenced acoustic chip fallback
            return "medium", 50.0, "Standard unreferenced acoustic survey sector"

    def calculate_debris_scores(
        self,
        target: Dict[str, Any],
        image_context: Optional[np.ndarray] = None,
        raster_meta: Optional[Dict[str, Any]] = None,
        dimensions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes the complete explainable scoring suite for a single detected debris target.
        """
        debris_id = target.get("object_id", "D00")
        raw_class = (target.get("class") or "marine_debris").lower()

        # -------------------------------------------------------------
        # 1. Detection Confidence (0.0 to 1.0 / 0 to 100%)
        # -------------------------------------------------------------
        det_conf = float(target.get("calibrated_confidence") or target.get("confidence") or 0.85)
        det_conf = min(1.0, max(0.0, det_conf))
        conf_pct = round(det_conf * 100, 1)

        # -------------------------------------------------------------
        # 2. Debris Type Base Hazard
        # -------------------------------------------------------------
        raw_val = None
        if raw_class in self.hazard_weights:
            raw_val = self.hazard_weights[raw_class]
        else:
            for k, v in self.hazard_weights.items():
                if k in raw_class or raw_class in k:
                    raw_val = v
                    break
        if raw_val is None:
            raw_val = self.hazard_weights.get("default", 65.0)

        try:
            base_hazard = float(raw_val)
        except Exception:
            base_hazard = 65.0
        
        # Determine qualitative marine hazard impact
        if base_hazard >= 85.0:
            marine_hazard_level = "very_high"
            marine_hazard_desc = "Severe threat to marine wildlife (ghost entanglement) and shallow vessel propulsion."
        elif base_hazard >= 75.0:
            marine_hazard_level = "high"
            marine_hazard_desc = "High impact on benthic habitat, commercial trawl gear snagging, and structural integrity."
        elif base_hazard >= 60.0:
            marine_hazard_level = "medium"
            marine_hazard_desc = "Moderate ecological degradation and marine debris accumulation."
        else:
            marine_hazard_level = "low"
            marine_hazard_desc = "Localized seabed obstruction with low biological toxicity."

        # -------------------------------------------------------------
        # 3. Object Extent / Size
        # -------------------------------------------------------------
        size_cat, eff_area, extent_score, extent_source = self.evaluate_object_size(target, dimensions)

        # -------------------------------------------------------------
        # 4. Location Sensitivity
        # -------------------------------------------------------------
        loc_level, loc_score, loc_desc = self.evaluate_location_sensitivity(target, raster_meta)

        # -------------------------------------------------------------
        # 5. Sonar Quality Assessment (Modulates Evidence Reliability)
        # -------------------------------------------------------------
        sonar_qual, reliability_mod = self.estimate_sonar_quality(image_context, target.get("quality_metrics"))

        # -------------------------------------------------------------
        # 6. Environmental / Hazard Risk Calculation (0-100)
        # Hazard Risk = Base Debris Hazard (60%) + Extent (25%) + Location Sensitivity (15%)
        # -------------------------------------------------------------
        raw_hazard_risk = (
            base_hazard * 0.60 +
            extent_score * 0.25 +
            loc_score * 0.15
        )
        hazard_risk = int(round(min(100.0, max(5.0, raw_hazard_risk))))
        hazard_level = self.categorize_score(hazard_risk)

        # -------------------------------------------------------------
        # 7. Inspection Priority Score Calculation (0-100)
        # Priority Score = [ Hazard Risk * w_h + Evidence(Confidence*100) * w_e + Extent * w_x + Loc * w_l ] * Reliability
        # -------------------------------------------------------------
        w_h = self.priority_weights.get("hazard_risk", 0.45)
        w_e = self.priority_weights.get("detection_evidence", 0.25)
        w_x = self.priority_weights.get("object_extent", 0.15)
        w_l = self.priority_weights.get("location_sensitivity", 0.15)

        # Dual-Model Consensus Bonus: Targets confirmed by BOTH YOLOv11 and Attention U-Net
        # possess highest acoustic verification credibility and urgent inspection priority.
        is_both = (
            target.get("source_category") == "BOTH" or
            len(target.get("sources", [])) > 1 or
            target.get("agreement", False)
        )
        dual_bonus = 12.0 if is_both else 0.0

        raw_priority = (
            hazard_risk * w_h +
            conf_pct * w_e +
            extent_score * w_x +
            loc_score * w_l +
            dual_bonus
        ) * reliability_mod

        priority_score = int(round(min(100.0, max(5.0, raw_priority))))
        priority_level = self.categorize_score(priority_score)

        # -------------------------------------------------------------
        # 8. Dynamic Factual Reasons Synthesis
        # -------------------------------------------------------------
        reasons = []
        clean_name = raw_class.replace("_", " ").title()

        # A. Debris Type
        if raw_class in ["fishing_net", "ghost_net"]:
            reasons.append(f"Ghost fishing net detected (critical wildlife entanglement threat)")
        elif raw_class == "shipwreck_fragment":
            reasons.append(f"Shipwreck fragment / structural debris (navigation hazard)")
        elif raw_class == "engine_debris":
            reasons.append(f"Metallic engine debris / machinery on seafloor")
        elif raw_class == "pipeline_or_cable":
            reasons.append(f"Subsea pipeline / cable asset (bottom gear snag risk)")
        elif raw_class == "riprap_debris":
            reasons.append(f"Riprap / quarry rock formation on seabed")
        else:
            reasons.append(f"{clean_name} debris signature identified")

        # B. AI Detection Evidence & Provenance
        if is_both:
            reasons.append("Parallel Dual-Path Consensus: Confirmed simultaneously by both YOLOv11 (bounding box) and Attention U-Net (pixel contour)")
        elif target.get("source_category") == "UNET_ONLY":
            reasons.append("Discovered exclusively by Attention U-Net semantic segmenter (recovered YOLO miss)")
        elif target.get("source_category") == "YOLO_ONLY":
            reasons.append("Discovered exclusively by YOLOv11 high-speed object detector")

        if conf_pct >= 90:
            reasons.append(f"High AI detection confidence ({conf_pct}%)")
        elif conf_pct >= 70:
            reasons.append(f"Validated AI detection confidence ({conf_pct}%)")
        else:
            reasons.append(f"Moderate AI confidence ({conf_pct}%) flagged for operator inspection")

        # C. Object Extent
        area_unit = "m²" if extent_source == "unet_segmentation" or dimensions else "px²"
        if size_cat == "large":
            reasons.append(f"Large estimated extent ({eff_area} {area_unit} area)")
        elif size_cat == "medium":
            reasons.append(f"Medium physical footprint ({eff_area} {area_unit})")
        else:
            reasons.append(f"Compact debris target ({eff_area} {area_unit})")

        # D. Marine Impact & Location
        if marine_hazard_level in ["very_high", "high"]:
            reasons.append("High marine ecosystem & vessel navigation hazard")
        if loc_level == "high":
            reasons.append("Located within sensitive marine corridor or fishing zone")

        # E. Sonar Evidence
        if sonar_qual in ["optimal", "good"]:
            reasons.append("Good acoustic backscatter contrast and shadow relief")
        elif sonar_qual == "moderate":
            reasons.append("Acceptable sonar quality supporting candidate evidence")

        # -------------------------------------------------------------
        # 9. Natural Language Dynamic Explanation
        # -------------------------------------------------------------
        prio_term = priority_level.lower()
        dynamic_explanation = (
            f"This target has been assigned an inspection priority of {priority_score}/100 ({priority_level}) "
            f"because it was classified as '{clean_name}' with {conf_pct}% AI detection confidence, "
            f"{size_cat} estimated extent ({eff_area} {area_unit}), and {marine_hazard_level.replace('_', ' ')} potential marine impact. "
            f"{loc_desc}."
        )

        # -------------------------------------------------------------
        # 10. Operational Action Recommendation
        # -------------------------------------------------------------
        if priority_level == "CRITICAL":
            recommendation = "Prioritize this debris for immediate ROV/diver inspection and recovery intervention."
            rec_code = "IMMEDIATE_INTERVENTION"
        elif priority_level == "HIGH":
            recommendation = "Prioritize this debris for inspection and possible salvage or containment."
            rec_code = "SCHEDULE_INSPECTION"
        elif priority_level == "MEDIUM":
            recommendation = "Monitor this debris and consider further multi-beam hydrographic investigation."
            rec_code = "MONITOR_TARGET"
        else:
            recommendation = "Low priority; continue monitoring on standard survey passes."
            rec_code = "LOW_PRIORITY"

        # -------------------------------------------------------------
        # 11. Contributing Factors Breakdown for Progress Bars
        # -------------------------------------------------------------
        contributing_factors = {
            "detection_confidence": {
                "label": "AI Detection Confidence",
                "value": conf_pct,
                "pct": conf_pct,
                "display": f"{conf_pct}%",
                "status": "High" if conf_pct >= 75 else ("Medium" if conf_pct >= 50 else "Low"),
                "color": "cyan"
            },
            "object_extent": {
                "label": "Object Extent",
                "value": round(extent_score, 1),
                "pct": round(extent_score, 1),
                "display": f"{size_cat.upper()} ({eff_area} {area_unit})",
                "status": size_cat.capitalize(),
                "color": "magenta"
            },
            "marine_hazard": {
                "label": "Marine Hazard Impact",
                "value": base_hazard,
                "pct": base_hazard,
                "display": marine_hazard_level.replace("_", " ").upper(),
                "status": marine_hazard_level.replace("_", " ").capitalize(),
                "color": "red" if base_hazard >= 75 else ("amber" if base_hazard >= 50 else "green")
            },
            "location_sensitivity": {
                "label": "Location Sensitivity",
                "value": round(loc_score, 1),
                "pct": round(loc_score, 1),
                "display": loc_level.upper(),
                "status": loc_level.capitalize(),
                "color": "amber" if loc_score >= 70 else "blue"
            },
            "sonar_reliability": {
                "label": "Sonar Evidence Reliability",
                "value": round(reliability_mod * 100, 1),
                "pct": round(reliability_mod * 100, 1),
                "display": sonar_qual.upper(),
                "status": sonar_qual.capitalize(),
                "color": "green" if reliability_mod >= 0.90 else "amber"
            }
        }

        return {
            "debris_id": debris_id,
            "type": raw_class,
            "display_name": clean_name,
            # Strict 3-Metric Separation
            "detection_confidence": round(det_conf, 4),
            "detection_confidence_pct": conf_pct,
            "hazard_risk": hazard_risk,
            "hazard_risk_level": hazard_level,
            "priority_score": priority_score,
            "priority_level": priority_level,
            "risk_level": priority_level.lower(), # Backward compatibility
            # Factors & Details
            "object_size": size_cat,
            "object_area": eff_area,
            "object_area_unit": area_unit,
            "extent_source": extent_source,
            "marine_hazard": marine_hazard_level,
            "location_sensitivity": loc_level,
            "sonar_quality": sonar_qual,
            "reliability_multiplier": reliability_mod,
            "contributing_factors": contributing_factors,
            "reasons": reasons,
            "dynamic_explanation": dynamic_explanation,
            "recommendation": recommendation,
            "recommendation_code": rec_code
        }
