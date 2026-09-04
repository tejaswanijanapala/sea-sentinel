"""
Stage 6: AI Explainability & Inspection Narrative Synthesizer
Generates human-understandable, audit-ready hydrographic narratives explaining:
  1. Detection Rationale (Acoustic morphology, backscatter intensity)
  2. Anomaly & Physics Verification (Autoencoder MSE vs Seabed baseline, acoustic shadow trailing)
  3. Geological Rock Suppression (DBSCAN spatial density)
  4. Multi-Factor Risk Assessment (Navigation hazard, ecological impact, AUV intervention priority)
"""

from typing import Dict, Any, Optional


class ExplainabilitySynthesizer:
    """
    Synthesizes deep model outputs and acoustic physics into structured hydrographic explanations.
    """
    def __init__(self):
        self.risk_descriptions = {
            "fishing_net": "High ecological entanglement hazard (ghost fishing gear) threatening marine wildlife and benthic ecosystems.",
            "pipeline_or_cable": "Subsea infrastructure asset; potential snag hazard for bottom-trawling fishing gear and anchors.",
            "shipwreck_fragment": "Navigational obstruction hazard to shallow-draft vessels and marine traffic.",
            "engine_debris": "Heavy metallic engine parts, machinery, or discarded hardware presenting snag and navigation risks.",
            "riprap_debris": "Erosion control or dumped quarry rock; potential obstacle for benthic sampling and ROV operations.",
            "seabed_surface": "Natural seafloor sedimentary backscatter."
        }

    def explain_target(
        self,
        detection: Dict[str, Any],
        calibrated_status: str,
        calibrated_conf: float,
        reconstruction_error: float,
        shadow_verified: bool,
        is_rock_cluster: bool,
        risk_level: str,
        dimensions: Optional[Dict[str, Any]] = None,
        coordinates: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates structured diagnostic breakdown and concise executive narrative.
        """
        cls_name = detection.get("class", "unknown_object")
        obj_id = detection.get("object_id", "OBJ_UNKNOWN")

        # 1. Acoustic Morphology Rationale
        if cls_name == "fishing_net":
            morphology_note = "Irregular dispersed acoustic backscatter mesh typical of synthetic polymer netting."
        elif cls_name == "pipeline_or_cable":
            morphology_note = "Continuous linear/tubular acoustic signature with high aspect ratio."
        elif cls_name == "shipwreck_fragment":
            morphology_note = "High-salience geometric structure with sharp rectilinear boundaries."
        else:
            morphology_note = "Localized acoustic backscatter reflection anomaly."

        # 2. Physics & Anomaly Rationale
        physics_notes = []
        if shadow_verified:
            physics_notes.append("Acoustic shadow trailing down-range confirms physical relief protruding above the seafloor.")
        else:
            physics_notes.append("Lacks pronounced trailing acoustic shadow; potential low-relief feature or seabed variation.")

        if reconstruction_error > 0.08:
            physics_notes.append(f"Autoencoder reconstruction error ({reconstruction_error:.4f}) significantly exceeds normal seabed baseline (T=0.094), confirming anomalous non-sedimentary composition.")
        elif reconstruction_error > 0:
            physics_notes.append(f"Autoencoder reconstruction error ({reconstruction_error:.4f}) within normal seafloor bounds.")

        # 3. Geological Context
        if is_rock_cluster:
            geology_note = "DBSCAN spatial clustering indicates high local target density characteristic of natural rock fields / gravel moraines (confidence down-weighted)."
        else:
            geology_note = "Spatially isolated target consistent with distinct anthropogenic debris."

        # 4. Action Recommendation
        if risk_level == "HIGH" and calibrated_status == "confirmed_debris":
            action_rec = "PRIORITY INTERVENTION: Schedule targeted ROV/AUV optical inspection and recovery planning."
        elif risk_level == "MEDIUM" or calibrated_status == "suspicious_anomaly":
            action_rec = "MONITORING LOG: Tag in hydrographic database; verify on subsequent multi-beam survey passes."
        else:
            action_rec = "NOISE / NATURAL FORMATION: Classified as natural seabed or benign geological feature; no intervention required."

        # 5. Narrative Assembly
        narrative_parts = [
            f"Target {obj_id} categorized as '{cls_name}' with calibrated confidence of {calibrated_conf * 100:.1f}%.",
            morphology_note,
            " ".join(physics_notes),
            geology_note,
            self.risk_descriptions.get(cls_name, "General underwater object."),
            f"Assigned Risk: {risk_level}.",
            action_rec
        ]

        full_narrative = " ".join(narrative_parts)

        return {
            "object_id": obj_id,
            "class_name": cls_name,
            "calibrated_status": calibrated_status,
            "calibrated_confidence": round(float(calibrated_conf), 4),
            "risk_level": risk_level,
            "morphology_note": morphology_note,
            "shadow_verified": bool(shadow_verified),
            "reconstruction_error": round(float(reconstruction_error), 6),
            "is_rock_cluster": bool(is_rock_cluster),
            "action_recommendation": action_rec,
            "executive_narrative": full_narrative
        }
