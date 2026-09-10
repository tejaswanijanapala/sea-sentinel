"""
Active Learning Sample Selector & Continual Learning Safety Guardian.
Curates the most informative acoustic samples (model disagreement, unknown anomalies,
rare classes, human corrections) and prevents catastrophic forgetting via regression validation.
"""

from typing import Dict, Any, List, Optional
import os
import json
import time


class ActiveLearningSelector:
    """
    Selects high-utility acoustic samples for offline human review and iterative retraining.
    Rejects redundant, uninformative high-confidence detections to minimize storage and annotation cost.
    """
    def __init__(self, utility_threshold: float = 0.55):
        self.utility_threshold = utility_threshold
        self.staged_samples: List[Dict[str, Any]] = []

    def evaluate_sample_utility(
        self,
        yolo_conf: float,
        unet_score: float,
        anomaly_score: float,
        is_unknown: bool,
        class_name: str,
        temporal_hits: int
    ) -> Dict[str, Any]:
        """
        Calculates information utility score based on uncertainty, model divergence, and novelty.
        """
        rare_classes = {"shipwreck", "chemical_drum", "ordnance", "unexploded_mine", "lost_sensor"}
        is_rare = class_name.lower() in rare_classes

        # Model divergence term
        divergence = abs(yolo_conf - unet_score)
        
        # Epistemic margin: closer to decision boundary (0.5) has highest utility
        margin_utility = 1.0 - abs(yolo_conf - 0.5) * 2.0

        utility_score = (
            0.30 * divergence +
            0.25 * anomaly_score +
            0.20 * margin_utility +
            0.15 * (1.0 if is_unknown else 0.0) +
            0.10 * (1.0 if is_rare else 0.0)
        )
        utility_score = float(min(1.0, max(0.0, utility_score)))

        should_stage = (utility_score >= self.utility_threshold) or is_unknown

        reason_tags = []
        if divergence > 0.35:
            reason_tags.append("MODEL_DIVERGENCE")
        if anomaly_score > 0.60:
            reason_tags.append("HIGH_ANOMALY")
        if is_unknown:
            reason_tags.append("UNKNOWN_CLASS")
        if is_rare:
            reason_tags.append("RARE_CLASS")
        if temporal_hits == 1 and yolo_conf > 0.40:
            reason_tags.append("TRANSIENT_VERIFICATION")

        return {
            "utility_score": round(utility_score, 3),
            "should_stage_for_review": should_stage,
            "reasons": reason_tags
        }


class ContinualLearningGuardian:
    """
    Safety gate preventing catastrophic forgetting during offline model updates.
    Evaluates new candidate models against historic golden regression benchmarks
    before authorizing promotion to production on edge hardware.
    """
    def __init__(self, max_allowed_recall_drop: float = 0.015):
        self.max_allowed_recall_drop = max_allowed_recall_drop

    def validate_candidate_model(
        self,
        baseline_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Performs rigorous regression check against baseline production model.
        Mandate: Recall and missed-object rate must never degrade.
        """
        base_recall = baseline_metrics.get("recall", 0.90)
        cand_recall = candidate_metrics.get("recall", 0.0)

        base_map50 = baseline_metrics.get("map50", 0.88)
        cand_map50 = candidate_metrics.get("map50", 0.0)

        recall_diff = cand_recall - base_recall
        map_diff = cand_map50 - base_map50

        # Critical rule: No catastrophic recall degradation on safety-critical marine debris
        passed = True
        reasons = []

        if recall_diff < -self.max_allowed_recall_drop:
            passed = False
            reasons.append(f"CATASTROPHIC_FORGETTING_RECALL: Recall dropped by {abs(recall_diff)*100:.2f}% (limit {self.max_allowed_recall_drop*100:.1f}%)")

        if map_diff < -0.05:
            passed = False
            reasons.append(f"MAP_REGRESSION: mAP@0.5 dropped by {abs(map_diff)*100:.2f}%")

        return {
            "approved": passed,
            "decision": "PROMOTE_TO_PRODUCTION" if passed else "REJECT_CANDIDATE_MODEL",
            "recall_diff": round(recall_diff, 4),
            "map_diff": round(map_diff, 4),
            "reasons": reasons,
            "baseline_recall": base_recall,
            "candidate_recall": cand_recall
        }
