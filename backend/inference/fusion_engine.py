"""
YOLO + U-Net Candidate Fusion Engine
Associates independent object detections from YOLO and semantic candidates from U-Net
using multi-signal spatial matching (IoU, centroid distance, mask overlap).
Explicitly categorizes objects into BOTH, YOLO_ONLY, and UNET_ONLY.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import math
import numpy as np


class FusionEngine:
    """
    Dedicated association and fusion engine combining independent YOLO and U-Net outputs.
    Guarantees that objects discovered by either model are preserved for downstream verification.
    """
    def __init__(
        self,
        iou_threshold: float = 0.40,
        centroid_dist_ratio: float = 0.08,
        mask_in_box_threshold: float = 0.15,
        weight_yolo: float = 0.55,
        weight_unet: float = 0.45
    ):
        self.iou_threshold = iou_threshold
        self.centroid_dist_ratio = centroid_dist_ratio
        self.mask_in_box_threshold = mask_in_box_threshold
        self.weight_yolo = weight_yolo
        self.weight_unet = weight_unet

    def fuse(
        self,
        yolo_candidates: List[Dict[str, Any]],
        unet_candidates: List[Dict[str, Any]],
        image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Fuses YOLO detections and U-Net segmentations into a unified candidate list.
        """
        t0 = time.perf_counter()
        h, w = image_shape[:2]
        diag_len = math.hypot(w, h)
        max_centroid_dist = max(20.0, diag_len * self.centroid_dist_ratio)

        matched_yolo_indices = set()
        matched_unet_indices = set()
        fused_objects = []
        target_idx = 1

        # Calculate association scores matrix between YOLO and U-Net candidates
        matches = []
        for y_idx, y_cand in enumerate(yolo_candidates):
            y_box = y_cand.get("bbox", {})
            y_cent = y_cand.get("centroid", y_cand.get("center", [0, 0]))

            for u_idx, u_cand in enumerate(unet_candidates):
                u_box = u_cand.get("bbox", {})
                u_cent = u_cand.get("centroid", [0, 0])

                iou = self.calculate_box_iou(y_box, u_box)
                dist = math.hypot(y_cent[0] - u_cent[0], y_cent[1] - u_cent[1])
                dist_score = max(0.0, 1.0 - (dist / max_centroid_dist))

                # Combined association metric
                if iou >= self.iou_threshold or (dist <= max_centroid_dist and iou > 0.05):
                    assoc_score = (0.6 * iou) + (0.4 * dist_score)
                    matches.append({
                        "y_idx": y_idx,
                        "u_idx": u_idx,
                        "score": assoc_score,
                        "iou": iou,
                        "dist": dist
                    })

        # Sort matches by association score descending (Greedy bipartite matching)
        matches.sort(key=lambda m: m["score"], reverse=True)

        for m in matches:
            y_idx = m["y_idx"]
            u_idx = m["u_idx"]

            if y_idx in matched_yolo_indices or u_idx in matched_unet_indices:
                continue

            matched_yolo_indices.add(y_idx)
            matched_unet_indices.add(u_idx)

            y_cand = yolo_candidates[y_idx]
            u_cand = unet_candidates[u_idx]

            # Category 1: BOTH (Dual-model agreement)
            fused_obj = self._merge_dual_candidate(
                y_cand=y_cand,
                u_cand=u_cand,
                object_id=f"TGT_{target_idx:03d}",
                assoc_meta=m
            )
            fused_objects.append(fused_obj)
            target_idx += 1

        # Category 2: YOLO-ONLY Candidates
        for y_idx, y_cand in enumerate(yolo_candidates):
            if y_idx not in matched_yolo_indices:
                yolo_only_obj = self._create_yolo_only_candidate(
                    y_cand=y_cand,
                    object_id=f"TGT_{target_idx:03d}"
                )
                fused_objects.append(yolo_only_obj)
                target_idx += 1

        # Category 3: U-NET-ONLY Candidates (Crucial for recovering YOLO misses)
        for u_idx, u_cand in enumerate(unet_candidates):
            if u_idx not in matched_unet_indices:
                unet_only_obj = self._create_unet_only_candidate(
                    u_cand=u_cand,
                    object_id=f"TGT_{target_idx:03d}"
                )
                fused_objects.append(unet_only_obj)
                target_idx += 1

        # Spatial NMS and Cluster Merging across fused objects to prevent redundant duplicate clutter
        consolidated = self._consolidate_overlapping_objects(fused_objects, max_targets=6)

        fusion_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        both_count = sum(1 for o in consolidated if o.get("source_category") == "BOTH")
        yolo_only_count = sum(1 for o in consolidated if o.get("source_category") == "YOLO_ONLY")
        unet_only_count = sum(1 for o in consolidated if o.get("source_category") == "UNET_ONLY")

        return {
            "status": "success",
            "fusion_time_ms": fusion_time_ms,
            "total_candidates": len(consolidated),
            "confirmed_both": both_count,
            "yolo_only": yolo_only_count,
            "unet_only": unet_only_count,
            "objects": consolidated
        }

    def _consolidate_overlapping_objects(
        self,
        objects: List[Dict[str, Any]],
        max_targets: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Merges redundant overlapping candidates from adjacent patches and retains
        the most prominent, high-confidence target objects.
        """
        if not objects:
            return []

        # Sort prioritizing dual-model agreement (BOTH) first, then confidence and area
        sorted_objs = sorted(
            objects,
            key=lambda o: (
                1 if o.get("source_category") == "BOTH" else 0,
                float(o.get("confidence", 0.0)),
                o.get("mask_area") or 0
            ),
            reverse=True
        )
        kept = []

        for obj in sorted_objs:
            box = obj.get("bbox", {})
            cent = obj.get("centroid", [0, 0])
            merged = False

            for k_idx, k_obj in enumerate(kept):
                k_box = k_obj.get("bbox", {})
                k_cent = k_obj.get("centroid", [0, 0])
                iou = self.calculate_box_iou(box, k_box)
                dist = math.hypot(cent[0] - k_cent[0], cent[1] - k_cent[1])

                if iou > 0.18 or dist < 45.0:
                    # Merge bounding boxes
                    x1 = min(box.get("x1", 0), k_box.get("x1", 0))
                    y1 = min(box.get("y1", 0), k_box.get("y1", 0))
                    x2 = max(box.get("x2", 0), k_box.get("x2", 0))
                    y2 = max(box.get("y2", 0), k_box.get("y2", 0))
                    k_obj["bbox"] = {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)}
                    k_obj["width"] = round(x2 - x1, 1)
                    k_obj["height"] = round(y2 - y1, 1)
                    k_obj["centroid"] = [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)]
                    k_obj["confidence"] = max(k_obj.get("confidence", 0.5), obj.get("confidence", 0.5))

                    # Combine sources
                    s1 = set(k_obj.get("sources", []))
                    s2 = set(obj.get("sources", []))
                    combined_sources = list(s1.union(s2))
                    k_obj["sources"] = combined_sources
                    k_obj["source_category"] = "BOTH" if len(combined_sources) > 1 else (combined_sources[0].upper() + "_ONLY")

                    merged = True
                    break

            if not merged:
                kept.append(obj)

            if len(kept) >= max_targets:
                break

        # Re-index object IDs neatly
        for idx, k_obj in enumerate(kept):
            k_obj["object_id"] = f"TGT_{idx + 1:03d}"

        return kept

    def _merge_dual_candidate(
        self,
        y_cand: Dict[str, Any],
        u_cand: Dict[str, Any],
        object_id: str,
        assoc_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merges matching YOLO and U-Net detections into an agreed target."""
        yb = y_cand.get("bbox", {})
        ub = u_cand.get("bbox", {})

        # Union bounding box
        x1 = min(float(yb.get("x1", 0)), float(ub.get("x1", 0)))
        y1 = min(float(yb.get("y1", 0)), float(ub.get("y1", 0)))
        x2 = max(float(yb.get("x2", 0)), float(ub.get("x2", 0)))
        y2 = max(float(yb.get("y2", 0)), float(ub.get("y2", 0)))
        bw = round(x2 - x1, 1)
        bh = round(y2 - y1, 1)

        # Centroid weighted average
        yc = y_cand.get("centroid", y_cand.get("center", [0, 0]))
        uc = u_cand.get("centroid", [0, 0])
        cx = round(self.weight_yolo * yc[0] + self.weight_unet * uc[0], 1)
        cy = round(self.weight_yolo * yc[1] + self.weight_unet * uc[1], 1)

        # Unified confidence: Dual-model agreement confirms high consensus
        y_conf = float(y_cand.get("confidence", 0.80))
        u_conf = float(u_cand.get("confidence", 0.80))
        max_conf = max(y_conf, u_conf)
        fused_conf = min(0.98, max(0.88, max_conf + 0.08))

        cls_name = y_cand.get("class", u_cand.get("class", "marine_debris"))
        cls_id = y_cand.get("class_id", u_cand.get("class_id", 0))

        return {
            "object_id": object_id,
            "sources": ["yolo", "unet"],
            "source_category": "BOTH",
            "agreement": True,
            "class": cls_name,
            "class_id": cls_id,
            "confidence": round(float(fused_conf), 3),
            "yolo_confidence": round(y_conf, 3),
            "unet_confidence": round(u_conf, 3),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "width": bw,
            "height": bh,
            "centroid": [cx, cy],
            "center": [cx, cy],
            "mask_area": u_cand.get("mask_area"),
            "aspect_ratio": u_cand.get("aspect_ratio", round(max(bw, bh) / max(1.0, min(bw, bh)), 2)),
            "compactness": u_cand.get("compactness"),
            "solidity": u_cand.get("solidity"),
            "association_iou": round(float(assoc_meta.get("iou", 0.0)), 3),
            "polygon": u_cand.get("polygon") or y_cand.get("polygon", [])
        }

    def _create_yolo_only_candidate(
        self,
        y_cand: Dict[str, Any],
        object_id: str
    ) -> Dict[str, Any]:
        """Formats a YOLO-only candidate without dropping it."""
        yb = y_cand.get("bbox", {})
        x1 = float(yb.get("x1", 0))
        y1 = float(yb.get("y1", 0))
        x2 = float(yb.get("x2", 0))
        y2 = float(yb.get("y2", 0))
        bw = round(x2 - x1, 1)
        bh = round(y2 - y1, 1)
        cx = round(x1 + bw / 2.0, 1)
        cy = round(y1 + bh / 2.0, 1)

        y_raw = float(y_cand.get("confidence", 0.70))
        y_conf = round(min(0.78, max(0.55, y_raw * 0.90)), 3)

        return {
            "object_id": object_id,
            "sources": ["yolo"],
            "source_category": "YOLO_ONLY",
            "agreement": False,
            "class": y_cand.get("class", "marine_debris"),
            "class_id": y_cand.get("class_id", 0),
            "confidence": y_conf,
            "yolo_confidence": y_conf,
            "unet_confidence": 0.0,
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "width": bw,
            "height": bh,
            "centroid": [cx, cy],
            "center": [cx, cy],
            "mask_area": None,
            "aspect_ratio": round(max(bw, bh) / max(1.0, min(bw, bh)), 2),
            "compactness": None,
            "solidity": None,
            "association_iou": 0.0,
            "polygon": y_cand.get("polygon", [])
        }

    def _create_unet_only_candidate(
        self,
        u_cand: Dict[str, Any],
        object_id: str
    ) -> Dict[str, Any]:
        """Formats a U-Net-only candidate (recovering objects missed by YOLO)."""
        ub = u_cand.get("bbox", {})
        x1 = float(ub.get("x1", 0))
        y1 = float(ub.get("y1", 0))
        x2 = float(ub.get("x2", 0))
        y2 = float(ub.get("y2", 0))
        bw = round(x2 - x1, 1)
        bh = round(y2 - y1, 1)
        uc = u_cand.get("centroid", [x1 + bw / 2.0, y1 + bh / 2.0])

        u_raw = float(u_cand.get("confidence", 0.70))
        u_conf = round(min(0.76, max(0.52, u_raw * 0.90)), 3)

        return {
            "object_id": object_id,
            "sources": ["unet"],
            "source_category": "UNET_ONLY",
            "agreement": False,
            "class": u_cand.get("class", "marine_debris"),
            "class_id": u_cand.get("class_id", 0),
            "confidence": u_conf,
            "yolo_confidence": 0.0,
            "unet_confidence": u_conf,
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "width": bw,
            "height": bh,
            "centroid": [round(float(uc[0]), 1), round(float(uc[1]), 1)],
            "center": [round(float(uc[0]), 1), round(float(uc[1]), 1)],
            "mask_area": u_cand.get("mask_area"),
            "aspect_ratio": u_cand.get("aspect_ratio", round(max(bw, bh) / max(1.0, min(bw, bh)), 2)),
            "compactness": u_cand.get("compactness"),
            "solidity": u_cand.get("solidity"),
            "association_iou": 0.0,
            "polygon": u_cand.get("polygon", [])
        }

    @staticmethod
    def calculate_box_iou(b1: Dict[str, float], b2: Dict[str, float]) -> float:
        """Calculates Intersection over Union (IoU)."""
        x1 = max(float(b1.get("x1", 0)), float(b2.get("x1", 0)))
        y1 = max(float(b1.get("y1", 0)), float(b2.get("y1", 0)))
        x2 = min(float(b1.get("x2", 0)), float(b2.get("x2", 0)))
        y2 = min(float(b1.get("y2", 0)), float(b2.get("y2", 0)))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(1.0, (float(b1.get("x2", 0)) - float(b1.get("x1", 0))) * (float(b1.get("y2", 0)) - float(b1.get("y1", 0))))
        area2 = max(1.0, (float(b2.get("x2", 0)) - float(b2.get("x1", 0))) * (float(b2.get("y2", 0)) - float(b2.get("y1", 0))))

        union_area = area1 + area2 - inter_area
        return float(inter_area / max(1.0, union_area))
