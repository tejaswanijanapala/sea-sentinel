"""
Layer 3: YOLO Debris Detection Core
Integrates Ultralytics YOLOv11/v8 for candidate region detection on Side-Scan Sonar imagery.
Supports configurable confidence thresholds, high-recall candidate extraction,
NMS filtering, and standardized candidate object schema.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YOLODetector:
    """
    Side-Scan Sonar Object Detector powered by Ultralytics YOLO.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.25, # High-recall threshold
        iou_thresh: float = 0.45,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Resolve device for Ultralytics (which rejects 'auto')
        if device == "auto" or device is None:
            try:
                import torch
                self.device = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device
        self.model = None
        self.is_model_loaded = False

        # Harmonized 5 debris classes
        self.classes = {
            0: "fishing_net",
            1: "pipeline_or_cable",
            2: "shipwreck_fragment",
            3: "engine_debris",
            4: "riprap_debris"
        }

        self._load_model()

    def _load_model(self):
        """
        Loads trained checkpoint or base YOLO model if available.
        """
        if not ULTRALYTICS_AVAILABLE:
            self.is_model_loaded = False
            return

        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.is_model_loaded = True
                if hasattr(self.model, "names") and self.model.names:
                    # Update classes if custom trained
                    self.classes = {int(k): v for k, v in self.model.names.items()}
            except Exception as e:
                print(f"[YOLODetector] Warning: Failed to load weights from {self.model_path}: {e}")
                self.is_model_loaded = False
        else:
            # Fallback to custom trained best.pt in project root models/yolo/best.pt
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            project_dir = os.path.dirname(backend_dir)
            trained_candidates = [
                os.path.join(project_dir, "models", "yolo", "best.onnx"),
                os.path.join(project_dir, "models", "yolo", "best_fp16.pt"),
                os.path.join(project_dir, "models", "yolo", "best.pt"),
                os.path.join(backend_dir, "models", "yolo", "best.pt"),
                os.path.join(backend_dir, "yolo11n.pt")
            ]
            loaded = False
            for cand in trained_candidates:
                if os.path.exists(cand):
                    try:
                        self.model = YOLO(cand)
                        self.is_model_loaded = True
                        if hasattr(self.model, "names") and self.model.names:
                            self.classes = {int(k): v for k, v in self.model.names.items()}
                        loaded = True
                        break
                    except Exception as e:
                        print(f"[YOLODetector] Warning: Failed to load from {cand}: {e}")
            if not loaded:
                self.is_model_loaded = False

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        conf_override: Optional[float] = None,
        tile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs high-recall object detection on sonar image or tile.
        Returns:
          - detections: list of dicts with bbox, class, confidence, centroid, source="yolo"
          - model_loaded: boolean
          - inference_time_ms: float
        """
        t0 = time.perf_counter()
        conf = conf_override if conf_override is not None else self.conf_thresh

        if not self.is_model_loaded:
            return {
                "status": "model_unavailable",
                "message": "Trained YOLO weights not found. Use training/train_yolo.py to train on the SSS dataset.",
                "model_loaded": False,
                "confidence_threshold": conf,
                "detections": [],
                "inference_time_ms": 0.0
            }

        try:
            # Run real Ultralytics inference
            results = self.model.predict(
                source=image_input,
                conf=conf,
                iou=self.iou_thresh,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            return {
                "status": "error",
                "message": f"YOLO inference failed: {str(e)}",
                "model_loaded": self.is_model_loaded,
                "confidence_threshold": conf,
                "detections": [],
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2)
            }

        detections = []
        det_id = 1

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0].cpu().numpy())
                cls_idx = int(box.cls[0].cpu().numpy())
                cls_name = self.classes.get(cls_idx, "marine_debris")

                x1 = round(float(xyxy[0]), 1)
                y1 = round(float(xyxy[1]), 1)
                x2 = round(float(xyxy[2]), 1)
                y2 = round(float(xyxy[3]), 1)
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                cx = round(x1 + bw / 2.0, 1)
                cy = round(y1 + bh / 2.0, 1)

                det_record = {
                    "object_id": f"YOLO_{det_id:03d}" if not tile_id else f"{tile_id}_YOLO_{det_id:03d}",
                    "source": "yolo",
                    "model": "YOLOv11",
                    "class": cls_name,
                    "class_id": cls_idx,
                    "confidence": round(conf_val, 3),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "width": round(bw, 1),
                    "height": round(bh, 1),
                    "center": [cx, cy],
                    "centroid": [cx, cy]
                }
                if tile_id:
                    det_record["tile_id"] = tile_id

                detections.append(det_record)
                det_id += 1

        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "status": "success",
            "model_loaded": True,
            "confidence_threshold": conf,
            "detections": detections,
            "total_detections": len(detections),
            "inference_time_ms": inference_time_ms
        }

    def detect_batch(
        self,
        image_inputs: List[np.ndarray],
        tile_ids: Optional[List[str]] = None,
        conf_override: Optional[float] = None,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Runs batched high-recall object detection on a list of tiles or images.
        Returns a list of detection result dictionaries, one per input image.
        """
        if not image_inputs:
            return []

        t0 = time.perf_counter()
        conf = conf_override if conf_override is not None else self.conf_thresh

        if not self.is_model_loaded:
            return [
                {
                    "status": "model_unavailable",
                    "detections": [],
                    "total_detections": 0,
                    "inference_time_ms": 0.0,
                    "tile_id": tile_ids[i] if tile_ids and i < len(tile_ids) else None
                }
                for i in range(len(image_inputs))
            ]

        batch_outputs = []
        try:
            # Run batched Ultralytics inference
            results = self.model.predict(
                source=image_inputs,
                conf=conf,
                iou=self.iou_thresh,
                device=self.device,
                batch=min(batch_size, len(image_inputs)),
                verbose=False
            )
        except Exception as e:
            # Fallback on per-image error
            return [
                {
                    "status": "error",
                    "message": str(e),
                    "detections": [],
                    "total_detections": 0,
                    "inference_time_ms": 0.0,
                    "tile_id": tile_ids[i] if tile_ids and i < len(tile_ids) else None
                }
                for i in range(len(image_inputs))
            ]

        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        avg_ms_per_tile = round(total_ms / max(1, len(image_inputs)), 2)

        for i, r in enumerate(results):
            tile_id = tile_ids[i] if tile_ids and i < len(tile_ids) else None
            detections = []
            boxes = r.boxes
            if boxes is not None:
                det_id = 1
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf_val = float(box.conf[0].cpu().numpy())
                    cls_idx = int(box.cls[0].cpu().numpy())
                    cls_name = self.classes.get(cls_idx, "marine_debris")

                    x1 = round(float(xyxy[0]), 1)
                    y1 = round(float(xyxy[1]), 1)
                    x2 = round(float(xyxy[2]), 1)
                    y2 = round(float(xyxy[3]), 1)
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    cx = round(x1 + bw / 2.0, 1)
                    cy = round(y1 + bh / 2.0, 1)

                    det_record = {
                        "object_id": f"{tile_id}_YOLO_{det_id:03d}" if tile_id else f"YOLO_{det_id:03d}",
                        "source": "yolo",
                        "model": "YOLOv11",
                        "class": cls_name,
                        "class_id": cls_idx,
                        "confidence": round(conf_val, 3),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "width": round(bw, 1),
                        "height": round(bh, 1),
                        "center": [cx, cy],
                        "centroid": [cx, cy]
                    }
                    if tile_id:
                        det_record["tile_id"] = tile_id

                    detections.append(det_record)
                    det_id += 1

            batch_outputs.append({
                "status": "success",
                "model_loaded": True,
                "confidence_threshold": conf,
                "tile_id": tile_id,
                "detections": detections,
                "total_detections": len(detections),
                "inference_time_ms": avg_ms_per_tile
            })

        return batch_outputs

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Draws bounding boxes and class labels directly onto the image canvas.
        """
        if image is None:
            return image

        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        default_colors = {
            "fishing_net": (0, 0, 255),          # Red
            "pipeline_or_cable": (255, 165, 0),  # Orange
            "shipwreck_fragment": (0, 140, 255), # Deep Orange
            "engine_debris": (0, 255, 255),      # Yellow / Gold
            "riprap_debris": (255, 0, 255)       # Magenta
        }
        colors = color_map or default_colors

        for d in detections:
            bbox = d.get("bbox", {})
            if not bbox:
                continue
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cls_name = d.get("class", "debris")
            conf = d.get("confidence", 0.0)

            col = colors.get(cls_name, (0, 255, 0))

            # Draw bounding rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)

            # Draw label banner
            label = f"{cls_name} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - baseline - 4)), (x1 + text_w + 4, y1), col, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(text_h, y1 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated
