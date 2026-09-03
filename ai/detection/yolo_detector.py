"""
Layer 3: YOLO Debris Detection Core
Integrates Ultralytics YOLOv11/v8 for candidate region detection on Side-Scan Sonar imagery.
Supports configurable confidence thresholds, NMS filtering, and bounding box extraction.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import os
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
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = None
        self.is_model_loaded = False

        # Harmonized classes
        self.classes = {
            0: "fishing_net",
            1: "pipeline_or_cable",
            2: "shipwreck_fragment",
            3: "engineering_platform",
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
            except Exception:
                self.is_model_loaded = False
        else:
            self.is_model_loaded = False

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        conf_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs object detection on sonar image or tile.
        Returns:
          - detections: list of dicts with bbox, class, confidence
          - model_loaded: boolean indicating real weights vs. untrained status
        """
        conf = conf_override if conf_override is not None else self.conf_thresh

        if not self.is_model_loaded:
            return {
                "status": "model_unavailable",
                "message": "Trained YOLO weights not found. Use training/train_yolo.py to train on the SSS dataset.",
                "model_loaded": False,
                "confidence_threshold": conf,
                "detections": []
            }

        # Run real Ultralytics inference
        results = self.model.predict(
            source=image_input,
            conf=conf,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )

        detections = []
        det_id = 1

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.classes.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "object_id": f"DEBRIS_{det_id:04d}",
                    "class_id": cls_id,
                    "class": cls_name,
                    "confidence": round(score, 3),
                    "bbox": {
                        "x1": round(xyxy[0], 1),
                        "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1),
                        "y2": round(xyxy[3], 1)
                    }
                })
                det_id += 1

        return {
            "status": "success",
            "model_loaded": True,
            "model_path": self.model_path,
            "confidence_threshold": conf,
            "total_detections": len(detections),
            "detections": detections
        }

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Renders visual bounding box overlays with class tags and confidence scores.
        """
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        default_colors = {
            "fishing_net": (0, 0, 255),          # Red
            "pipeline_or_cable": (255, 165, 0),  # Orange
            "shipwreck_fragment": (0, 140, 255), # Deep Orange
            "engineering_platform": (0, 255, 255),# Yellow
            "riprap_debris": (255, 0, 255)       # Magenta
        }
        colors = color_map or default_colors

        for d in detections:
            bbox = d["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
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
