import os

filepath = 'backend/ai/detection/yolo_detector.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace detect()
start_detect = content.find('    def detect(')
end_detect = content.find('    def detect_batch(')

new_detect = '''    def detect(
        self,
        image_input: Union[str, np.ndarray],
        conf_override: Optional[float] = None,
        tile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        conf = conf_override if conf_override is not None else self.conf_thresh

        if not self.is_model_loaded or self.model is None:
            return {
                "status": "model_unavailable",
                "message": "Trained YOLO weights not found. Use training/train_yolo.py to train on the SSS dataset.",
                "model_loaded": False,
                "confidence_threshold": conf,
                "detections": [],
                "inference_time_ms": 0.0
            }

        detections = []
        try:
            if isinstance(image_input, str):
                img_cv = cv2.imread(image_input)
            else:
                img_cv = image_input

            # Run YOLO inference
            results = self.model(img_cv, conf=conf, verbose=False)
            
            det_id = 1
            for result in results:
                boxes = result.boxes
                if boxes is None: continue
                
                for box in boxes:
                    b = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    bw, bh = x2 - x1, y2 - y1
                    conf_val = float(box.conf[0].cpu().numpy())
                    cls_idx = int(box.cls[0].cpu().numpy())
                    cls_name = self.classes.get(cls_idx, "unknown")
                    cx, cy = round(x1 + bw / 2.0, 1), round(y1 + bh / 2.0, 1)

                    det_record = {
                        "object_id": f"YOLO_{det_id:03d}" if not tile_id else f"{tile_id}_YOLO_{det_id:03d}",
                        "source": "yolo",
                        "model": "YOLOv11",
                        "class": cls_name,
                        "class_id": cls_idx,
                        "confidence": round(conf_val, 3),
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "width": float(bw),
                        "height": float(bh),
                        "center": [cx, cy],
                        "centroid": [cx, cy]
                    }
                    if tile_id:
                        det_record["tile_id"] = tile_id

                    detections.append(det_record)
                    det_id += 1
                    
        except Exception as e:
            return {
                "status": "error",
                "message": f"YOLO inference failed: {str(e)}",
                "model_loaded": self.is_model_loaded,
                "confidence_threshold": conf,
                "detections": [],
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2)
            }

        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "status": "success",
            "model_loaded": True,
            "confidence_threshold": conf,
            "detections": detections,
            "total_detections": len(detections),
            "inference_time_ms": inference_time_ms
        }

'''

# Replace detect_batch()
start_batch = content.find('    def detect_batch(')
end_batch = content.find('    def draw_detections(')

new_batch = '''    def detect_batch(
        self,
        image_inputs: List[np.ndarray],
        tile_ids: Optional[List[str]] = None,
        conf_override: Optional[float] = None,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        if not image_inputs:
            return []

        t0 = time.perf_counter()
        conf = conf_override if conf_override is not None else self.conf_thresh

        if not self.is_model_loaded or self.model is None:
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
            # We can process sequentially for simplicity, or use batched YOLO inference
            for i, image_input in enumerate(image_inputs):
                tile_id = tile_ids[i] if tile_ids and i < len(tile_ids) else None
                res = self.detect(image_input, conf_override=conf, tile_id=tile_id)
                res["tile_id"] = tile_id
                batch_outputs.append(res)
        except Exception as e:
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

        return batch_outputs

'''

content = content[:start_detect] + new_detect + new_batch + content[end_batch:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("YOLO fixed")
