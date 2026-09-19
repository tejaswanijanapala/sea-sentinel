import os

filepath = 'backend/ai/segmentation/unet_segmenter.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix segment_roi()
start_roi = content.find('    def segment_roi(')
end_roi = content.find('    def segment_full_image(')

new_roi = '''    def segment_roi(
        self,
        image_patch: Union[np.ndarray, None],
        threshold: Optional[float] = None,
        offset_xy: Optional[Tuple[int, int]] = (0, 0)
    ) -> Dict[str, Any]:
        if image_patch is None or not isinstance(image_patch, np.ndarray) or image_patch.size == 0:
            return {
                "status": "error",
                "message": "Invalid or empty image patch provided.",
                "mask_available": False,
                "mask": None,
                "polygon": []
            }

        thresh = threshold if threshold is not None else self.confidence_threshold
        h_orig, w_orig = image_patch.shape[:2]
        ox, oy = offset_xy if offset_xy else (0, 0)

        # Convert to grayscale float [0, 1]
        if image_patch.ndim == 3:
            gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_patch.copy()

        binary_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        mean_conf = 0.0

        if self.is_model_loaded and self.model is not None:
            try:
                resized = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                norm_img = resized.astype(np.float32) / 255.0
                tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(self.device)

                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

                orig_prob = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                binary_mask = (orig_prob >= thresh).astype(np.uint8)
                if np.any(binary_mask > 0):
                    mean_conf = float(np.mean(orig_prob[binary_mask > 0]))
            except Exception as e:
                binary_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # Morphological smoothing
        if np.any(binary_mask > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Extract contours and convert to global polygon vertices
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_pts = []
        if len(contours) > 0:
            # Sort by contour area and take largest
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, max(1.0, epsilon), True)
            for pt in approx:
                px, py = pt[0]
                polygon_pts.append([round(float(ox + px), 1), round(float(oy + py), 1)])

        total_area = int(np.sum(binary_mask))

        return {
            "status": "success",
            "mask_available": True if total_area > 0 else False,
            "mask": binary_mask if total_area > 0 else None,
            "polygon": polygon_pts,
            "contours_count": len(contours),
            "total_area_px": total_area,
            "mean_confidence": round(mean_conf, 3),
            "model_type": self.model_type
        }

'''

content = content[:start_roi] + new_roi + content[end_roi:]

# 2. Fix segment_full_image()
start_full = content.find('    def segment_full_image(')
end_full = content.find('    def extract_candidate_objects(')

new_full = '''    def segment_full_image(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
        min_area: Optional[int] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        thresh = threshold if threshold is not None else self.confidence_threshold
        min_comp_area = min_area if min_area is not None else self.min_component_area_px

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "status": "error",
                "message": "Invalid or empty image provided to U-Net.",
                "mask_available": False,
                "objects": [],
                "inference_time_ms": 0.0
            }

        h, w = image.shape[:2]

        if not self.is_model_loaded or self.model is None:
            return {
                "status": "model_unavailable",
                "message": "Trained U-Net weights not found. Use training/train_unet.py to generate checkpoints.",
                "mask_available": False,
                "mask": np.zeros((h, w), dtype=np.uint8),
                "probability_map": np.zeros((h, w), dtype=np.float32),
                "objects": [],
                "inference_time_ms": 0.0
            }

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        try:
            # Simple full-image resize approach for real U-Net inference
            resized = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            norm_img = resized.astype(np.float32) / 255.0
            tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Restore original scale
            full_prob = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (full_prob >= thresh).astype(np.uint8)

            if np.any(binary_mask > 0):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            else:
                cleaned_mask = binary_mask

            # Extract independent object candidates
            objects = self.extract_candidate_objects(
                binary_mask=cleaned_mask,
                probability_map=full_prob,
                min_area=min_comp_area,
                max_area_ratio=self.max_component_area_ratio
            )
        except Exception as e:
            logger.error(f"[UNetSegmenter] Full image segmentation error: {e}")
            cleaned_mask = np.zeros((h, w), dtype=np.uint8)
            full_prob = np.zeros((h, w), dtype=np.float32)
            objects = []

        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "status": "success",
            "model_type": self.model_type,
            "mask_available": True if len(objects) > 0 else False,
            "mask": cleaned_mask,
            "probability_map": full_prob,
            "objects": objects,
            "total_objects": len(objects),
            "total_debris_area_px": int(np.sum(cleaned_mask > 0)),
            "inference_time_ms": inference_time_ms
        }

'''

content = content[:start_full] + new_full + content[end_full:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("U-Net fixed")
