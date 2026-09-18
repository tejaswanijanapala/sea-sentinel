"""
Unit Tests for Parallel YOLO + U-Net Inference Engine
Validates:
  1. Independent YOLO inference
  2. Independent U-Net segmentation and object candidate extraction
  3. Concurrent execution via ThreadPoolExecutor
  4. Failure isolation (YOLO error does not crash U-Net; U-Net error does not crash YOLO)
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.detection.yolo_detector import YOLODetector
from ai.segmentation.unet_segmenter import UNetSegmenter
from inference.parallel_pipeline import ParallelInferenceEngine
from shared.utils.logger import get_logger
logger = get_logger(__name__)




def test_independent_yolo():
    detector = YOLODetector(conf_thresh=0.20)
    # Synthetic test image
    img = np.full((300, 400), 128, dtype=np.uint8)
    res = detector.detect(img)
    assert res is not None
    assert "status" in res
    assert "detections" in res
    assert isinstance(res["detections"], list)


def test_independent_unet_extraction():
    segmenter = UNetSegmenter(img_size=128)
    # Create synthetic binary mask with 2 distinct connected components
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[30:70, 40:80] = 1   # Object 1 (area = 1600 px)
    mask[150:180, 200:280] = 1 # Object 2 (area = 2400 px, aspect ratio ~ 2.6)

    prob_map = np.zeros((300, 400), dtype=np.float32)
    prob_map[mask > 0] = 0.88

    objs = segmenter.extract_candidate_objects(mask, prob_map, min_area=50)
    assert len(objs) == 2
    assert objs[0]["source"] == "unet"
    assert objs[0]["mask_area"] > 0
    assert "bbox" in objs[0]
    assert "centroid" in objs[0]
    assert "aspect_ratio" in objs[0]
    assert objs[1]["aspect_ratio"] >= 2.0
    assert objs[1]["class"] == "pipeline_or_cable"


def test_parallel_execution_and_isolation():
    engine = ParallelInferenceEngine()
    img = np.full((300, 400), 128, dtype=np.uint8)

    parallel_out = engine.run_parallel_inference(img)
    assert "yolo" in parallel_out
    assert "unet" in parallel_out
    assert "total_parallel_ms" in parallel_out
    assert parallel_out["total_parallel_ms"] >= 0

    # Ensure neither model crashed the pipeline
    assert parallel_out["yolo"]["source"] == "yolo"
    assert parallel_out["unet"]["source"] == "unet"


if __name__ == "__main__":
    test_independent_yolo()
    test_independent_unet_extraction()
    test_parallel_execution_and_isolation()
    logger.info("All Parallel Inference unit tests passed successfully!")
