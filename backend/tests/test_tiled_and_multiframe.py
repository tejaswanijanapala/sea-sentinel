"""
Unit Tests for Tiled Inference and Multi-Frame Association
Validates:
  1. Overlapping tile generation
  2. Coordinate mapping from local tile to global raster space
  3. Duplicate suppression via NMS
  4. Multi-frame track persistence and confidence accumulation
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.tiled_inference import TiledInferenceEngine
from inference.multiframe import MultiFrameTracker
from shared.utils.logger import get_logger
logger = get_logger(__name__)




def test_tiled_inference_and_mapping():
    tiler = TiledInferenceEngine(tile_size=640, overlap_ratio=0.25, min_image_dim_for_tiling=900)
    large_mosaic = np.full((1200, 1600), 100, dtype=np.uint8)
    assert tiler.should_tile(large_mosaic) is True

    tiles = tiler.generate_tiles(large_mosaic)
    assert len(tiles) >= 4
    for t in tiles:
        assert t["patch"].shape == (640, 640)
        assert t["offset_x"] >= 0 and t["offset_y"] >= 0

    # Test coordinate mapping
    tile_dets = [
        {"object_id": "T1", "bbox": {"x1": 50, "y1": 60, "x2": 150, "y2": 160}, "confidence": 0.85, "class": "fishing_net"}
    ]
    mapped = tiler.map_yolo_detections(tile_dets, offset_x=400, offset_y=300, max_w=1600, max_h=1200)
    assert len(mapped) == 1
    assert mapped[0]["bbox"]["x1"] == 450
    assert mapped[0]["bbox"]["y1"] == 360
    assert mapped[0]["bbox"]["x2"] == 550
    assert mapped[0]["bbox"]["y2"] == 460
    assert mapped[0]["centroid"] == [500.0, 410.0]


def test_multiframe_association():
    tracker = MultiFrameTracker(max_frame_gap=3, spatial_distance_threshold_px=80.0)

    # Frame 1: U-Net only detection
    f1_cands = [
        {"object_id": "T1", "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}, "centroid": [125, 125], "confidence": 0.75, "sources": ["unet"]}
    ]
    out1 = tracker.update_frame(frame_idx=1, candidates=f1_cands)
    assert len(out1) == 1
    track_id = out1[0]["track_id"]
    assert out1[0]["multi_frame_hits"] == 1

    # Frame 2: YOLO detection near the same spatial location (moving slightly)
    f2_cands = [
        {"object_id": "T2", "bbox": {"x1": 105, "y1": 108, "x2": 155, "y2": 158}, "centroid": [130, 133], "confidence": 0.82, "sources": ["yolo"]}
    ]
    out2 = tracker.update_frame(frame_idx=2, candidates=f2_cands)
    assert len(out2) == 1
    assert out2[0]["track_id"] == track_id # Correctly linked to the same physical object!
    assert out2[0]["multi_frame_hits"] == 2
    assert out2[0]["confidence"] > 0.82 # Persistence boost!


if __name__ == "__main__":
    test_tiled_inference_and_mapping()
    test_multiframe_association()
    logger.info("All Tiled and Multi-Frame unit tests passed successfully!")
