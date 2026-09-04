"""
Unit / Smoke Test for SIH57 Runnable Skeletons
Verifies modular boundaries and agent pipeline without requiring external heavyweight packages.
"""
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.preprocessing.pipeline import SonarPreprocessor
from ai.detection.yolo_detector import YOLODetector
from ai.segmentation.unet_segmenter import UNetSegmenter
from ai.anomaly_detection.autoencoder import AnomalyDetector
from ai.measurement.estimator import DimensionEstimator
from ai.geospatial.geotagger import GeospatialEngine
from agent.orchestrator import SIHPipelineAgent

def test_skeletons_runnable():
    print("Testing Preprocessor...")
    prep = SonarPreprocessor()
    val = prep.validate_image("dummy_nonexistent.tif")
    assert val["valid"] is False

    print("Testing YOLODetector...")
    yolo = YOLODetector()
    det = yolo.detect(None)
    assert det["status"] == "model_unavailable"

    print("Testing UNetSegmenter...")
    unet = UNetSegmenter()
    seg = unet.segment_roi(None)
    assert seg["status"] == "model_unavailable"

    print("Testing DimensionEstimator...")
    measurer = DimensionEstimator()
    dims = measurer.estimate_dimensions({"x1": 100, "y1": 200, "x2": 150, "y2": 260}, raster_res=(1.0, 1.0))
    assert dims["width_m"] == 50.0
    assert dims["length_m"] == 60.0
    assert dims["area_sq_m"] == 3000.0

    print("Testing GeospatialEngine Case A...")
    geo = GeospatialEngine()
    meta = {
        "crs": "EPSG:26918",
        "transform": [587384.0, 1.0, 0.0, 4734001.0, 0.0, -1.0],
        "width": 1000,
        "height": 1000,
        "res": (1.0, 1.0)
    }
    case = geo.classify_georef_case(meta)
    assert case == "A"
    center = geo.get_object_center({"x1": 100, "y1": 200, "x2": 200, "y2": 300})
    x_map, y_map = geo.locate_case_a(center, meta)
    assert x_map == 587384.0 + 150.0 * 1.0
    assert y_map == 4734001.0 - 250.0 * 1.0

    print("Testing Agent Orchestrator...")
    agent = SIHPipelineAgent()
    # Test with non-existent image
    res = agent.analyze_image("test_image.tif")
    assert res["status"] == "rejected"
    print("All module skeletons verified successfully!")

if __name__ == "__main__":
    test_skeletons_runnable()
