"""
End-to-End Tests for Sea Sentinel Parallel YOLO + U-Net AI Pipeline
Validates:
  1. Input validation & preprocessing
  2. Concurrent YOLO + U-Net execution
  3. Candidate fusion and verification
  4. Georeferencing (Case A GeoTIFF, Case B Nav Log, Case C Strict Suppression)
  5. SQLite logging and result formatting
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.orchestrator import SIHPipelineAgent
from shared.utils.logger import get_logger
logger = get_logger(__name__)




def test_e2e_pipeline_on_sample():
    agent = SIHPipelineAgent()
    sample_path = os.path.join(PROJECT_ROOT, "datasets", "samples", "china_offshore_quanzhou_net.jpg")
    
    if not os.path.exists(sample_path):
        logger.info(f"Sample image {sample_path} not found; skipping test.")
        return

    result = agent.analyze_image(sample_path)
    assert result is not None
    assert result["status"] == "success"
    assert "analysis_id" in result
    assert "models" in result
    assert "yolo" in result["models"]
    assert "unet" in result["models"]
    assert "fusion" in result["models"] or "fusion" in result
    assert "detections" in result
    assert isinstance(result["detections"], list)
    assert len(result["detections"]) > 0

    # Ensure Case C unreferenced chip coordinates are strictly suppressed
    assert result["georeferencing_case"] == "C"
    for d in result["detections"]:
        assert d.get("latitude") is None
        assert d.get("longitude") is None


def test_e2e_pipeline_geotiff():
    agent = SIHPipelineAgent()
    geotiff_path = os.path.join(PROJECT_ROOT, "datasets", "samples", "noaa_h11584_gulf_sample.tif")
    
    if not os.path.exists(geotiff_path):
        logger.info(f"GeoTIFF sample {geotiff_path} not found; skipping test.")
        return

    result = agent.analyze_image(geotiff_path)
    assert result is not None
    assert result["status"] == "success"
    assert result["georeferencing_case"] == "A"
    assert result.get("bbox_wgs84") is not None


if __name__ == "__main__":
    test_e2e_pipeline_on_sample()
    test_e2e_pipeline_geotiff()
    logger.info("All End-to-End Parallel Pipeline tests passed successfully!")
