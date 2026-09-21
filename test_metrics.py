import sys
import os
sys.path.insert(0, os.path.abspath('backend'))
from backend.app.main import CACHED_ANALYSES
from backend.evaluation.metrics_engine import get_metrics_engine

try:
    metrics_eng = get_metrics_engine()
    res = metrics_eng.evaluate_image(
        image_path=r'C:\Users\jaish\.gemini\antigravity-ide\scratch\sea-sentinel\backend\tests\test_images\sample_sonar.jpg',
        detections=[],
        segmentation_mask=None
    )
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
