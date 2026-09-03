"""
Unit Tests for Stage 2 Preprocessing Pipeline
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.preprocessing.pipeline import SonarPreprocessor
from ai.preprocessing.batch_processor import BatchPreprocessor

def test_normalization():
    proc = SonarPreprocessor()
    test_img = np.array([[10, 50], [100, 200]], dtype=np.uint8)

    # Test [0, 1] float
    norm_f = proc.normalize(test_img, method="min_max")
    assert np.min(norm_f) == 0.0
    assert np.max(norm_f) == 1.0

    # Test [0, 255] uint8
    norm_u = proc.normalize(test_img, method="min_max_uint8")
    assert norm_u.dtype == np.uint8
    assert np.min(norm_u) == 0
    assert np.max(norm_u) == 255

def test_lee_speckle_filter():
    proc = SonarPreprocessor()
    # Create uniform patch with added multiplicative noise
    base = np.full((100, 100), 128.0, dtype=np.float32)
    noise = np.random.normal(1.0, 0.2, (100, 100))
    noisy = np.clip(base * noise, 0, 255).astype(np.uint8)

    denoised = proc.denoise_speckle_lee(noisy, size=5, noise_var=0.2)
    # Variance of the uniform area should be substantially reduced
    assert np.var(denoised[10:90, 10:90]) < np.var(noisy[10:90, 10:90])

def test_clahe_contrast():
    proc = SonarPreprocessor()
    low_contrast = np.random.randint(90, 110, (100, 100), dtype=np.uint8)
    enhanced = proc.enhance_contrast_clahe(low_contrast, clip_limit=3.0, grid_size=(8, 8))
    # Standard deviation (contrast) should increase
    assert np.std(enhanced) >= np.std(low_contrast)

def test_shadow_highlight_extraction():
    proc = SonarPreprocessor()
    # Create synthetic sonar patch: background=100, highlight=240, shadow=10
    synthetic = np.full((64, 64), 100, dtype=np.uint8)
    synthetic[10:25, 10:25] = 240 # Bright debris contact
    synthetic[26:50, 10:25] = 10  # Acoustic shadow trailing it

    res = proc.extract_shadow_highlight_pair(synthetic, shadow_thresh=35, highlight_thresh=195, min_area_px=10)
    assert res["shadow_pixel_count"] > 0
    assert res["highlight_pixel_count"] > 0
    assert len(res["candidate_highlights"]) >= 1

def test_tiling():
    proc = SonarPreprocessor()
    large_mosaic = np.random.randint(20, 200, (1500, 2000), dtype=np.uint8)
    tiles = proc.tile_mosaic(large_mosaic, tile_size=(640, 640), overlap=128)
    assert len(tiles) > 0
    # First tile should start at (0, 0)
    assert tiles[0]["bounds"]["col_start"] == 0
    assert tiles[0]["bounds"]["row_start"] == 0
    assert tiles[0]["patch"].shape == (640, 640)

def test_full_pipeline_process():
    proc = SonarPreprocessor()
    test_img = np.random.randint(10, 220, (128, 128), dtype=np.uint8)
    out = proc.process(test_img)
    assert out["status"] == "success"
    assert out["preprocessed_image"].shape == (128, 128)
    assert "metrics" in out
    assert out["metrics"]["enl_improvement_factor"] > 0

if __name__ == "__main__":
    test_normalization()
    test_lee_speckle_filter()
    test_clahe_contrast()
    test_shadow_highlight_extraction()
    test_tiling()
    test_full_pipeline_process()
    print("All Stage 2 preprocessing unit tests passed successfully!")
