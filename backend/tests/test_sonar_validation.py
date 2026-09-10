import unittest
import os
import sys
import tempfile
import numpy as np
import cv2

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ai.preprocessing.pipeline import SonarPreprocessor


class TestSonarValidation(unittest.TestCase):
    def setUp(self):
        self.preprocessor = SonarPreprocessor()
        self.temp_dir = tempfile.mkdtemp()

    def test_authentic_sonar_grayscale_accepted(self):
        # Create single-channel acoustic waterfall mock
        sonar_img = np.random.randint(40, 180, (200, 400), dtype=np.uint8)
        sonar_path = os.path.join(self.temp_dir, "test_sonar_scan.png")
        cv2.imwrite(sonar_path, sonar_img)

        res = self.preprocessor.validate_image(sonar_path)
        self.assertTrue(res["valid"])
        self.assertTrue(res["is_sonar"])

    def test_optical_color_photo_rejected(self):
        # Create an optical RGB photo with strong colors (e.g. blue sky, green grass, red car)
        optical_img = np.zeros((200, 300, 3), dtype=np.uint8)
        optical_img[:100, :, 0] = 230  # Blue
        optical_img[100:, :, 1] = 200  # Green
        optical_img[50:150, 100:200, 2] = 240  # Red

        optical_path = os.path.join(self.temp_dir, "optical_holiday_photo.jpg")
        cv2.imwrite(optical_path, optical_img)

        res = self.preprocessor.validate_image(optical_path)
        self.assertFalse(res["valid"])
        self.assertFalse(res["is_sonar"])
        self.assertIn("not an authentic Side-Scan Sonar", res["error"])

    def test_blank_uniform_image_rejected(self):
        # Uniform solid white image
        blank_img = np.full((100, 100), 255, dtype=np.uint8)
        blank_path = os.path.join(self.temp_dir, "blank_document.png")
        cv2.imwrite(blank_path, blank_img)

        res = self.preprocessor.validate_image(blank_path)
        self.assertFalse(res["valid"])
        self.assertFalse(res["is_sonar"])


if __name__ == "__main__":
    unittest.main()
