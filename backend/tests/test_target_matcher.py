"""
Unit & Integration Tests for Duplicate Target Matching & Spatial Deduplication (Unittest).
"""

import unittest
import tempfile
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from backend.database.local_db import LocalDatabase
from backend.duplicate_detection.spatial_matcher import TargetMatchingService


class TestTargetMatcher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_spatial.db")
        self.db = LocalDatabase(db_path=self.db_path)
        self.matcher = TargetMatchingService(self.db, matching_radius_m=15.0)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_duplicate_target_merging(self):
        # 1. First observation of a ghost net at (12.0000, 80.0000)
        t_id1, is_dup1, target1 = self.matcher.match_or_create_target(
            class_name="ghost_net",
            latitude=12.0000,
            longitude=80.0000,
            confidence=0.88,
            depth=15.0,
            uncertainty_radius_m=4.5
        )
        self.assertFalse(is_dup1)
        self.assertEqual(target1["observation_count"], 1)
        self.assertTrue(target1["target_id"].startswith("SS-"))

        # 2. Second observation of the same target 5 meters away
        t_id2, is_dup2, target2 = self.matcher.match_or_create_target(
            class_name="ghost_net",
            latitude=12.00004,
            longitude=80.00003,
            confidence=0.92,
            depth=15.2,
            uncertainty_radius_m=3.8
        )
        self.assertTrue(is_dup2)
        self.assertEqual(t_id2, t_id1)
        self.assertEqual(target2["observation_count"], 2)
        self.assertEqual(target2["confidence"], 0.92)
        self.assertEqual(target2["uncertainty_radius"], 3.8)

        # 3. Third observation of a completely different target 500 meters away
        t_id3, is_dup3, target3 = self.matcher.match_or_create_target(
            class_name="metal",
            latitude=12.0050,
            longitude=80.0050,
            confidence=0.85
        )
        self.assertFalse(is_dup3)
        self.assertNotEqual(t_id3, t_id1)
        self.assertEqual(target3["observation_count"], 1)

        # Database query should return exactly 2 distinct targets
        all_targets = self.db.get_all_targets()
        self.assertEqual(len(all_targets), 2)


if __name__ == "__main__":
    unittest.main()
