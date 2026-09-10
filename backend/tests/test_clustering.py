"""
Unit & Integration Tests for DBSCAN Spatial Clustering (Unittest).
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
from backend.ai.geospatial.clustering_service import ClusteringService


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_cluster.db")
        self.db = LocalDatabase(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_dbscan_clustering_dense_group_and_noise(self):
        # Cluster 1: 3 targets within 20m of each other around (13.0827, 80.2707)
        self.db.insert_or_update_target({"target_id": "SS-0001", "class_name": "ghost_net", "latitude": 13.08270, "longitude": 80.27070, "confidence": 0.90})
        self.db.insert_or_update_target({"target_id": "SS-0002", "class_name": "ghost_net", "latitude": 13.08275, "longitude": 80.27072, "confidence": 0.88})
        self.db.insert_or_update_target({"target_id": "SS-0003", "class_name": "fishing_gear", "latitude": 13.08268, "longitude": 80.27075, "confidence": 0.91})

        # Isolated target: 5km away
        self.db.insert_or_update_target({"target_id": "SS-0004", "class_name": "metal", "latitude": 13.12000, "longitude": 80.32000, "confidence": 0.85})

        cluster_service = ClusteringService(self.db, epsilon_meters=50.0, min_samples=2)
        clusters = cluster_service.recalculate_clusters()

        self.assertEqual(len(clusters), 1, f"Expected 1 cluster, got {len(clusters)}")
        c1 = clusters[0]
        self.assertEqual(c1["target_count"], 3)
        self.assertIn(c1["dominant_class"], ["ghost_net", "fishing_gear"])
        self.assertGreater(c1["radius_m"], 0)

        # Target 4 should have cluster_id as None (noise)
        t4 = self.db.get_target_by_id("SS-0004")
        self.assertIsNone(t4["cluster_id"])


if __name__ == "__main__":
    unittest.main()
