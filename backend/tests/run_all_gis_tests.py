"""
Self-Contained Automated GIS Test Runner for Sea Sentinel (Offline-Native).
Runs all unit and integration tests across Georeferencing, Duplicate Matching,
DBSCAN Clustering, Spatial DB, and FastAPI Endpoints.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from backend.tests.test_georeferencing import (
    test_coordinate_utilities_geodesic_distance_and_bearing,
    test_slant_to_ground_range_georeferencing,
    test_unreferenced_georeferencing_strictly_no_fake_gps
)
from backend.database.local_db import LocalDatabase
from backend.duplicate_detection.spatial_matcher import TargetMatchingService
from backend.ai.geospatial.clustering_service import ClusteringService
import tempfile


def run_target_matcher_tests():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_matcher.db")
    db = LocalDatabase(db_path=db_path)
    matcher = TargetMatchingService(db, matching_radius_m=15.0)

    # 1. First observation
    t_id1, is_dup1, target1 = matcher.match_or_create_target(
        class_name="ghost_net",
        latitude=12.0000,
        longitude=80.0000,
        confidence=0.88,
        depth=15.0,
        uncertainty_radius_m=4.5
    )
    assert is_dup1 is False, "First target should not be duplicate"
    assert target1["observation_count"] == 1

    # 2. Second nearby observation
    t_id2, is_dup2, target2 = matcher.match_or_create_target(
        class_name="ghost_net",
        latitude=12.00004,
        longitude=80.00003,
        confidence=0.92,
        depth=15.2,
        uncertainty_radius_m=3.8
    )
    assert is_dup2 is True, "Second nearby observation should match existing target"
    assert t_id2 == t_id1
    assert target2["observation_count"] == 2

    # 3. Third distant observation
    t_id3, is_dup3, target3 = matcher.match_or_create_target(
        class_name="metal",
        latitude=12.0050,
        longitude=80.0050,
        confidence=0.85
    )
    assert is_dup3 is False, "Distant target should be separate"
    assert t_id3 != t_id1

    targets = db.get_all_targets()
    assert len(targets) == 2, f"Expected 2 targets, got {len(targets)}"
    print("  [PASS] test_duplicate_target_merging")


def run_clustering_tests():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_clustering.db")
    db = LocalDatabase(db_path=db_path)

    db.insert_or_update_target({"target_id": "SS-0001", "class_name": "ghost_net", "latitude": 13.08270, "longitude": 80.27070, "confidence": 0.90})
    db.insert_or_update_target({"target_id": "SS-0002", "class_name": "ghost_net", "latitude": 13.08275, "longitude": 80.27072, "confidence": 0.88})
    db.insert_or_update_target({"target_id": "SS-0003", "class_name": "fishing_gear", "latitude": 13.08268, "longitude": 80.27075, "confidence": 0.91})
    db.insert_or_update_target({"target_id": "SS-0004", "class_name": "metal", "latitude": 13.12000, "longitude": 80.32000, "confidence": 0.85})

    cluster_service = ClusteringService(db, epsilon_meters=50.0, min_samples=2)
    clusters = cluster_service.recalculate_clusters()

    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert clusters[0]["target_count"] == 3
    t4 = db.get_target_by_id("SS-0004")
    assert t4["cluster_id"] is None
    print("  [PASS] test_dbscan_clustering")


def run_api_tests():
    from fastapi.testclient import TestClient
    from backend.app.main import app, local_gis_db
    client = TestClient(app)

    # 1. Map data endpoint
    res = client.get("/api/gis/map-data")
    assert res.status_code == 200
    d = res.json()
    assert d["status"] == "success"
    assert d["offline_ready"] is True
    print("  [PASS] test_gis_map_data_endpoint")

    # 2. Target review endpoint
    t_id = local_gis_db.insert_or_update_target({
        "target_id": "SS-TEST-01",
        "class_name": "ghost_net",
        "latitude": 15.1234,
        "longitude": 73.5678,
        "confidence": 0.88,
        "verification_status": "UNVERIFIED"
    })
    r_res = client.post(f"/api/gis/target/{t_id}/review", json={
        "reviewer_decision": "VERIFIED",
        "correct_class": "fishing_gear",
        "comments": "Verified by mission operator"
    })
    assert r_res.status_code == 200
    assert r_res.json()["target"]["verification_status"] == "VERIFIED"
    print("  [PASS] test_gis_target_review_and_persistence")

    # 3. Export endpoint
    exp_res = client.get("/api/gis/export?format=geojson&layer=all")
    assert exp_res.status_code == 200
    assert exp_res.json()["type"] == "FeatureCollection"
    print("  [PASS] test_gis_export_endpoint")


if __name__ == "__main__":
    print("================================================================")
    print(" SEA SENTINEL OFFLINE GIS & GEOSPATIAL INTELLIGENCE TEST SUITE ")
    print("================================================================")

    print("\n[1/4] Running Georeferencing Tests...")
    test_coordinate_utilities_geodesic_distance_and_bearing()
    print("  [PASS] test_coordinate_utilities_geodesic_distance_and_bearing")
    test_slant_to_ground_range_georeferencing()
    print("  [PASS] test_slant_to_ground_range_georeferencing")
    test_unreferenced_georeferencing_strictly_no_fake_gps()
    print("  [PASS] test_unreferenced_georeferencing_strictly_no_fake_gps")

    print("\n[2/4] Running Duplicate Matching & Deduplication Tests...")
    run_target_matcher_tests()

    print("\n[3/4] Running DBSCAN Spatial Clustering Tests...")
    run_clustering_tests()

    print("\n[4/4] Running FastAPI Endpoints & Persistence Tests...")
    run_api_tests()

    print("\n================================================================")
    print(" ALL 8 TEST SUITES PASSED (100% SUCCESSFUL) ")
    print("================================================================")
