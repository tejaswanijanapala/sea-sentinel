"""
End-to-End API Tests for Sea Sentinel GIS Endpoints (Unittest).
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from fastapi.testclient import TestClient
from backend.app.main import app, local_gis_db


class TestGISAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_gis_map_data_endpoint(self):
        res = self.client.get("/api/gis/map-data")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["offline_ready"])
        self.assertIn("targets", data)
        self.assertIn("clusters", data)
        self.assertIn("survey_tracks", data)
        self.assertIn("survey_coverage", data)
        self.assertIn("statistics", data)

    def test_gis_target_review_and_persistence(self):
        # Insert a target into the database
        t_id = local_gis_db.insert_or_update_target({
            "target_id": "SS-TEST-99",
            "class_name": "ghost_net",
            "latitude": 15.1234,
            "longitude": 73.5678,
            "confidence": 0.88,
            "verification_status": "UNVERIFIED"
        })

        # Perform human review via API
        review_payload = {
            "reviewer_decision": "VERIFIED",
            "correct_class": "fishing_gear",
            "comments": "Inspected by mission operator; verified as lost crab trap.",
            "new_confidence": 0.96,
            "reviewer_name": "Lead Hydrographer"
        }
        res = self.client.post(f"/api/gis/target/{t_id}/review", json=review_payload)
        self.assertEqual(res.status_code, 200)
        r_data = res.json()
        self.assertEqual(r_data["status"], "success")
        self.assertEqual(r_data["target"]["verification_status"], "VERIFIED")
        self.assertEqual(r_data["target"]["class_name"], "fishing_gear")

        # Query target detail endpoint
        detail_res = self.client.get(f"/api/gis/target/{t_id}")
        self.assertEqual(detail_res.status_code, 200)
        detail_data = detail_res.json()
        self.assertGreaterEqual(len(detail_data["reviews"]), 1)

    def test_gis_export_endpoints(self):
        # GeoJSON export
        res_geojson = self.client.get("/api/gis/export?format=geojson&layer=all")
        self.assertEqual(res_geojson.status_code, 200)
        self.assertEqual(res_geojson.headers["content-type"], "application/json")
        gj = res_geojson.json()
        self.assertEqual(gj["type"], "FeatureCollection")

        # CSV export
        res_csv = self.client.get("/api/gis/export?format=csv&layer=all")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers["content-type"])
        self.assertIn("Target ID", res_csv.text)

        # KML export
        res_kml = self.client.get("/api/gis/export?format=kml&layer=all")
        self.assertEqual(res_kml.status_code, 200)
        self.assertIn("<kml", res_kml.text)


if __name__ == "__main__":
    unittest.main()
