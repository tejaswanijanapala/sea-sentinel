"""
Stage 7 Unit Test Suite: Dimension Estimation & Geospatial Analysis
Validates:
  1. Module 5 Georeferencing Case Classification (Case A, B, and C)
  2. Case A Deterministic Affine Transform & PyProj geodetic conversion to WGS84
  3. Case B Sonar Navigation Geometry, Slant-to-Ground range projection, and Geodesic forward bearing
  4. Case C Unreferenced handling (strict data integrity: never fabricates fake coordinates)
  5. Dimension Estimator: Axis-aligned bbox, rotated minimum area rect, and mask contours
  6. Multi-format export: GeoJSON FeatureCollection and Hydrographic CSV
"""

import os
import sys
import tempfile
import json
import csv
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.geospatial.geotagger import GeospatialEngine
from ai.measurement.estimator import DimensionEstimator


def test_geotagger_case_classification():
    """Verify Case A, Case B, and Case C classification logic."""
    engine = GeospatialEngine()

    # Case A: Has both CRS and transform
    meta_a = {
        "crs": "EPSG:26918",
        "transform": [598655.0, 1.0, 0.0, 4733469.0, 0.0, -1.0]
    }
    assert engine.classify_georef_case(meta_a) == "A"

    # Case B: No transform/crs, but navigation log present
    nav_b = {"latitude": 13.0827, "longitude": 80.2707, "heading": 45.0}
    assert engine.classify_georef_case(None, nav_log=nav_b) == "B"

    # Case C: Neither present
    assert engine.classify_georef_case(None, None) == "C"
    assert engine.classify_georef_case({"crs": None, "transform": None}) == "C"


def test_case_a_affine_and_pyproj_transformation():
    """Verify Case A affine transform and geodetic conversion using real NOAA Hudson River parameters."""
    engine = GeospatialEngine()

    meta = {
        "crs": "EPSG:26918",  # NAD83 UTM Zone 18N
        "transform": [598655.0, 1.0, 0.0, 4733469.29, 0.0, -1.0],
        "res": (1.0, 1.0)
    }

    # Target pixel centroid
    pixel_center = (100.0, 100.0)

    # Stage 3a: Affine transform
    x_map, y_map = engine.locate_case_a(pixel_center, meta)
    assert x_map == 598755.0
    assert y_map == 4733369.29

    # Stage 4: Geodetic datum conversion via PyProj to WGS84
    lat, lon = engine.to_lat_lon(x_map, y_map, meta["crs"])
    assert lat is not None and lon is not None
    assert 42.0 <= lat <= 43.5, f"Expected Hudson River latitude ~42.7 deg, got {lat}"
    assert -74.5 <= lon <= -73.0, f"Expected Hudson River longitude ~-73.8 deg, got {lon}"


def test_case_b_navigation_geometry():
    """Verify Case B slant-to-ground range correction and geodesic projection."""
    engine = GeospatialEngine()

    # Vessel at Chennai coast, heading directly North (0 degrees)
    nav_log = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "heading": 0.0,
        "altitude_m": 10.0
    }
    waterfall_dims = (1000, 2000)  # Nadir at col 1000

    # Target on Starboard side (col 1500, +500px from nadir)
    pixel_center = (1500.0, 500.0)

    lat, lon, uncertainty = engine.locate_case_b(
        pixel_center=pixel_center,
        waterfall_dims=waterfall_dims,
        nav_log=nav_log,
        slant_range_m=100.0,
        altitude_m=10.0
    )

    assert lat is not None and lon is not None
    assert uncertainty >= 5.0
    # Heading is North, Starboard is East -> Latitude should stay ~13.0827, Longitude should be East (> 80.2707)
    assert abs(lat - 13.0827) < 0.001
    assert lon > 80.2707


def test_case_c_unreferenced_no_fabrication():
    """Verify Case C strictly reports unreferenced without fabricating fake coordinates."""
    engine = GeospatialEngine()

    det = {"object_id": "OBJ_CHIP_01", "class": "fishing_net", "confidence": 0.88, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 60}}
    record = engine.create_object_record(
        detection=det,
        lat=None,
        lon=None,
        length_m=None,
        width_m=None,
        case="C",
        uncertainty_m=0.0
    )

    assert record["coordinates_available"] is False
    assert record["latitude"] is None
    assert record["longitude"] is None
    assert record["dimensions_available"] is False
    assert record["coordinate_system"] == "UNREFERENCED"


def test_dimension_estimator_axis_aligned():
    """Verify metric calculations on bounding box."""
    estimator = DimensionEstimator()

    bbox = {"x1": 20, "y1": 30, "x2": 60, "y2": 90}
    # 1. With authentic resolution (0.5 m/px)
    dims = estimator.estimate_dimensions(bbox, raster_res=(0.5, 0.5))
    assert dims["dimensions_metric"] is True
    assert dims["width_m"] == 20.0  # (60 - 20) * 0.5
    assert dims["length_m"] == 30.0  # (90 - 30) * 0.5
    assert dims["area_sq_m"] == 600.0
    assert dims["aspect_ratio"] == 1.5

    # 2. Without resolution (missing metadata)
    dims_none = estimator.estimate_dimensions(bbox, raster_res=None)
    assert dims_none["dimensions_metric"] is False
    assert dims_none["length_m"] is None
    assert dims_none["width_m"] is None
    assert "unavailable" in dims_none["message"].lower()


def test_dimension_estimator_rotated_mask():
    """Verify rotated minimum bounding rectangle on oriented object (e.g. diagonal pipeline)."""
    estimator = DimensionEstimator()

    # Create a 200x200 canvas with a 45-degree tilted bar
    canvas = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(canvas, (40, 40), (140, 140), 255, thickness=12)

    res = estimator.estimate_mask_dimensions(canvas, raster_res=(0.25, 0.25))
    assert res["dimensions_metric"] is True
    assert res["dimension_type"] == "oriented_contour"
    assert res["length_m"] > res["width_m"]
    assert res["area_sq_m"] > 0
    assert res["orientation_deg"] != 0.0


def test_geojson_and_csv_export():
    """Verify GeoJSON and CSV export validity."""
    engine = GeospatialEngine()

    records = [
        {
            "object_id": "TGT_001",
            "class": "fishing_net",
            "confidence": 0.85,
            "latitude": 42.7474,
            "longitude": -73.7945,
            "length_m": 15.2,
            "width_m": 6.4,
            "area_sq_m": 97.28,
            "georeferencing_case": "A",
            "position_uncertainty_m": 1.5,
            "pixel_bbox": {"x1": 50, "y1": 50, "x2": 120, "y2": 110}
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        geojson_file = os.path.join(tmpdir, "test_out.geojson")
        csv_file = os.path.join(tmpdir, "test_out.csv")

        engine.export_geojson(records, geojson_file)
        engine.export_csv(records, csv_file)

        # Validate GeoJSON
        assert os.path.exists(geojson_file)
        with open(geojson_file, "r") as f:
            data = json.load(f)
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 1
            feat = data["features"][0]
            assert feat["geometry"]["coordinates"] == [-73.7945, 42.7474]
            assert feat["properties"]["object_id"] == "TGT_001"

        # Validate CSV
        assert os.path.exists(csv_file)
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["object_id"] == "TGT_001"


if __name__ == "__main__":
    print("Running Stage 7 Unit Tests...")
    test_geotagger_case_classification()
    print("  [PASSED] test_geotagger_case_classification")
    test_case_a_affine_and_pyproj_transformation()
    print("  [PASSED] test_case_a_affine_and_pyproj_transformation")
    test_case_b_navigation_geometry()
    print("  [PASSED] test_case_b_navigation_geometry")
    test_case_c_unreferenced_no_fabrication()
    print("  [PASSED] test_case_c_unreferenced_no_fabrication")
    test_dimension_estimator_axis_aligned()
    print("  [PASSED] test_dimension_estimator_axis_aligned")
    test_dimension_estimator_rotated_mask()
    print("  [PASSED] test_dimension_estimator_rotated_mask")
    test_geojson_and_csv_export()
    print("  [PASSED] test_geojson_and_csv_export")
    print("All Stage 7 unit tests executed successfully!")
