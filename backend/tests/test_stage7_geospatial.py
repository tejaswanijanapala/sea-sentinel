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
from shared.utils.logger import get_logger
logger = get_logger(__name__)




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


def test_noaa_h11584_georeferencing():
    """Verify NOAA Survey H11584 user-defined GeoKey 32767 is decoded to UTM 16N (EPSG:32616) in Gulf of Mexico."""
    engine = GeospatialEngine()

    h11584_path = os.path.join(PROJECT_ROOT, "outputs", "uploads", "264a97dc_H11584_SSSAB_1m_445kHz_1of2.tif")
    if os.path.exists(h11584_path):
        meta = engine.read_raster_metadata(h11584_path)
        assert meta["georeferenced"] is True
        assert meta["crs"] == "EPSG:32616", f"Expected EPSG:32616 for NOAA H11584, got {meta['crs']}"
        assert meta["res"] == (1.0, 1.0)
        assert "H11584" in meta.get("dataset_profile", "")

        # Origin tiepoint check (Mobile Bay / Mississippi Sound Gulf waters)
        x_map, y_map = engine.locate_case_a((0.0, 0.0), meta)
        lat, lon = engine.to_lat_lon(x_map, y_map, meta["crs"])
        assert lat is not None and lon is not None
        assert 30.10 <= lat <= 30.25, f"Expected Gulf of Mexico latitude ~30.19 deg, got {lat}"
        assert -87.95 <= lon <= -87.80, f"Expected Gulf of Mexico longitude ~-87.88 deg, got {lon}"


def test_usgs_14bim05_georeferencing_parameters():
    """Verify USGS DS 1005 (14BIM05 SSS 50cm) parameters transform to Breton Island, Louisiana."""
    engine = GeospatialEngine()

    # USGS DS 1005 Tile 1 authentic parameters
    meta_usgs = {
        "crs": "EPSG:32616",  # WGS 84 / UTM Zone 16N
        "transform": [284946.0, 0.50, 0.0, 3259333.473, 0.0, -0.50],
        "res": (0.50, 0.50),
        "width": 10219,
        "height": 8875,
        "georeferenced": True
    }

    # Transform origin to WGS84
    x_map, y_map = engine.locate_case_a((0.0, 0.0), meta_usgs)
    lat, lon = engine.to_lat_lon(x_map, y_map, meta_usgs["crs"])
    assert lat is not None and lon is not None
    assert 29.35 <= lat <= 29.50, f"Expected Breton Island LA latitude ~29.44 deg, got {lat}"
    assert -89.30 <= lon <= -89.10, f"Expected Breton Island LA longitude ~-89.22 deg, got {lon}"


def test_zenodo_unreferenced_data_integrity():
    """Verify Zenodo 20048164 image chips are classified as Case C with coordinates withheld."""
    engine = GeospatialEngine()

    # Use existing sample chip
    chip_path = os.path.join(PROJECT_ROOT, "datasets", "processed", "yolo_dataset", "images", "train", "quanzhou_HN_001.jpg")
    if os.path.exists(chip_path):
        meta = engine.read_raster_metadata(chip_path)
        assert meta["georeferenced"] is False
        assert meta["crs"] is None
        assert meta["transform"] is None
        assert "China Offshore" in meta.get("dataset_profile", "")

        case = engine.classify_georef_case(meta)
        assert case == "C", f"Expected Case C for Zenodo chip, got {case}"


def test_sidecar_nav_log_ingestion():
    """Verify sidecar navigation JSON file is automatically detected and parsed."""
    engine = GeospatialEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_img = os.path.join(tmpdir, "sonar_run_01.png")
        fake_nav = os.path.join(tmpdir, "sonar_run_01.nav.json")
        with open(fake_img, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

        nav_content = {
            "latitude": 29.4450,
            "longitude": -89.2172,
            "heading": 135.0,
            "altitude_m": 12.0,
            "slant_range_m": 75.0
        }
        with open(fake_nav, "w", encoding="utf-8") as f:
            json.dump(nav_content, f)

        meta = engine.read_raster_metadata(fake_img)
        assert meta["georeferenced"] is True
        assert meta["nav_log"] is not None
        assert meta["nav_log"]["latitude"] == 29.4450

        case = engine.classify_georef_case(meta)
        assert case == "B"


if __name__ == "__main__":
    logger.info("Running Stage 7 Unit Tests...")
    test_geotagger_case_classification()
    logger.info("  [PASSED] test_geotagger_case_classification")
    test_case_a_affine_and_pyproj_transformation()
    logger.info("  [PASSED] test_case_a_affine_and_pyproj_transformation")
    test_case_b_navigation_geometry()
    logger.info("  [PASSED] test_case_b_navigation_geometry")
    test_case_c_unreferenced_no_fabrication()
    logger.info("  [PASSED] test_case_c_unreferenced_no_fabrication")
    test_dimension_estimator_axis_aligned()
    logger.info("  [PASSED] test_dimension_estimator_axis_aligned")
    test_dimension_estimator_rotated_mask()
    logger.info("  [PASSED] test_dimension_estimator_rotated_mask")
    test_geojson_and_csv_export()
    logger.info("  [PASSED] test_geojson_and_csv_export")
    test_noaa_h11584_georeferencing()
    logger.info("  [PASSED] test_noaa_h11584_georeferencing")
    test_usgs_14bim05_georeferencing_parameters()
    logger.info("  [PASSED] test_usgs_14bim05_georeferencing_parameters")
    test_zenodo_unreferenced_data_integrity()
    logger.info("  [PASSED] test_zenodo_unreferenced_data_integrity")
    test_sidecar_nav_log_ingestion()
    logger.info("  [PASSED] test_sidecar_nav_log_ingestion")
    logger.info("All Stage 7 unit tests executed successfully!")
