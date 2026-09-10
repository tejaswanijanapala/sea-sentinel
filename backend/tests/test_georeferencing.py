"""
Unit & Integration Tests for SSS Georeferencing & Coordinate Transformation Engine.
"""

import math
from backend.ai.geospatial.metadata_service import SonarMetadata, MetadataService
from backend.ai.geospatial.georeferencing_engine import (
    GeoreferencingEngine, SonarConfiguration, CoordinateUtilities
)


def test_coordinate_utilities_geodesic_distance_and_bearing():
    dist = CoordinateUtilities.distance_between_coordinates(0.0, 0.0, 0.0, 1.0)
    assert 111000 <= dist <= 112000, f"Expected approx 111.3km, got {dist}"

    bearing = CoordinateUtilities.bearing_between_coordinates(0.0, 0.0, 0.0, 1.0)
    assert round(bearing, 1) == 90.0

    lat2, lon2 = CoordinateUtilities.destination_coordinate(0.0, 0.0, 90.0, 111319.5)
    assert abs(lat2) < 0.001
    assert abs(lon2 - 1.0) < 0.005


def test_slant_to_ground_range_georeferencing():
    engine = GeoreferencingEngine()
    meta = SonarMetadata(
        filename="test_sonar.png",
        file_path="test_sonar.png",
        latitude=10.0,
        longitude=80.0,
        heading=0.0,  # Heading North
        depth=20.0,
        altitude=5.0,
        sonar_range=50.0,
        metadata_source="TEST",
        metadata_quality="HIGH"
    )

    # Starboard detection (right half of 1000px image: x=750, y=500, w=50, h=50)
    res_stbd = engine.georeference_detection(
        bbox=[725, 475, 50, 50],
        image_shape=(1000, 1000),
        metadata=meta
    )

    assert res_stbd.georeference_quality == "EXACT"
    assert res_stbd.port_starboard == "STARBOARD"
    assert res_stbd.bearing_deg == 90.0  # Heading North + 90 deg = East
    assert res_stbd.longitude > 80.0     # Target is to the East
    assert abs(res_stbd.latitude - 10.0) < 0.0001
    assert 0 < res_stbd.uncertainty_radius_m < 15.0

    # Port detection (left half of 1000px image: x=250, y=500, w=50, h=50)
    res_port = engine.georeference_detection(
        bbox=[225, 475, 50, 50],
        image_shape=(1000, 1000),
        metadata=meta
    )
    assert res_port.port_starboard == "PORT"
    assert res_port.bearing_deg == 270.0 # Heading North - 90 deg = West
    assert res_port.longitude < 80.0     # Target is to the West


def test_unreferenced_georeferencing_strictly_no_fake_gps():
    engine = GeoreferencingEngine()
    meta = SonarMetadata(
        filename="unref_sonar.png",
        file_path="unref_sonar.png",
        latitude=None,
        longitude=None,
        metadata_source="UNAVAILABLE",
        metadata_quality="UNAVAILABLE"
    )

    res = engine.georeference_detection(
        bbox=[100, 100, 50, 50],
        image_shape=(640, 640),
        metadata=meta
    )

    assert res.georeference_quality == "UNREFERENCED"
    assert res.latitude is None
    assert res.longitude is None
    assert "latitude" in res.missing_metadata
    assert "longitude" in res.missing_metadata
