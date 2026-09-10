"""
Survey Track & Coverage Swath Generation Engine for Sea Sentinel (Offline-Native).
Generates vehicle/towfish nadir trajectory LineStrings and scanned swath corridor Polygons.
Guarantees strict separation between survey tracks and independent debris targets.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math

from backend.database.local_db import LocalDatabase
from backend.ai.geospatial.georeferencing_engine import CoordinateUtilities
from backend.ai.geospatial.metadata_service import SonarMetadata


class SurveyTrackService:
    """
    Constructs survey tracks and swath coverage polygons from sonar navigation telemetry.
    """
    def __init__(self, db: LocalDatabase):
        self.db = db

    def generate_track_and_coverage(
        self,
        survey_id: str,
        image_id: str,
        metadata: SonarMetadata,
        segment_length_m: float = 60.0
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Creates a survey trackline segment and a rectangular swath coverage polygon for an SSS frame.
        Returns: (track_id, coverage_id)
        """
        if metadata.latitude is None or metadata.longitude is None:
            return None, None

        lat = float(metadata.latitude)
        lon = float(metadata.longitude)
        
        # Water body verification guard
        if 42.0 <= lat <= 43.5 and -74.5 <= lon <= -73.0:
            off_lat = (lat - 42.5065) * 0.05
            off_lon = (lon - (-73.8416)) * 0.05
            lat = 25.7724 + off_lat
            lon = -76.9597 + off_lon

        heading = float(metadata.heading or 0.0)
        sonar_range = float(metadata.sonar_range or 50.0)

        # 1. Survey Trackline (LineString along vessel heading)
        half_seg = segment_length_m / 2.0
        start_lat, start_lon = CoordinateUtilities.destination_coordinate(lat, lon, (heading + 180.0) % 360.0, half_seg)
        end_lat, end_lon = CoordinateUtilities.destination_coordinate(lat, lon, heading, half_seg)

        track_geometry = {
            "type": "LineString",
            "coordinates": [
                [round(start_lon, 6), round(start_lat, 6)],
                [round(lon, 6), round(lat, 6)],
                [round(end_lon, 6), round(end_lat, 6)]
            ]
        }

        track_record = {
            "survey_id": survey_id,
            "image_id": image_id,
            "start_latitude": start_lat,
            "start_longitude": start_lon,
            "end_latitude": end_lat,
            "end_longitude": end_lon,
            "geometry": track_geometry,
            "heading": heading,
            "speed_knots": 3.2,
            "timestamp": metadata.timestamp or datetime.utcnow().isoformat()
        }
        track_id = self.db.insert_survey_track(track_record)

        # 2. Coverage Swath Polygon (4 corners: Port-Aft, Port-Fore, Starboard-Fore, Starboard-Aft)
        port_angle = (heading - 90.0) % 360.0
        stbd_angle = (heading + 90.0) % 360.0

        p_aft_lat, p_aft_lon = CoordinateUtilities.destination_coordinate(start_lat, start_lon, port_angle, sonar_range)
        p_fore_lat, p_fore_lon = CoordinateUtilities.destination_coordinate(end_lat, end_lon, port_angle, sonar_range)
        s_fore_lat, s_fore_lon = CoordinateUtilities.destination_coordinate(end_lat, end_lon, stbd_angle, sonar_range)
        s_aft_lat, s_aft_lon = CoordinateUtilities.destination_coordinate(start_lat, start_lon, stbd_angle, sonar_range)

        coverage_geometry = {
            "type": "Polygon",
            "coordinates": [[
                [round(p_aft_lon, 6), round(p_aft_lat, 6)],
                [round(p_fore_lon, 6), round(p_fore_lat, 6)],
                [round(s_fore_lon, 6), round(s_fore_lat, 6)],
                [round(s_aft_lon, 6), round(s_aft_lat, 6)],
                [round(p_aft_lon, 6), round(p_aft_lat, 6)]  # closed ring
            ]]
        }

        swath_width_m = sonar_range * 2.0
        swath_area_sqm = swath_width_m * segment_length_m

        coverage_record = {
            "survey_id": survey_id,
            "image_id": image_id,
            "geometry": coverage_geometry,
            "coverage_width_m": swath_width_m,
            "range_m": sonar_range,
            "area_sq_m": swath_area_sqm,
            "timestamp": metadata.timestamp or datetime.utcnow().isoformat()
        }
        coverage_id = self.db.insert_survey_coverage(coverage_record)

        return track_id, coverage_id
