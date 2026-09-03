"""
Layer 5: Geospatial Association & Geotagging Engine
Implements Module 5 Five-Stage Geotagging Algorithm:
  Stage 1: Ingestion of Raster Metadata & Detections (GeoTIFF, ESRI World Files, Nav Logs)
  Stage 2: Georeferencing Case Classification (Case A: Mosaics; Case B: Raw Navigation; Case C: Unreferenced)
  Stage 3: Deterministic Location Determination:
           - Case A: Affine GeoTransform Matrix (col, row -> X_map, Y_map)
           - Case B: Slant-to-Ground Range Correction + AUV/Towfish Geodesic Projection
  Stage 4: Lat/Lon Geodetic Conversion to WGS84 (EPSG:4326) via PyProj
  Stage 5: Final Geotag Assembly & Multi-Format Export (GeoJSON, CSV, JSON)
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import os
import json
import csv
import math
import numpy as np
import pyproj
from pyproj import Transformer, Geod

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


class GeospatialEngine:
    """
    Deterministic location determination and geotagging engine for underwater sonar targets.
    """
    def __init__(self, target_crs: str = "EPSG:4326"):
        self.target_crs = target_crs
        self._transformers = {}
        self.geod = Geod(ellps="WGS84")

    # -----------------------------------------------------------------
    # Stage 1: Ingestion of Raster Metadata
    # -----------------------------------------------------------------
    def read_raster_metadata(self, raster_path: str) -> Dict[str, Any]:
        """
        Reads geospatial metadata from GeoTIFF tags or ESRI World Files (.tfw).
        """
        default_meta = {
            "crs": None,
            "transform": None,
            "width": 0,
            "height": 0,
            "res": (1.0, 1.0),
            "nodata": None,
            "georeferenced": False
        }

        if not os.path.exists(raster_path):
            return default_meta

        # 1. Try reading via tifffile (fast, lightweight, pure python)
        if TIFFFILE_AVAILABLE and raster_path.lower().endswith((".tif", ".tiff")):
            try:
                with tifffile.TiffFile(raster_path) as tif:
                    page = tif.pages[0]
                    w = page.imagewidth
                    h = page.imagelength

                    tags = {t.name: t.value for t in page.tags}
                    pixel_scale = tags.get("ModelPixelScaleTag")
                    tiepoint = tags.get("ModelTiepointTag")
                    geokey_dir = tags.get("GeoKeyDirectoryTag")

                    crs_str = None
                    transform = None
                    res = (1.0, 1.0)

                    # Extract CRS from GeoKeyDirectory (e.g. ProjectedCSTypeGeoKey 3072)
                    if geokey_dir:
                        # GeoKeyDirectory format: [Header (4 ints), Key1 (4 ints), Key2...]
                        for i in range(4, len(geokey_dir), 4):
                            key_id = geokey_dir[i]
                            val = geokey_dir[i + 3]
                            if key_id in (3072, 2048):  # ProjectedCSTypeGeoKey or GeographicTypeGeoKey
                                crs_str = f"EPSG:{val}"
                                break

                    # Compute Affine Transform from Tiepoint and Pixel Scale
                    if tiepoint and pixel_scale:
                        # Tiepoint: (i, j, k, x, y, z)
                        # PixelScale: (scale_x, scale_y, scale_z)
                        i, j, k, x0, y0, z0 = tiepoint[:6]
                        dx = float(pixel_scale[0])
                        dy = -float(pixel_scale[1])  # Raster Y goes downward
                        x_origin = x0 - (i * dx)
                        y_origin = y0 - (j * dy)
                        transform = [x_origin, dx, 0.0, y_origin, 0.0, dy]
                        res = (abs(dx), abs(dy))

                    if crs_str or transform:
                        return {
                            "crs": crs_str,
                            "transform": transform,
                            "width": w,
                            "height": h,
                            "res": res,
                            "nodata": None,
                            "georeferenced": True
                        }
            except Exception as e:
                pass

        # 2. Check for ESRI World File (.tfw, .jgw, .pgw)
        base, ext = os.path.splitext(raster_path)
        world_ext = ext[0] + ext[1] + "w"  # e.g. .tif -> .tfw
        world_path = f"{base}{world_ext}"

        if os.path.exists(world_path):
            try:
                with open(world_path, "r") as wf:
                    lines = [float(l.strip()) for l in wf if l.strip()]
                    if len(lines) >= 6:
                        # World file format: [A (dx), D (rot_y), B (rot_x), E (dy), C (x0), F (y0)]
                        dx, rot_y, rot_x, dy, x0, y0 = lines[:6]
                        transform = [x0, dx, rot_x, y0, rot_y, dy]
                        res = (abs(dx), abs(dy))
                        return {
                            "crs": "EPSG:32619",  # Default UTM projection for DH_NOAA Boston Harbor
                            "transform": transform,
                            "width": 1000,
                            "height": 1000,
                            "res": res,
                            "nodata": None,
                            "georeferenced": True
                        }
            except Exception:
                pass

        # 3. Unreferenced image fallback
        return default_meta

    # -----------------------------------------------------------------
    # Stage 2: Georeferencing Case Classification
    # -----------------------------------------------------------------
    def classify_georef_case(
        self,
        raster_meta: Optional[Dict[str, Any]] = None,
        nav_log: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Classifies input into:
          - Case A: Valid georeferenced mosaic / GeoTIFF (Direct Affine matrix)
          - Case B: Raw acoustic waterfall with navigation log (Dead-reckoning projection)
          - Case C: Unreferenced image chip (No spatial reference available)
        """
        has_crs = bool(raster_meta and raster_meta.get("crs"))
        has_transform = bool(raster_meta and raster_meta.get("transform"))

        if has_crs and has_transform:
            return "A"
        elif nav_log and ("latitude" in nav_log and "longitude" in nav_log and "heading" in nav_log):
            return "B"
        else:
            return "C"

    def get_object_center(self, bbox: Dict[str, float]) -> Tuple[float, float]:
        """Calculates pixel centroid (col, row) of bounding box."""
        col = (bbox.get("x1", 0.0) + bbox.get("x2", 0.0)) / 2.0
        row = (bbox.get("y1", 0.0) + bbox.get("y2", 0.0)) / 2.0
        return (col, row)

    # -----------------------------------------------------------------
    # Stage 3a: Case A Direct Affine Transformation
    # -----------------------------------------------------------------
    def locate_case_a(
        self,
        pixel_center: Tuple[float, float],
        raster_meta: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Applies forward affine matrix:
          X_map = c0 + col * c1 + row * c2
          Y_map = c3 + col * c4 + row * c5
        """
        col, row = pixel_center
        transform = raster_meta.get("transform")
        if transform and isinstance(transform, (list, tuple)) and len(transform) >= 6:
            c0, c1, c2, c3, c4, c5 = transform[:6]
            x_map = c0 + col * c1 + row * c2
            y_map = c3 + col * c4 + row * c5
            return (round(x_map, 3), round(y_map, 3))
        return (col, row)

    # -----------------------------------------------------------------
    # Stage 3b: Case B Navigation Geometry & Slant-to-Ground Projection
    # -----------------------------------------------------------------
    def locate_case_b(
        self,
        pixel_center: Tuple[float, float],
        waterfall_dims: Tuple[int, int],
        nav_log: Dict[str, Any],
        slant_range_m: float = 75.0,
        altitude_m: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Performs Case B Dead-Reckoning Sonar Navigation Georeferencing:
          1. Identifies nadir track line (center column of waterfall scan).
          2. Computes slant-range to target based on across-track pixel offset.
          3. Applies slant-range to ground-range projection: R_ground = sqrt(R_slant^2 - H^2).
          4. Projects forward along towfish heading +/- 90 degrees using WGS84 Geodesic math.
        Returns:
          (latitude, longitude, position_uncertainty_m)
        """
        towfish_lat = float(nav_log.get("latitude", 0.0))
        towfish_lon = float(nav_log.get("longitude", 0.0))
        heading_deg = float(nav_log.get("heading", 0.0))
        h_alt = float(altitude_m or nav_log.get("altitude_m", 12.0))

        col, row = pixel_center
        h_img, w_img = waterfall_dims
        nadir_col = w_img / 2.0

        # Across-track pixel offset from nadir (negative = Port, positive = Starboard)
        offset_px = col - nadir_col
        max_half_width_px = w_img / 2.0

        # Slant range in meters
        r_slant = (abs(offset_px) / max(1.0, max_half_width_px)) * slant_range_m

        # Ground range projection (Pythagorean theorem)
        if r_slant >= h_alt:
            r_ground = math.sqrt(r_slant ** 2 - h_alt ** 2)
        else:
            # Inside blind nadir altitude zone
            r_ground = r_slant

        # Bearing: Port is Heading - 90 deg; Starboard is Heading + 90 deg
        if offset_px >= 0:
            target_bearing = (heading_deg + 90.0) % 360.0
        else:
            target_bearing = (heading_deg - 90.0) % 360.0

        # Geodesic forward projection (Vincenty/Karney WGS84)
        target_lon, target_lat, _ = self.geod.fwd(
            lons=towfish_lon,
            lats=towfish_lat,
            az=target_bearing,
            dist=r_ground
        )

        # Case B uncertainty budget: GPS accuracy (3m) + towfish layback/yaw (5m)
        uncertainty_m = 7.5

        return (round(float(target_lat), 6), round(float(target_lon), 6), uncertainty_m)

    # -----------------------------------------------------------------
    # Stage 4: Geodetic Datum Conversion to WGS84 (EPSG:4326)
    # -----------------------------------------------------------------
    def to_lat_lon(
        self,
        x_map: float,
        y_map: float,
        source_crs: Optional[str]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Converts projected grid coordinates (e.g. NAD83 UTM 18N) to geodetic WGS84 (lat, lon).
        """
        if not source_crs:
            return None, None

        if source_crs not in self._transformers:
            try:
                self._transformers[source_crs] = Transformer.from_crs(
                    source_crs,
                    self.target_crs,
                    always_xy=True
                )
            except Exception as e:
                print(f"[GeospatialEngine] Warning: Could not create transformer for {source_crs}: {e}")
                return None, None

        transformer = self._transformers[source_crs]
        try:
            lon, lat = transformer.transform(x_map, y_map)
            return round(float(lat), 6), round(float(lon), 6)
        except Exception:
            return None, None

    # -----------------------------------------------------------------
    # Stage 5: Final Geotag Assembly & Report Export
    # -----------------------------------------------------------------
    def create_object_record(
        self,
        detection: Dict[str, Any],
        lat: Optional[float],
        lon: Optional[float],
        length_m: Optional[float],
        width_m: Optional[float],
        case: str,
        uncertainty_m: float
    ) -> Dict[str, Any]:
        """
        Assembles standardized target record compliant with hydrographic reporting.
        """
        has_coords = (lat is not None and lon is not None)
        has_dims = (length_m is not None and width_m is not None)

        area_sq_m = round(length_m * width_m, 2) if has_dims else None

        return {
            "object_id": detection.get("object_id", "OBJ_001"),
            "class": detection.get("class", "unknown_debris"),
            "confidence": detection.get("confidence", 0.0),
            "coordinates_available": has_coords,
            "latitude": lat,
            "longitude": lon,
            "coordinate_system": "WGS84 (EPSG:4326)" if has_coords else "UNREFERENCED",
            "position_uncertainty_m": uncertainty_m if has_coords else None,
            "georeferencing_case": case,
            "dimensions_available": has_dims,
            "length_m": length_m,
            "width_m": width_m,
            "area_sq_m": area_sq_m,
            "pixel_bbox": detection.get("bbox", {}),
            "timestamp": datetime.utcnow().isoformat()
        }

    def export_geojson(self, records: List[Dict[str, Any]], output_path: str):
        """
        Exports detections as standard GeoJSON FeatureCollection for GIS visualization (Leaflet, QGIS).
        """
        features = []
        for r in records:
            lat = r.get("latitude")
            lon = r.get("longitude")
            if lat is not None and lon is not None:
                properties = {k: v for k, v in r.items() if k not in ("latitude", "longitude")}
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]  # GeoJSON standard: [Longitude, Latitude]
                    },
                    "properties": properties
                }
                features.append(feat)

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

    def export_csv(self, records: List[Dict[str, Any]], output_path: str):
        """Exports tabular hydrographic CSV."""
        if not records:
            return
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Flatten dictionary values
        flat_records = []
        for r in records:
            item = dict(r)
            if isinstance(item.get("pixel_bbox"), dict):
                item["pixel_bbox"] = json.dumps(item["pixel_bbox"])
            if isinstance(item.get("explanation"), dict):
                item["explanation"] = item["explanation"].get("executive_narrative", "")
            flat_records.append(item)

        fieldnames = list(flat_records[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_records)
