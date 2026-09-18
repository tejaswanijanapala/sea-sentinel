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
import re
import json
import csv
import math
import numpy as np
import pyproj
from pyproj import Transformer, Geod
from shared.utils.logger import get_logger
logger = get_logger(__name__)



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
    # Stage 1: Ingestion of Raster Metadata & Navigation Telemetry
    # -----------------------------------------------------------------
    def read_raster_metadata(self, raster_path: str) -> Dict[str, Any]:
        """
        Reads geospatial metadata from GeoTIFF tags, user-defined GeoKeys,
        ESRI World Files (.tfw), companion PRJ files, or sidecar navigation logs.
        """
        default_meta = {
            "crs": None,
            "transform": None,
            "width": 0,
            "height": 0,
            "res": (1.0, 1.0),
            "nodata": None,
            "georeferenced": False,
            "bbox_wgs84": None,
            "center_wgs84": None,
            "nav_log": None,
            "dataset_profile": None
        }

        if not os.path.exists(raster_path):
            return default_meta

        # Check for sidecar navigation file in same directory
        nav_log = self._find_sidecar_nav_log(raster_path)
        default_meta["nav_log"] = nav_log

        # 1. Try reading via tifffile
        if TIFFFILE_AVAILABLE and raster_path.lower().endswith((".tif", ".tiff")):
            try:
                with tifffile.TiffFile(raster_path) as tif:
                    page = tif.pages[0]
                    w = int(page.imagewidth)
                    h = int(page.imagelength)

                    tags = {t.name: t.value for t in page.tags}
                    pixel_scale = tags.get("ModelPixelScaleTag")
                    tiepoint = tags.get("ModelTiepointTag")
                    geokey_dir = tags.get("GeoKeyDirectoryTag")
                    double_params = tags.get("GeoDoubleParamsTag") or ()
                    ascii_params = str(tags.get("GeoAsciiParamsTag") or "")
                    model_transform = tags.get("ModelTransformationTag")

                    crs_str = None
                    transform = None
                    res = (1.0, 1.0)

                    # A. Compute Affine Transform from ModelTiepointTag + ModelPixelScaleTag
                    if tiepoint and pixel_scale:
                        i, j, k, x0, y0, z0 = tiepoint[:6]
                        dx = float(pixel_scale[0])
                        dy = -float(pixel_scale[1])  # Raster Y progresses downward
                        x_origin = float(x0) - (float(i) * dx)
                        y_origin = float(y0) - (float(j) * dy)
                        transform = [x_origin, dx, 0.0, y_origin, 0.0, dy]
                        res = (abs(dx), abs(dy))
                    elif model_transform and len(model_transform) >= 16:
                        # 4x4 matrix: [R11, R12, R13, Tx, R21, R22, R23, Ty, ...]
                        transform = [
                            float(model_transform[3]), float(model_transform[0]), float(model_transform[1]),
                            float(model_transform[7]), float(model_transform[4]), float(model_transform[5])
                        ]
                        res = (abs(float(model_transform[0])), abs(float(model_transform[5])))

                    # B. Extract CRS from GeoKeyDirectory
                    if geokey_dir:
                        keys = {}
                        for idx in range(4, len(geokey_dir), 4):
                            k_id = geokey_dir[idx]
                            k_loc = geokey_dir[idx + 1]
                            k_count = geokey_dir[idx + 2]
                            k_offset = geokey_dir[idx + 3]
                            if k_loc == 0:
                                keys[k_id] = k_offset
                            elif k_loc == 34736 and double_params:  # GeoDoubleParamsTag
                                if k_offset < len(double_params):
                                    keys[k_id] = double_params[k_offset]
                            elif k_loc == 34737 and ascii_params:  # GeoAsciiParamsTag
                                keys[k_id] = ascii_params[k_offset:k_offset + k_count].strip("|\x00")

                        # Check ProjectedCSTypeGeoKey (3072)
                        proj_code = keys.get(3072)
                        if proj_code and proj_code != 32767:
                            crs_str = f"EPSG:{proj_code}"
                        elif proj_code == 32767:
                            # User-defined projection (e.g. NOAA Survey H11584)
                            # Inspect double parameters for Central Meridian (key 3080/3081)
                            cm = None
                            # In GeoTIFF standard, 3080 or 3081 can be ProjNatOriginLongGeoKey
                            for cand_k in (3080, 3081):
                                if cand_k in keys and isinstance(keys[cand_k], (int, float)):
                                    if -180.0 <= keys[cand_k] <= 180.0 and keys[cand_k] != 0.0:
                                        cm = float(keys[cand_k])
                                        break

                            # If not in keys directly, inspect double_params for longitude meridian
                            if cm is None and double_params:
                                for dp in double_params:
                                    if isinstance(dp, (int, float)) and -180.0 <= dp <= 180.0 and dp not in (0.0, 0.9996, 500000.0, 6378137.0, 6356752.314):
                                        cm = float(dp)
                                        break

                            if cm is not None:
                                # Standard Universal Transverse Mercator (UTM) zone derivation:
                                # Central Meridian = -183 + 6 * Zone => Zone = round((cm + 183) / 6)
                                zone = int(round((cm + 183.0) / 6.0))
                                if 1 <= zone <= 60:
                                    crs_str = f"EPSG:326{zone:02d}"
                                else:
                                    crs_str = f"+proj=tmerc +lat_0=0 +lon_0={cm} +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs"

                        # Check GeographicTypeGeoKey (2048) if projected not found
                        if not crs_str and 2048 in keys and keys[2048] != 32767:
                            crs_str = f"EPSG:{keys[2048]}"

                    # C. Check GeoAsciiParamsTag if CRS is still unresolved
                    if not crs_str and ascii_params:
                        m = re.search(r"UTM zone (\d+)([NS])", ascii_params, re.IGNORECASE)
                        if m:
                            z = int(m.group(1))
                            hemi = m.group(2).upper()
                            if "NAD83" in ascii_params.upper():
                                crs_str = f"EPSG:269{z:02d}"
                            else:
                                crs_str = f"EPSG:326{z:02d}" if hemi == "N" else f"EPSG:327{z:02d}"
                        elif "WGS 84" in ascii_params or "WGS84" in ascii_params:
                            crs_str = "EPSG:4326"

                    # D. Companion .prj file fallback
                    if not crs_str:
                        crs_str = self._read_companion_prj(raster_path)

                    if crs_str and transform:
                        meta_res = {
                            "crs": crs_str,
                            "transform": transform,
                            "width": w,
                            "height": h,
                            "res": res,
                            "nodata": tags.get("GDAL_NODATA"),
                            "georeferenced": True,
                            "nav_log": nav_log,
                            "dataset_profile": self._detect_dataset_profile(raster_path, crs_str, transform)
                        }
                        # Calculate bounding box in WGS84
                        bbox_wgs84, center_wgs84 = self._compute_raster_extents(transform, w, h, crs_str)
                        meta_res["bbox_wgs84"] = bbox_wgs84
                        meta_res["center_wgs84"] = center_wgs84
                        return meta_res

            except Exception as e:
                logger.error(f"[GeospatialEngine] Warning: Exception reading GeoTIFF {raster_path}: {e}")

        # 2. Check for ESRI World File (.tfw, .jgw, .pgw, .wld)
        base, ext = os.path.splitext(raster_path)
        world_candidates = [
            f"{base}.tfw", f"{base}.jgw", f"{base}.pgw", f"{base}.wld",
            f"{base}{ext[0] + ext[1] + 'w' if len(ext) >= 3 else '.tfw'}"
        ]
        world_path = next((p for p in world_candidates if os.path.exists(p)), None)

        if world_path:
            try:
                with open(world_path, "r") as wf:
                    lines = [float(l.strip()) for l in wf if l.strip()]
                    if len(lines) >= 6:
                        # World file format: [A (dx), D (rot_y), B (rot_x), E (dy), C (x0), F (y0)]
                        dx, rot_y, rot_x, dy, x0, y0 = lines[:6]
                        transform = [x0, dx, rot_x, y0, rot_y, dy]
                        res = (abs(dx), abs(dy))

                        # Deduce CRS from .prj or filename
                        crs_str = self._read_companion_prj(raster_path)
                        if not crs_str:
                            crs_str = self._infer_crs_from_path(raster_path)

                        if crs_str:
                            bbox_wgs84, center_wgs84 = self._compute_raster_extents(transform, 1000, 1000, crs_str)
                            return {
                                "crs": crs_str,
                                "transform": transform,
                                "width": 1000,
                                "height": 1000,
                                "res": res,
                                "nodata": None,
                                "georeferenced": True,
                                "bbox_wgs84": bbox_wgs84,
                                "center_wgs84": center_wgs84,
                                "nav_log": nav_log,
                                "dataset_profile": self._detect_dataset_profile(raster_path, crs_str, transform)
                            }
            except Exception as e:
                logger.warning(f"[GeospatialEngine] Warning reading world file {world_path}: {e}")

        # 3. Check for standalone Navigation Log (Case B)
        if nav_log and ("latitude" in nav_log and "longitude" in nav_log and "heading" in nav_log):
            return {
                "crs": "WGS84 Navigation Log",
                "transform": None,
                "width": 1000,
                "height": 1000,
                "res": (1.0, 1.0),
                "nodata": None,
                "georeferenced": True,
                "bbox_wgs84": None,
                "center_wgs84": {"lat": nav_log["latitude"], "lon": nav_log["longitude"]},
                "nav_log": nav_log,
                "dataset_profile": "Case B Towfish Telemetry"
            }

        # 4. Strict Unreferenced Fallback (Case C, e.g. Zenodo 20048164 China Offshore SSS-AI)
        default_meta["dataset_profile"] = self._detect_dataset_profile(raster_path, None, None)
        return default_meta

    def _find_sidecar_nav_log(self, raster_path: str) -> Optional[Dict[str, Any]]:
        """Looks for accompanying sidecar navigation JSON or CSV file."""
        base, _ = os.path.splitext(raster_path)
        dir_name = os.path.dirname(raster_path) or "."
        stem = os.path.basename(base)

        candidates = [
            f"{base}.nav.json",
            f"{base}_nav.json",
            f"{base}.json",
            os.path.join(dir_name, "nav_log.json"),
            os.path.join(dir_name, "navigation.json")
        ]

        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "latitude" in data and "longitude" in data:
                        return data
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "latitude" in data[0]:
                        return data[0]
                except Exception:
                    pass

        # Check for CSV nav log
        csv_candidates = [
            f"{base}.nav.csv",
            f"{base}_nav.csv",
            os.path.join(dir_name, "nav.csv"),
            os.path.join(dir_name, "trackline.csv")
        ]
        for cp in csv_candidates:
            if os.path.exists(cp):
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        row = next(reader, None)
                        if row:
                            lat = float(row.get("latitude") or row.get("lat", 0))
                            lon = float(row.get("longitude") or row.get("lon", 0))
                            heading = float(row.get("heading") or row.get("course", 0))
                            alt = float(row.get("altitude_m") or row.get("altitude", 12.0))
                            return {"latitude": lat, "longitude": lon, "heading": heading, "altitude_m": alt}
                except Exception:
                    pass

        return None

    def _read_companion_prj(self, raster_path: str) -> Optional[str]:
        """Reads companion ESRI .prj file and parses CRS."""
        base, _ = os.path.splitext(raster_path)
        prj_path = f"{base}.prj"
        if os.path.exists(prj_path):
            try:
                with open(prj_path, "r", encoding="utf-8") as f:
                    wkt = f.read().strip()
                if wkt:
                    crs_obj = pyproj.CRS.from_wkt(wkt)
                    epsg = crs_obj.to_epsg()
                    if epsg:
                        return f"EPSG:{epsg}"
                    return crs_obj.to_string()
            except Exception:
                pass
        return None

    def _infer_crs_from_path(self, raster_path: str) -> Optional[str]:
        """Infers standard hydrographic CRS from file name tokens."""
        lower = raster_path.lower()
        if "utm16n" in lower or "utm16" in lower:
            return "EPSG:32616"
        elif "utm18n" in lower or "utm18" in lower:
            return "EPSG:26918"
        elif "utm19n" in lower or "utm19" in lower:
            return "EPSG:32619"
        elif "wgs84" in lower:
            return "EPSG:4326"
        return None

    def _detect_dataset_profile(self, raster_path: str, crs: Optional[str], transform: Optional[List[float]]) -> str:
        """Detects and registers known public dataset source."""
        fname = os.path.basename(raster_path).lower()
        if "h11584" in fname:
            return "NOAA NOS Hydrographic Survey H11584 (Gulf of Mexico, UTM 16N, 1.0m/px)"
        elif "14bim05" in fname:
            return "USGS DS 1005 Barrier Islands Survey (Breton Sound LA, UTM 16N, 0.50m/px)"
        elif "ny_hr09" in fname or "hudson" in fname:
            return "NOAA Hudson River Survey (Albany NY Corridor, NAD83 UTM 18N, 1.0m/px)"
        elif any(k in fname for k in ("quanzhou", "dongying", "china-offshore", "zenodo")):
            return "China Offshore SSS-AI (Zenodo 20048164, Image Chips Only, Unreferenced)"
        elif crs:
            return f"Georeferenced Sonar Mosaic ({crs})"
        else:
            return "Unreferenced Acoustic Chip (Case C)"

    def _compute_raster_extents(
        self,
        transform: List[float],
        width: int,
        height: int,
        source_crs: str
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Calculates geographic bounding box and center in WGS84 for a raster."""
        try:
            x0, dx, rot_x, y0, rot_y, dy = transform[:6]
            corners = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            lats, lons = [], []
            for col, row in corners:
                x_map = x0 + col * dx + row * rot_x
                y_map = y0 + col * rot_y + row * dy
                lat, lon = self.to_lat_lon(x_map, y_map, source_crs)
                if lat is not None and lon is not None:
                    lats.append(lat)
                    lons.append(lon)

            if lats and lons:
                bbox = {
                    "min_lat": round(min(lats), 6),
                    "max_lat": round(max(lats), 6),
                    "min_lon": round(min(lons), 6),
                    "max_lon": round(max(lons), 6)
                }
                center = {
                    "lat": round(sum(lats) / len(lats), 6),
                    "lon": round(sum(lons) / len(lons), 6)
                }
                return bbox, center
        except Exception:
            pass
        return None, None

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

        effective_nav = nav_log or (raster_meta.get("nav_log") if raster_meta else None)
        if effective_nav and ("latitude" in effective_nav and "longitude" in effective_nav and "heading" in effective_nav):
            return "B"

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
                logger.warning(f"[GeospatialEngine] Warning: Could not create transformer for {source_crs}: {e}")
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
