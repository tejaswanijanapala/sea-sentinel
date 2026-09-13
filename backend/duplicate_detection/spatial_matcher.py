"""
Spatial Target Matching & Duplicate Target Detection Engine for Sea Sentinel (Offline-Native).
Deduplicates repeated acoustic observations of the same physical ocean debris across multiple SSS images/tracks.
Uses R-Tree/Bounding-Box pre-filtering + high-precision Geodesic distance verification + class compatibility scoring.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math

from backend.database.local_db import LocalDatabase
from backend.ai.geospatial.georeferencing_engine import CoordinateUtilities


# Compatible debris class mapping
CLASS_COMPATIBILITY = {
    "ghost_net": ["ghost_net", "fishing_gear", "rope", "debris", "plastic"],
    "fishing_gear": ["fishing_gear", "ghost_net", "rope", "debris", "metal"],
    "plastic": ["plastic", "debris", "unknown"],
    "metal": ["metal", "debris", "shipwreck_debris"],
    "rope": ["rope", "ghost_net", "fishing_gear", "debris"],
    "tyre": ["tyre", "rubber", "debris"],
    "container": ["container", "metal", "debris"],
    "debris": ["debris", "ghost_net", "fishing_gear", "plastic", "metal", "rope", "tyre", "unknown"],
    "unknown": ["unknown", "debris", "other"]
}


class TargetMatchingService:
    """
    Service responsible for matching incoming georeferenced detections against persistent physical targets.
    """
    def __init__(self, db: LocalDatabase, matching_radius_m: float = 15.0):
        self.db = db
        self.matching_radius_m = matching_radius_m

    def match_or_create_target(
        self,
        class_name: str,
        latitude: Optional[float],
        longitude: Optional[float],
        confidence: float,
        depth: float = 0.0,
        uncertainty_radius_m: float = 5.0,
        georeference_quality: str = "EXACT",
        thumbnail_path: str = "",
        exclude_target_ids: Optional[set] = None
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Matches a detection to an existing target or creates a new one.
        Returns: (target_id, is_duplicate, target_dict)
        """
        # If detection has no geographic coordinates, create unique unreferenced target
        if latitude is None or longitude is None:
            new_target = {
                "class_name": class_name,
                "latitude": None,
                "longitude": None,
                "depth": depth,
                "confidence": confidence,
                "observation_count": 1,
                "verification_status": "UNVERIFIED",
                "target_status": "ACTIVE",
                "uncertainty_radius": uncertainty_radius_m,
                "georeference_quality": "UNREFERENCED",
                "thumbnail_path": thumbnail_path
            }
            t_id = self.db.insert_or_update_target(new_target)
            new_target["target_id"] = t_id
            return t_id, False, new_target

        # 1. Bounding-box spatial pre-filter (approx 50m window: ~0.0005 deg)
        lat_delta = max(0.0005, (self.matching_radius_m * 2.0) / 111320.0)
        lon_delta = max(0.0005, (self.matching_radius_m * 2.0) / (111320.0 * max(0.1, math.cos(math.radians(latitude)))))
        candidates = self.db.find_nearby_targets(latitude, longitude, lat_delta=lat_delta, lon_delta=lon_delta)

        best_target = None
        min_distance = float("inf")
        normalized_class = class_name.lower().replace(" ", "_").replace("-", "_")

        for cand in candidates:
            cand_id = cand.get("target_id")
            if exclude_target_ids and cand_id in exclude_target_ids:
                continue
            cand_lat = cand.get("latitude")
            cand_lon = cand.get("longitude")
            if cand_lat is None or cand_lon is None:
                continue

            dist_m = CoordinateUtilities.distance_between_coordinates(latitude, longitude, cand_lat, cand_lon)
            if dist_m <= self.matching_radius_m:
                cand_class = (cand.get("class_name") or "debris").lower().replace(" ", "_").replace("-", "_")
                # Check class compatibility
                allowed_classes = CLASS_COMPATIBILITY.get(normalized_class, [normalized_class, "debris", "unknown"])
                if cand_class in allowed_classes or normalized_class in allowed_classes:
                    if dist_m < min_distance:
                        min_distance = dist_m
                        best_target = cand

        # 2. If matching target found -> Update and merge
        if best_target:
            t_id = best_target["target_id"]
            old_obs = int(best_target.get("observation_count", 1))
            new_obs = old_obs + 1

            # Centroid weighted coordinates
            old_lat = float(best_target["latitude"])
            old_lon = float(best_target["longitude"])
            refined_lat = (old_lat * old_obs + latitude) / new_obs
            refined_lon = (old_lon * old_obs + longitude) / new_obs

            # Refined confidence and uncertainty
            refined_conf = max(float(best_target.get("confidence", 0.8)), confidence)
            refined_uncertainty = min(float(best_target.get("uncertainty_radius", 5.0)), uncertainty_radius_m)

            updated_target = {
                "target_id": t_id,
                "class_name": best_target["class_name"],  # Preserve or human-verified
                "latitude": refined_lat,
                "longitude": refined_lon,
                "depth": depth if depth > 0 else best_target.get("depth", 0.0),
                "confidence": refined_conf,
                "first_seen": best_target.get("first_seen"),
                "last_seen": datetime.utcnow().isoformat(),
                "observation_count": new_obs,
                "cluster_id": best_target.get("cluster_id"),
                "verification_status": best_target.get("verification_status", "UNVERIFIED"),
                "target_status": "ACTIVE",
                "uncertainty_radius": refined_uncertainty,
                "georeference_quality": best_target.get("georeference_quality", "EXACT"),
                "thumbnail_path": thumbnail_path or best_target.get("thumbnail_path", "")
            }
            self.db.insert_or_update_target(updated_target)
            return t_id, True, updated_target

        # 3. Otherwise -> Create new physical target
        new_target = {
            "class_name": class_name,
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "confidence": confidence,
            "first_seen": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
            "observation_count": 1,
            "verification_status": "UNVERIFIED",
            "target_status": "ACTIVE",
            "uncertainty_radius": uncertainty_radius_m,
            "georeference_quality": georeference_quality,
            "thumbnail_path": thumbnail_path
        }
        t_id = self.db.insert_or_update_target(new_target)
        new_target["target_id"] = t_id
        return t_id, False, new_target
