"""
DBSCAN Spatial Clustering Engine for Sea Sentinel (Offline-Native).
Performs density-based clustering on persistent ocean debris targets using geodesic distance in meters.
Groups concentrated debris fields, isolates noise/scattered targets, and generates cluster bounding boxes.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter
import math

from backend.database.local_db import LocalDatabase
from backend.ai.geospatial.georeferencing_engine import CoordinateUtilities


class ClusteringService:
    """
    DBSCAN spatial clustering service for geographic points (WGS84) with metric epsilon.
    """
    def __init__(self, db: LocalDatabase, epsilon_meters: float = 250.0, min_samples: int = 2):
        self.db = db
        self.epsilon_meters = epsilon_meters
        self.min_samples = min_samples

    def recalculate_clusters(
        self,
        epsilon_meters: Optional[float] = None,
        min_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs DBSCAN clustering over all georeferenced targets, updates target cluster_ids,
        and saves cluster summary records to the database.
        """
        eps = epsilon_meters if epsilon_meters is not None else self.epsilon_meters
        min_pts = min_samples if min_samples is not None else self.min_samples

        targets = self.db.get_all_targets()
        georef_targets = [t for t in targets if t.get("latitude") is not None and t.get("longitude") is not None]

        if not georef_targets:
            self.db.save_clusters([])
            return []

        n = len(georef_targets)
        # Precompute distance matrix or neighbor lists
        neighbors_map: List[List[int]] = []
        for i in range(n):
            lat1 = float(georef_targets[i]["latitude"])
            lon1 = float(georef_targets[i]["longitude"])
            nbrs = []
            for j in range(n):
                lat2 = float(georef_targets[j]["latitude"])
                lon2 = float(georef_targets[j]["longitude"])
                dist = CoordinateUtilities.distance_between_coordinates(lat1, lon1, lat2, lon2)
                if dist <= eps:
                    nbrs.append(j)
            neighbors_map.append(nbrs)

        # DBSCAN core algorithm
        cluster_labels = [-1] * n  # -1 = unassigned / noise
        current_cluster_id = 0
        visited: Set[int] = set()

        for i in range(n):
            if i in visited:
                continue
            visited.add(i)

            neighbors = list(neighbors_map[i])
            if len(neighbors) < min_pts:
                cluster_labels[i] = -1  # Noise / singleton
            else:
                current_cluster_id += 1
                cluster_labels[i] = current_cluster_id

                # Expand cluster
                queue = list(neighbors)
                while queue:
                    neighbor_idx = queue.pop(0)
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        sub_nbrs = neighbors_map[neighbor_idx]
                        if len(sub_nbrs) >= min_pts:
                            queue.extend([idx for idx in sub_nbrs if idx not in queue])

                    if cluster_labels[neighbor_idx] == -1:
                        cluster_labels[neighbor_idx] = current_cluster_id

        # Aggregate cluster geometries and statistics
        clusters_summary = []
        clusters_grouped: Dict[int, List[Dict[str, Any]]] = {}

        for i, cid in enumerate(cluster_labels):
            t = georef_targets[i]
            if cid > 0:
                cluster_tag = f"CL-{cid:02d}"
                t["cluster_id"] = cluster_tag
                if cid not in clusters_grouped:
                    clusters_grouped[cid] = []
                clusters_grouped[cid].append(t)
            else:
                t["cluster_id"] = None

            # Update target in database
            self.db.insert_or_update_target(t)

        # Build cluster records
        for cid, member_targets in clusters_grouped.items():
            lats = [float(m["latitude"]) for m in member_targets]
            lons = [float(m["longitude"]) for m in member_targets]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            # Compute maximum radius from center
            max_radius = 0.0
            for lat, lon in zip(lats, lons):
                d = CoordinateUtilities.distance_between_coordinates(center_lat, center_lon, lat, lon)
                if d > max_radius:
                    max_radius = d
            radius_m = max(5.0, round(max_radius, 1))

            # Area in km² and density
            area_km2 = (math.pi * (radius_m / 1000.0) ** 2)
            density = round(len(member_targets) / max(0.001, area_km2), 1)

            # Dominant class
            class_counts = Counter([m.get("class_name", "debris") for m in member_targets])
            dominant_class = class_counts.most_common(1)[0][0]

            bounding_box = {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lon": min(lons),
                "max_lon": max(lons)
            }

            cluster_record = {
                "cluster_id": f"CL-{cid:02d}",
                "center_latitude": center_lat,
                "center_longitude": center_lon,
                "target_count": len(member_targets),
                "radius_m": radius_m,
                "density": density,
                "cluster_type": "DBSCAN",
                "dominant_class": dominant_class,
                "bounding_box": bounding_box
            }
            clusters_summary.append(cluster_record)

        self.db.save_clusters(clusters_summary)
        return clusters_summary
