"""
Stage 5: DBSCAN Rock Cluster Suppression Filter
Suppresses false-positive detections caused by natural seafloor geological formations:
  - Rocky outcrops, gravel beds, and glacial moraines appear as dense clusters of small acoustic highlights.
  - Man-made anthropogenic debris (lost fishing gear, abandoned pipes, ship fragments) is typically isolated or has distinct continuous linear geometry.
Uses native spatial density clustering with zero external heavy dependencies.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class DBSCANRockFilter:
    """
    Identifies and down-weights natural rock fields using density-based spatial clustering.
    """
    def __init__(
        self,
        eps: float = 60.0,          # Pixel distance radius for neighborhood
        min_samples: int = 4,       # Minimum count to form a geological cluster
        max_rock_area_px: float = 900.0  # Typical small rock threshold (approx 30x30 px)
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.max_rock_area_px = max_rock_area_px

    def _euclidean_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Euclidean distance matrix between 2D coordinates.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _dbscan(self, points: np.ndarray) -> np.ndarray:
        """
        Native, dependency-free DBSCAN implementation.
        Returns:
            labels: Array of cluster IDs (-1 for noise/isolated points, >=0 for cluster indices)
        """
        n = len(points)
        if n == 0:
            return np.array([], dtype=int)

        dist_matrix = self._euclidean_distance_matrix(points)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = np.where(dist_matrix[i] <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                # Marked as noise/isolated
                labels[i] = -1
            else:
                labels[i] = cluster_id
                seed_set = list(neighbors[neighbors != i])

                for seed in seed_set:
                    if not visited[seed]:
                        visited[seed] = True
                        seed_neighbors = np.where(dist_matrix[seed] <= self.eps)[0]
                        if len(seed_neighbors) >= self.min_samples:
                            for sn in seed_neighbors:
                                if sn not in seed_set:
                                    seed_set.append(sn)

                    if labels[seed] == -1:
                        labels[seed] = cluster_id

                cluster_id += 1

        return labels

    def filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes candidate detections, clusters their bounding box centers,
        and flags natural rock field clusters.
        """
        if not detections:
            return []

        # Extract center coordinates and areas
        centers = []
        areas = []
        for det in detections:
            bbox = det.get("bbox", {})
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            centers.append([cx, cy])
            areas.append(w * h)

        points = np.array(centers, dtype=np.float32)
        labels = self._dbscan(points)

        # Count samples in each cluster
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        cluster_counts = dict(zip(unique_labels, counts))

        augmented_detections = []
        for idx, det in enumerate(detections):
            det_copy = dict(det)
            lbl = int(labels[idx])
            area = areas[idx]

            is_in_cluster = (lbl >= 0)
            cluster_size = cluster_counts.get(lbl, 1) if is_in_cluster else 1

            # High density + small rock-like area indicates natural rock outcrop
            is_rock_field = is_in_cluster and (cluster_size >= self.min_samples) and (area <= self.max_rock_area_px)
            rock_penalty = 0.55 if is_rock_field else (0.25 if is_in_cluster else 0.0)

            det_copy["cluster_id"] = lbl
            det_copy["is_rock_cluster"] = bool(is_rock_field)
            det_copy["rock_density_penalty"] = round(float(rock_penalty), 3)
            augmented_detections.append(det_copy)

        return augmented_detections
