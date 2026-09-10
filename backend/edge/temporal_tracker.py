"""
Lightweight 2D Kalman Filter & Centroid/IoU Temporal Multi-Object Tracker.
Enforces temporal consistency across consecutive Side-Scan Sonar pings.
Boosts confidence for persistent acoustic returns and suppresses single-ping transient clutter.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import time
import numpy as np


class TrackState:
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTED = "COASTED"
    DELETED = "DELETED"


class SonarTrack:
    """
    State of a single persistent debris track with 2D Kalman filter.
    State vector: [x, y, vx, vy]^T
    """
    def __init__(self, track_id: str, centroid: Tuple[float, float], bbox: Dict[str, float], initial_cand: Dict[str, Any]):
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.bbox = bbox
        self.class_name = initial_cand.get("class", "unknown_debris")
        self.candidate_history: List[Dict[str, Any]] = [initial_cand]

        # Kalman Filter State & Covariances
        self.x = np.array([centroid[0], centroid[1], 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([20.0, 20.0, 10.0, 10.0]).astype(np.float32)
        self.Q = np.diag([2.0, 2.0, 1.0, 1.0]).astype(np.float32)  # Process noise
        self.R = np.diag([10.0, 10.0]).astype(np.float32)           # Measurement noise

    def predict(self, dt: float = 1.0):
        """Kalman Prediction Step."""
        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.state = TrackState.COASTED

    def update(self, measurement: Tuple[float, float], cand: Dict[str, Any]):
        """Kalman Measurement Update Step."""
        z = np.array(measurement, dtype=np.float32)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        y = z - H @ self.x  # Innovation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman Gain

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        self.bbox = cand.get("bbox", self.bbox)
        self.class_name = cand.get("class", self.class_name)
        self.candidate_history.append(cand)
        if len(self.candidate_history) > 10:
            self.candidate_history.pop(0)

        if self.hits >= 2:
            self.state = TrackState.CONFIRMED

    @property
    def centroid(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


class SonarKalmanTracker:
    """
    Multi-ping acoustic target tracker for AUV/ROV side-scan surveys.
    Uses gating distance and IoU to maintain track continuity across swaths.
    """
    def __init__(
        self,
        max_coasting_pings: int = 3,
        gating_distance_px: float = 65.0,
        min_iou_overlap: float = 0.15
    ):
        self.max_coasting_pings = max_coasting_pings
        self.gating_distance_px = gating_distance_px
        self.min_iou_overlap = min_iou_overlap
        self.tracks: Dict[str, SonarTrack] = {}
        self.next_id = 1
        self.last_update_time = time.perf_counter()

    @staticmethod
    def _compute_iou(boxA: Dict[str, float], boxB: Dict[str, float]) -> float:
        xA = max(boxA.get("x1", 0), boxB.get("x1", 0))
        yA = max(boxA.get("y1", 0), boxB.get("y1", 0))
        xB = min(boxA.get("x2", 0), boxB.get("x2", 0))
        yB = min(boxA.get("y2", 0), boxB.get("y2", 0))

        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(1.0, (boxA.get("x2", 0) - boxA.get("x1", 0)) * (boxA.get("y2", 0) - boxA.get("y1", 0)))
        areaB = max(1.0, (boxB.get("x2", 0) - boxB.get("x1", 0)) * (boxB.get("y2", 0) - boxB.get("y1", 0)))
        return inter / float(areaA + areaB - inter)

    def update(self, candidates: List[Dict[str, Any]], dt: float = 1.0) -> List[Dict[str, Any]]:
        """
        Predicts existing tracks and associates new frame detections.
        Returns candidates enriched with track_id, temporal_hits, and track_state.
        """
        # 1. Predict all active tracks
        for track in self.tracks.values():
            track.predict(dt=dt)

        matched_tracks = set()
        matched_cands = set()
        updated_candidates: List[Dict[str, Any]] = []

        # 2. Association: Match candidates to existing tracks
        for c_idx, cand in enumerate(candidates):
            c_cent = cand.get("centroid", (0.0, 0.0))
            if isinstance(c_cent, list):
                c_cent = (float(c_cent[0]), float(c_cent[1]))
            c_box = cand.get("bbox", {})

            best_track_id = None
            min_dist = float("inf")

            for t_id, track in self.tracks.items():
                if t_id in matched_tracks:
                    continue
                t_cent = track.centroid
                dist = math.hypot(c_cent[0] - t_cent[0], c_cent[1] - t_cent[1])
                iou = self._compute_iou(c_box, track.bbox)

                # Gate: Centroid distance < threshold or significant bounding box IoU
                if (dist < self.gating_distance_px or iou > self.min_iou_overlap) and dist < min_dist:
                    min_dist = dist
                    best_track_id = t_id

            if best_track_id is not None:
                track = self.tracks[best_track_id]
                track.update(c_cent, cand)
                matched_tracks.add(best_track_id)
                matched_cands.add(c_idx)

                cand_copy = dict(cand)
                cand_copy["track_id"] = best_track_id
                cand_copy["temporal_hits"] = track.hits
                cand_copy["track_state"] = track.state
                cand_copy["temporal_consistency_score"] = min(1.0, 0.4 + 0.15 * track.hits)
                updated_candidates.append(cand_copy)

        # 3. Create new tracks for unmatched candidates
        for c_idx, cand in enumerate(candidates):
            if c_idx not in matched_cands:
                c_cent = cand.get("centroid", (0.0, 0.0))
                if isinstance(c_cent, list):
                    c_cent = (float(c_cent[0]), float(c_cent[1]))
                c_box = cand.get("bbox", {})
                new_id = f"TRK_{self.next_id:04d}"
                self.next_id += 1

                new_track = SonarTrack(new_id, c_cent, c_box, cand)
                self.tracks[new_id] = new_track

                cand_copy = dict(cand)
                cand_copy["track_id"] = new_id
                cand_copy["temporal_hits"] = 1
                cand_copy["track_state"] = TrackState.TENTATIVE
                cand_copy["temporal_consistency_score"] = 0.45
                updated_candidates.append(cand_copy)

        # 4. Prune dead tracks that have coasted beyond tolerance
        dead_ids = [
            t_id for t_id, track in self.tracks.items()
            if track.time_since_update > self.max_coasting_pings
        ]
        for t_id in dead_ids:
            del self.tracks[t_id]

        return updated_candidates
