"""
Local SQLite Database Manager for Sea Sentinel (Edge-First, Offline-Native GIS).
Maintains persistent spatial database for:
- survey_images
- detections
- targets (deduplicated persistent physical objects)
- survey_tracks (vessel/towfish trajectory line strings)
- survey_coverage (swath polygon areas)
- clusters (DBSCAN clusters)
- reviews (human-in-the-loop review history)
- sync_queue & model_registry
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import sqlite3
import uuid
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_DIR = os.path.join(PROJECT_ROOT, "outputs", "database")
DB_PATH = os.path.join(DB_DIR, "sea_sentinel_edge.db")


class LocalDatabase:
    """
    Robust local SQLite spatial database with automatic table creation, WAL mode for concurrency,
    indexed spatial queries, and multi-tier transaction support.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self):
        with self.get_connection() as conn:
            # 1. Base surveys table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS surveys (
                survey_id TEXT PRIMARY KEY,
                image_id TEXT,
                image_name TEXT,
                timestamp TEXT,
                processing_mode TEXT,
                hardware TEXT,
                latency_ms REAL,
                total_objects INTEGER DEFAULT 0,
                status TEXT,
                sync_status TEXT DEFAULT 'PENDING',
                synced_at TEXT,
                metadata_json TEXT
            );
            """)

            # 2. Base detections table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id TEXT UNIQUE,
                image_id TEXT,
                survey_id TEXT,
                target_id TEXT,
                class_label TEXT,
                confidence REAL,
                detection_source TEXT,
                bbox_x REAL,
                bbox_y REAL,
                bbox_width REAL,
                bbox_height REAL,
                segmentation_available INTEGER DEFAULT 0,
                segmentation_json TEXT,
                sonar_x REAL,
                sonar_y REAL,
                latitude REAL,
                longitude REAL,
                length_m REAL,
                width_m REAL,
                area_m2 REAL,
                uncertainty_radius REAL,
                georeference_method TEXT,
                georeference_quality TEXT,
                risk_score TEXT,
                risk_category TEXT,
                habitat_overlap_json TEXT,
                model_name TEXT,
                model_version TEXT,
                model_hash TEXT,
                created_at TEXT
            );
            """)

            # Safe column migration for detections if pre-existing schema lacked columns
            cursor = conn.execute("PRAGMA table_info(detections);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            needed_cols = {
                "detection_id": "TEXT",
                "image_id": "TEXT",
                "target_id": "TEXT",
                "bbox_x": "REAL",
                "bbox_y": "REAL",
                "bbox_width": "REAL",
                "bbox_height": "REAL",
                "segmentation_available": "INTEGER DEFAULT 0",
                "sonar_x": "REAL",
                "sonar_y": "REAL",
                "uncertainty_radius": "REAL DEFAULT 5.0",
                "georeference_method": "TEXT",
                "georeference_quality": "TEXT",
                "model_name": "TEXT",
                "model_version": "TEXT",
                "model_hash": "TEXT"
            }
            for col_name, col_type in needed_cols.items():
                if col_name not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE detections ADD COLUMN {col_name} {col_type};")
                    except Exception:
                        pass

            # 3. survey_images: Full SSS image metadata table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS survey_images (
                image_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                timestamp TEXT,
                latitude REAL,
                longitude REAL,
                heading REAL,
                depth REAL,
                sonar_range REAL,
                altitude REAL,
                sensor_id TEXT,
                frequency TEXT,
                metadata_source TEXT,
                metadata_quality TEXT,
                processing_status TEXT DEFAULT 'PENDING',
                created_at TEXT
            );
            """)

            # 4. targets: Persistent deduplicated physical debris objects
            conn.execute("""
            CREATE TABLE IF NOT EXISTS targets (
                target_id TEXT PRIMARY KEY,
                class_name TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                depth REAL,
                confidence REAL,
                first_seen TEXT,
                last_seen TEXT,
                observation_count INTEGER DEFAULT 1,
                cluster_id TEXT,
                verification_status TEXT DEFAULT 'UNVERIFIED',
                target_status TEXT DEFAULT 'ACTIVE',
                uncertainty_radius REAL DEFAULT 5.0,
                georeference_quality TEXT DEFAULT 'EXACT',
                thumbnail_path TEXT,
                metadata_json TEXT
            );
            """)

            # 5. survey_tracks: Vessel/Towfish trajectory path LineStrings
            conn.execute("""
            CREATE TABLE IF NOT EXISTS survey_tracks (
                track_id TEXT PRIMARY KEY,
                survey_id TEXT,
                image_id TEXT,
                start_latitude REAL,
                start_longitude REAL,
                end_latitude REAL,
                end_longitude REAL,
                geometry_json TEXT,
                heading REAL,
                speed_knots REAL,
                timestamp TEXT
            );
            """)

            # 6. survey_coverage: Scanned swath corridor polygons
            conn.execute("""
            CREATE TABLE IF NOT EXISTS survey_coverage (
                coverage_id TEXT PRIMARY KEY,
                survey_id TEXT,
                image_id TEXT,
                geometry_json TEXT,
                coverage_width_m REAL,
                range_m REAL,
                area_sq_m REAL,
                timestamp TEXT
            );
            """)

            # 7. clusters: DBSCAN spatial clusters
            conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                center_latitude REAL,
                center_longitude REAL,
                target_count INTEGER DEFAULT 0,
                radius_m REAL,
                density REAL,
                cluster_type TEXT DEFAULT 'DBSCAN',
                dominant_class TEXT,
                bounding_box_json TEXT,
                created_at TEXT
            );
            """)

            # 8. reviews: Human verification history
            conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                review_id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                reviewer_decision TEXT NOT NULL,
                correct_class TEXT,
                comments TEXT,
                old_confidence REAL,
                new_confidence REAL,
                reviewer_name TEXT DEFAULT 'Operator',
                created_at TEXT,
                FOREIGN KEY (target_id) REFERENCES targets(target_id) ON DELETE CASCADE
            );
            """)

            # 9. sync_queue, model_registry, benchmark_history
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                survey_id TEXT UNIQUE,
                payload_json TEXT,
                enqueued_at TEXT,
                sync_status TEXT DEFAULT 'PENDING',
                attempts INTEGER DEFAULT 0,
                last_error TEXT,
                synced_at TEXT
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                model_name TEXT,
                version TEXT,
                architecture TEXT,
                checksum_sha256 TEXT,
                weights_path TEXT,
                is_active INTEGER DEFAULT 1,
                status TEXT,
                updated_at TEXT,
                PRIMARY KEY (model_name, version)
            );

            CREATE TABLE IF NOT EXISTS benchmark_history (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                sample_count INTEGER,
                p50_ms REAL,
                p95_ms REAL,
                max_ms REAL,
                avg_ms REAL,
                slowest_component TEXT,
                budget_status TEXT,
                system_specs TEXT
            );

            -- Indexes for fast queries
            CREATE INDEX IF NOT EXISTS idx_survey_images_time ON survey_images(timestamp);
            CREATE INDEX IF NOT EXISTS idx_targets_coords ON targets(latitude, longitude);
            CREATE INDEX IF NOT EXISTS idx_targets_class ON targets(class_name);
            CREATE INDEX IF NOT EXISTS idx_targets_verif ON targets(verification_status);
            CREATE INDEX IF NOT EXISTS idx_detections_target ON detections(target_id);
            CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id);
            CREATE INDEX IF NOT EXISTS idx_detections_coords ON detections(latitude, longitude);
            CREATE INDEX IF NOT EXISTS idx_survey_tracks_time ON survey_tracks(timestamp);
            CREATE INDEX IF NOT EXISTS idx_survey_coverage_time ON survey_coverage(timestamp);
            CREATE INDEX IF NOT EXISTS idx_reviews_target ON reviews(target_id);
            """)

            # Try creating SQLite R-Tree virtual table for target spatial indexing if supported
            try:
                conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS target_spatial_idx USING rtree(
                    id INTEGER PRIMARY KEY,
                    min_lat REAL, max_lat REAL,
                    min_lon REAL, max_lon REAL
                );
                """)
            except sqlite3.OperationalError:
                pass  # Fallback to standard B-tree index on lat/lon
            conn.commit()

    # -----------------------------------------------------------------
    # Survey Images Management
    # -----------------------------------------------------------------
    def insert_survey_image(self, img_data: Dict[str, Any]) -> str:
        image_id = img_data.get("image_id") or f"IMG_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:21]}"
        ts = img_data.get("timestamp") or datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO survey_images (
                image_id, filename, file_path, timestamp, latitude, longitude,
                heading, depth, sonar_range, altitude, sensor_id, frequency,
                metadata_source, metadata_quality, processing_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                img_data.get("filename", "unknown_sonar.png"),
                img_data.get("file_path", ""),
                ts,
                img_data.get("latitude"),
                img_data.get("longitude"),
                img_data.get("heading", 0.0),
                img_data.get("depth", 0.0),
                img_data.get("sonar_range", 50.0),
                img_data.get("altitude", 5.0),
                img_data.get("sensor_id", "SSS_EDGETECH_4200"),
                img_data.get("frequency", "400kHz"),
                img_data.get("metadata_source", "GEOTIFF"),
                img_data.get("metadata_quality", "HIGH"),
                img_data.get("processing_status", "COMPLETED"),
                ts
            ))
            conn.commit()
        return image_id

    def get_survey_images(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM survey_images ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Targets Management (Deduplicated Persistent Entities)
    # -----------------------------------------------------------------
    def insert_or_update_target(self, target_data: Dict[str, Any]) -> str:
        t_id = target_data.get("target_id")
        if not t_id:
            with self.get_connection() as conn:
                count_cur = conn.execute("SELECT COUNT(*) FROM targets")
                next_num = count_cur.fetchone()[0] + 1
                t_id = f"SS-{next_num:04d}"

        ts = datetime.utcnow().isoformat()
        first_seen = target_data.get("first_seen") or ts
        last_seen = target_data.get("last_seen") or ts

        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO targets (
                target_id, class_name, latitude, longitude, depth, confidence,
                first_seen, last_seen, observation_count, cluster_id,
                verification_status, target_status, uncertainty_radius,
                georeference_quality, thumbnail_path, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t_id,
                target_data.get("class_name", "debris"),
                target_data.get("latitude"),
                target_data.get("longitude"),
                target_data.get("depth", 0.0),
                float(target_data.get("confidence", 0.8)),
                first_seen,
                last_seen,
                int(target_data.get("observation_count", 1)),
                target_data.get("cluster_id"),
                target_data.get("verification_status", "UNVERIFIED"),
                target_data.get("target_status", "ACTIVE"),
                float(target_data.get("uncertainty_radius", 5.0)),
                target_data.get("georeference_quality", "EXACT"),
                target_data.get("thumbnail_path", ""),
                json.dumps(target_data.get("metadata", {}))
            ))
            conn.commit()
        return t_id

    def get_all_targets(self, min_confidence: float = 0.0, class_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            query = "SELECT * FROM targets WHERE confidence >= ?"
            params: List[Any] = [min_confidence]
            if class_filter and class_filter.lower() != "all":
                query += " AND LOWER(class_name) = LOWER(?)"
                params.append(class_filter)
            query += " ORDER BY last_seen DESC"
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_target_by_id(self, target_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM targets WHERE target_id = ?", (target_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # -----------------------------------------------------------------
    # Candidate Targets Nearby Spatial Query (Bounding Box Pre-Filter)
    # -----------------------------------------------------------------
    def find_nearby_targets(self, lat: float, lon: float, lat_delta: float = 0.0005, lon_delta: float = 0.0005) -> List[Dict[str, Any]]:
        """Finds targets within a bounding box window for fast distance verification."""
        if lat is None or lon is None:
            return []
        with self.get_connection() as conn:
            cursor = conn.execute("""
            SELECT * FROM targets 
            WHERE latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
            """, (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta))
            return [dict(r) for r in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Survey Tracks & Coverage Swaths
    # -----------------------------------------------------------------
    def insert_survey_track(self, track_data: Dict[str, Any]) -> str:
        track_id = track_data.get("track_id") or f"TRK_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:21]}"
        ts = track_data.get("timestamp") or datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO survey_tracks (
                track_id, survey_id, image_id, start_latitude, start_longitude,
                end_latitude, end_longitude, geometry_json, heading, speed_knots, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                track_data.get("survey_id"),
                track_data.get("image_id"),
                track_data.get("start_latitude"),
                track_data.get("start_longitude"),
                track_data.get("end_latitude"),
                track_data.get("end_longitude"),
                json.dumps(track_data.get("geometry", {})),
                track_data.get("heading", 0.0),
                track_data.get("speed_knots", 3.0),
                ts
            ))
            conn.commit()
        return track_id

    def get_all_survey_tracks(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM survey_tracks ORDER BY timestamp ASC")
            tracks = []
            for r in cursor.fetchall():
                d = dict(r)
                if d.get("geometry_json"):
                    try:
                        d["geometry"] = json.loads(d["geometry_json"])
                    except Exception:
                        d["geometry"] = {}
                tracks.append(d)
            return tracks

    def insert_survey_coverage(self, cov_data: Dict[str, Any]) -> str:
        coverage_id = cov_data.get("coverage_id") or f"COV_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:21]}"
        ts = cov_data.get("timestamp") or datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO survey_coverage (
                coverage_id, survey_id, image_id, geometry_json, coverage_width_m,
                range_m, area_sq_m, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                coverage_id,
                cov_data.get("survey_id"),
                cov_data.get("image_id"),
                json.dumps(cov_data.get("geometry", {})),
                cov_data.get("coverage_width_m", 100.0),
                cov_data.get("range_m", 50.0),
                cov_data.get("area_sq_m", 5000.0),
                ts
            ))
            conn.commit()
        return coverage_id

    def get_all_survey_coverage(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM survey_coverage ORDER BY timestamp ASC")
            coverages = []
            for r in cursor.fetchall():
                d = dict(r)
                if d.get("geometry_json"):
                    try:
                        d["geometry"] = json.loads(d["geometry_json"])
                    except Exception:
                        d["geometry"] = {}
                coverages.append(d)
            return coverages

    # -----------------------------------------------------------------
    # DBSCAN Clusters
    # -----------------------------------------------------------------
    def save_clusters(self, clusters: List[Dict[str, Any]]):
        ts = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute("DELETE FROM clusters")
            for c in clusters:
                conn.execute("""
                INSERT INTO clusters (
                    cluster_id, center_latitude, center_longitude, target_count,
                    radius_m, density, cluster_type, dominant_class, bounding_box_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.get("cluster_id"),
                    c.get("center_latitude"),
                    c.get("center_longitude"),
                    c.get("target_count", 0),
                    c.get("radius_m", 0.0),
                    c.get("density", 0.0),
                    c.get("cluster_type", "DBSCAN"),
                    c.get("dominant_class", "debris"),
                    json.dumps(c.get("bounding_box", {})),
                    ts
                ))
            conn.commit()

    def get_all_clusters(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM clusters ORDER BY target_count DESC")
            clusters = []
            for r in cursor.fetchall():
                d = dict(r)
                if d.get("bounding_box_json"):
                    try:
                        d["bounding_box"] = json.loads(d["bounding_box_json"])
                    except Exception:
                        d["bounding_box"] = {}
                clusters.append(d)
            return clusters

    # -----------------------------------------------------------------
    # Human-in-the-Loop Reviews
    # -----------------------------------------------------------------
    def insert_review(self, review_data: Dict[str, Any]) -> str:
        r_id = review_data.get("review_id") or f"REV_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:21]}"
        ts = datetime.utcnow().isoformat()
        target_id = review_data["target_id"]
        decision = review_data.get("reviewer_decision", "VERIFIED")
        correct_class = review_data.get("correct_class")
        comments = review_data.get("comments", "")
        old_conf = float(review_data.get("old_confidence", 0.8))
        new_conf = float(review_data.get("new_confidence", old_conf))

        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO reviews (
                review_id, target_id, reviewer_decision, correct_class,
                comments, old_confidence, new_confidence, reviewer_name, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r_id, target_id, decision, correct_class,
                comments, old_conf, new_conf,
                review_data.get("reviewer_name", "Operator"), ts
            ))

            # Update target record without overwriting original AI detection
            update_fields = ["verification_status = ?", "last_seen = ?"]
            params: List[Any] = [decision, ts]
            if correct_class:
                update_fields.append("class_name = ?")
                params.append(correct_class)
            if new_conf:
                update_fields.append("confidence = ?")
                params.append(new_conf)
            params.append(target_id)

            conn.execute(f"UPDATE targets SET {', '.join(update_fields)} WHERE target_id = ?", params)
            conn.commit()
        return r_id

    def get_target_reviews(self, target_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM reviews WHERE target_id = ? ORDER BY created_at DESC", (target_id,))
            return [dict(r) for r in cursor.fetchall()]

    # -----------------------------------------------------------------
    # Complete Map & Dashboard Statistics
    # -----------------------------------------------------------------
    def get_gis_dashboard_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            total_targets = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            verified_targets = conn.execute("SELECT COUNT(*) FROM targets WHERE verification_status = 'VERIFIED'").fetchone()[0]
            unverified_targets = conn.execute("SELECT COUNT(*) FROM targets WHERE verification_status = 'UNVERIFIED'").fetchone()[0]
            unknown_targets = conn.execute("SELECT COUNT(*) FROM targets WHERE LOWER(class_name) IN ('unknown', 'other')").fetchone()[0]
            total_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
            total_images = conn.execute("SELECT COUNT(*) FROM survey_images").fetchone()[0]
            total_tracks = conn.execute("SELECT COUNT(*) FROM survey_tracks").fetchone()[0]
            coverage_area_sqm = conn.execute("SELECT COALESCE(SUM(area_sq_m), 0.0) FROM survey_coverage").fetchone()[0]
            surveyed_km2 = round(coverage_area_sqm / 1_000_000.0, 3)

            # Class breakdown
            class_counts_cur = conn.execute("SELECT class_name, COUNT(*) as cnt FROM targets GROUP BY class_name")
            class_breakdown = {r["class_name"]: r["cnt"] for r in class_counts_cur.fetchall()}

            return {
                "total_targets": total_targets,
                "verified_targets": verified_targets,
                "unverified_targets": unverified_targets,
                "unknown_targets": unknown_targets,
                "total_clusters": total_clusters,
                "total_images": total_images,
                "total_tracks": total_tracks,
                "surveyed_area_km2": surveyed_km2,
                "class_breakdown": class_breakdown,
                "offline_status": "OFFLINE READY"
            }

    # -----------------------------------------------------------------
    # Backward-Compatible Survey & Detections Ingestion
    # -----------------------------------------------------------------
    def insert_survey(self, survey_data: Dict[str, Any], detections: List[Dict[str, Any]]) -> str:
        s_id = survey_data.get("survey_id") or survey_data.get("analysis_id") or f"SURV_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        ts = survey_data.get("timestamp") or datetime.utcnow().isoformat()
        mode = survey_data.get("processing_mode") or survey_data.get("mode") or "balanced"
        hw = survey_data.get("hardware") or "CPU_EDGE"
        latency = survey_data.get("latency_ms") or 0.0

        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO surveys 
            (survey_id, image_id, image_name, timestamp, processing_mode, hardware, latency_ms, total_objects, status, sync_status, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """, (
                s_id,
                survey_data.get("image_id", s_id),
                survey_data.get("image_name", "sonar_scan.png"),
                ts,
                mode,
                hw,
                latency,
                len(detections),
                "COMPLETED",
                json.dumps(survey_data)
            ))

            for d in detections:
                bbox = d.get("bbox") or d.get("pixel_bbox") or d.get("yolo_bbox") or d.get("unet_bbox") or []
                if isinstance(bbox, dict):
                    bx = float(bbox.get("x1", 0.0))
                    by = float(bbox.get("y1", 0.0))
                    bw = max(1.0, float(bbox.get("x2", bx + 50.0)) - bx)
                    bh = max(1.0, float(bbox.get("y2", by + 50.0)) - by)
                elif isinstance(bbox, (list, tuple)):
                    bx = float(bbox[0]) if len(bbox) > 0 else 0.0
                    by = float(bbox[1]) if len(bbox) > 1 else 0.0
                    bw = float(bbox[2]) if len(bbox) > 2 else 50.0
                    bh = float(bbox[3]) if len(bbox) > 3 else 50.0
                else:
                    bx, by, bw, bh = 0.0, 0.0, 50.0, 50.0

                det_id = d.get("detection_id") or f"DET_{uuid.uuid4().hex[:8].upper()}"
                
                conn.execute("""
                INSERT OR REPLACE INTO detections 
                (detection_id, survey_id, image_id, target_id, class_label, confidence, detection_source, 
                 bbox_x, bbox_y, bbox_width, bbox_height, segmentation_available, segmentation_json, 
                 latitude, longitude, length_m, width_m, area_m2, uncertainty_radius, georeference_method, 
                 georeference_quality, risk_score, risk_category, habitat_overlap_json, model_name, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    det_id,
                    s_id,
                    survey_data.get("image_id", s_id),
                    d.get("target_id"),
                    d.get("class", "debris"),
                    float(d.get("calibrated_confidence", d.get("confidence", 0.8))),
                    "/".join(d.get("sources", ["fusion"])),
                    bx, by, bw, bh,
                    1 if d.get("polygon") else 0,
                    json.dumps(d.get("polygon", [])),
                    d.get("latitude"),
                    d.get("longitude"),
                    d.get("length_m", 1.0),
                    d.get("width_m", 1.0),
                    d.get("area_sq_m", 1.0),
                    d.get("uncertainty_radius_m", 5.0),
                    d.get("georeference_method", "EXACT"),
                    d.get("georeference_quality", "EXACT"),
                    d.get("risk_score", "MEDIUM"),
                    d.get("risk_category", "MODERATE"),
                    json.dumps(d.get("habitat_overlaps", [])),
                    "YOLO11+U-Net",
                    "v2.1",
                    ts
                ))

            # Store in sync queue
            full_payload = {
                "survey": survey_data,
                "detections": detections,
                "created_at": ts
            }
            conn.execute("""
            INSERT OR REPLACE INTO sync_queue (survey_id, payload_json, enqueued_at, sync_status, attempts)
            VALUES (?, ?, ?, 'PENDING', 0)
            """, (s_id, json.dumps(full_payload), ts))
            conn.commit()

        return s_id

    def get_recent_surveys(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("""
            SELECT survey_id, image_name, timestamp, processing_mode, latency_ms, total_objects, status, sync_status 
            FROM surveys ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_georeferenced_targets(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("""
            SELECT t.*, count(d.id) as total_detections
            FROM targets t
            LEFT JOIN detections d ON t.target_id = d.target_id
            WHERE t.latitude IS NOT NULL AND t.longitude IS NOT NULL
            GROUP BY t.target_id
            ORDER BY t.last_seen DESC
            """)
            return [dict(r) for r in cursor.fetchall()]


def uuid_short() -> str:
    import uuid
    return uuid.uuid4().hex[:8]
