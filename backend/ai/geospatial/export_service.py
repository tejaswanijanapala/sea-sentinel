"""
Multi-Format GIS Export Engine for Sea Sentinel (Offline-Native).
Generates standard spatial exports in:
- GeoJSON (FeatureCollections for targets, clusters, tracks, coverage)
- CSV (Tabular export with full audit and confidence scores)
- KML (3D Keyhole Markup Language placemarks and polygons)
- GeoPackage / SQLite Spatial Data Layer
"""

from typing import Dict, Any, List, Optional
import json
import csv
import io
import os
import sqlite3

from backend.database.local_db import LocalDatabase


class GISExportService:
    """
    Dedicated service for exporting Sea Sentinel spatial datasets offline.
    """
    def __init__(self, db: LocalDatabase):
        self.db = db

    def export_geojson(self, layer: str = "all") -> Dict[str, Any]:
        """Exports targets, clusters, tracks, and/or coverage as a GeoJSON FeatureCollection."""
        features = []

        # 1. Targets Layer
        if layer in ("all", "targets"):
            targets = self.db.get_all_targets()
            for t in targets:
                if t.get("latitude") is not None and t.get("longitude") is not None:
                    feat = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [round(float(t["longitude"]), 6), round(float(t["latitude"]), 6)]
                        },
                        "properties": {
                            "layer_type": "debris_target",
                            "target_id": t["target_id"],
                            "class_name": t["class_name"],
                            "confidence": round(float(t["confidence"]), 4),
                            "depth_m": round(float(t.get("depth", 0.0)), 2),
                            "uncertainty_radius_m": round(float(t.get("uncertainty_radius", 5.0)), 2),
                            "observation_count": int(t.get("observation_count", 1)),
                            "cluster_id": t.get("cluster_id"),
                            "verification_status": t.get("verification_status", "UNVERIFIED"),
                            "georeference_quality": t.get("georeference_quality", "EXACT"),
                            "first_seen": t.get("first_seen"),
                            "last_seen": t.get("last_seen")
                        }
                    }
                    features.append(feat)

        # 2. Clusters Layer
        if layer in ("all", "clusters"):
            clusters = self.db.get_all_clusters()
            for c in clusters:
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(c["center_longitude"]), 6), round(float(c["center_latitude"]), 6)]
                    },
                    "properties": {
                        "layer_type": "debris_cluster",
                        "cluster_id": c["cluster_id"],
                        "target_count": c["target_count"],
                        "radius_m": c["radius_m"],
                        "density": c["density"],
                        "dominant_class": c["dominant_class"],
                        "cluster_type": c.get("cluster_type", "DBSCAN")
                    }
                }
                features.append(feat)

        # 3. Survey Tracks Layer
        if layer in ("all", "tracks"):
            tracks = self.db.get_all_survey_tracks()
            for trk in tracks:
                geom = trk.get("geometry") or {}
                if geom.get("coordinates"):
                    feat = {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "layer_type": "survey_track",
                            "track_id": trk["track_id"],
                            "survey_id": trk.get("survey_id"),
                            "heading": trk.get("heading"),
                            "speed_knots": trk.get("speed_knots"),
                            "timestamp": trk.get("timestamp")
                        }
                    }
                    features.append(feat)

        # 4. Survey Coverage Layer
        if layer in ("all", "coverage"):
            coverages = self.db.get_all_survey_coverage()
            for cov in coverages:
                geom = cov.get("geometry") or {}
                if geom.get("coordinates"):
                    feat = {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "layer_type": "survey_coverage",
                            "coverage_id": cov["coverage_id"],
                            "survey_id": cov.get("survey_id"),
                            "range_m": cov.get("range_m"),
                            "area_sq_m": cov.get("area_sq_m"),
                            "timestamp": cov.get("timestamp")
                        }
                    }
                    features.append(feat)

        return {
            "type": "FeatureCollection",
            "name": f"Sea_Sentinel_GIS_Export_{layer}",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

    def export_csv(self) -> str:
        """Exports targets table as CSV string."""
        targets = self.db.get_all_targets()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Target ID", "Class", "Confidence", "Latitude", "Longitude",
            "Depth (m)", "Accuracy (+/- m)", "Observations", "Cluster ID",
            "Verification Status", "Georeference Quality", "First Seen", "Last Seen"
        ])
        for t in targets:
            writer.writerow([
                t["target_id"],
                t["class_name"],
                round(float(t["confidence"]), 4),
                t.get("latitude"),
                t.get("longitude"),
                round(float(t.get("depth", 0.0)), 2),
                round(float(t.get("uncertainty_radius", 5.0)), 2),
                t.get("observation_count", 1),
                t.get("cluster_id") or "NONE",
                t.get("verification_status", "UNVERIFIED"),
                t.get("georeference_quality", "EXACT"),
                t.get("first_seen"),
                t.get("last_seen")
            ])
        return output.getvalue()

    def export_kml(self) -> str:
        """Exports targets as KML document for Google Earth / GIS viewers."""
        targets = self.db.get_all_targets()
        kml = ['<?xml version="1.0" encoding="UTF-8"?>']
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('  <Document>')
        kml.append('    <name>Sea Sentinel Ocean Debris GIS Intelligence</name>')

        for t in targets:
            if t.get("latitude") is not None and t.get("longitude") is not None:
                kml.append('    <Placemark>')
                kml.append(f'      <name>{t["target_id"]} - {t["class_name"]}</name>')
                kml.append(f'      <description><![CDATA[')
                kml.append(f'        <b>Class:</b> {t["class_name"]}<br/>')
                kml.append(f'        <b>Confidence:</b> {float(t["confidence"])*100:.1f}%<br/>')
                kml.append(f'        <b>Depth:</b> {t.get("depth", 0.0):.1f} m<br/>')
                kml.append(f'        <b>Accuracy:</b> &plusmn;{t.get("uncertainty_radius", 5.0):.1f} m<br/>')
                kml.append(f'        <b>Observations:</b> {t.get("observation_count", 1)}<br/>')
                kml.append(f'        <b>Status:</b> {t.get("verification_status", "UNVERIFIED")}<br/>')
                kml.append(f'      ]]></description>')
                kml.append('      <Point>')
                kml.append(f'        <coordinates>{t["longitude"]},{t["latitude"]},0</coordinates>')
                kml.append('      </Point>')
                kml.append('    </Placemark>')

        kml.append('  </Document>')
        kml.append('</kml>')
        return "\n".join(kml)
