"""
Stage 7: Geospatial Report Exporter CLI
Extracts georeferenced targets and exports:
  - GeoJSON FeatureCollection for Web GIS / Leaflet / QGIS
  - Hydrographic Survey CSV for maritime inspection compliance
"""

import os
import sys
import argparse
import json
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.geospatial.geotagger import GeospatialEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Export Georeferenced Sonar Targets to GeoJSON/CSV")
    parser.add_argument("--db-path", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "audit", "survey_audit.db"),
                        help="Path to SQLite survey audit database")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "geospatial"),
                        help="Directory to save exported GeoJSON and CSV")
    return parser.parse_args()


def export(args):
    os.makedirs(args.output_dir, exist_ok=True)
    engine = GeospatialEngine()

    print("=" * 75)
    print("STAGE 7: GEOSPATIAL TARGET REPORT EXPORTER")
    print("Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology")
    print("=" * 75)

    if not os.path.exists(args.db_path):
        print(f"[ERROR] Database not found at: {args.db_path}")
        return

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM target_detections WHERE lat IS NOT NULL AND lon IS NOT NULL")
        rows = cursor.fetchall()
        targets = [dict(r) for r in rows]
    finally:
        conn.close()

    print(f"Loaded {len(targets)} georeferenced targets from audit database.")

    if not targets:
        print("[INFO] No targets with coordinates found in audit database.")
        return

    # Export GeoJSON
    geojson_path = os.path.join(args.output_dir, "survey_targets.geojson")
    engine.export_geojson(targets, geojson_path)
    print(f"  [EXPORTED GEOJSON]: {geojson_path}")

    # Export CSV
    csv_path = os.path.join(args.output_dir, "survey_targets_hydrographic.csv")
    engine.export_csv(targets, csv_path)
    print(f"  [EXPORTED CSV]:     {csv_path}")

    # Bounding extent
    lats = [t["lat"] for t in targets if t.get("lat") is not None]
    lons = [t["lon"] for t in targets if t.get("lon") is not None]
    if lats and lons:
        print("\nSurvey Spatial Extent:")
        print(f"  Latitude:  [{min(lats):.6f}, {max(lats):.6f}]")
        print(f"  Longitude: [{min(lons):.6f}, {max(lons):.6f}]")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    args = parse_args()
    export(args)
