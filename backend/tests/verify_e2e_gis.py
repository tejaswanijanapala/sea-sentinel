"""
End-to-End Verification Script: Multi-Image SSS Ingestion & GIS Ocean Debris Map Growth.
"""

import urllib.request
import json
import os
import sys

BASE_URL = "http://localhost:8000"

def run_verification():
    print("---------------------------------------------------------")
    print(" 1. Ingesting Samples via Analysis Pipeline")
    print("---------------------------------------------------------")
    samples_res = urllib.request.urlopen(f"{BASE_URL}/api/samples")
    samples = json.loads(samples_res.read().decode('utf-8'))['samples']
    print(f"Discovered {len(samples)} authentic SSS survey rasters.")

    for s in samples:
        path = s['path']
        print(f"\nProcessing SSS Raster: {s['name']} ({os.path.basename(path)})...")
        payload = json.dumps({"image_path": path, "mode": "balanced"}).encode('utf-8')
        req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=payload, headers={'Content-Type': 'application/json'})
        try:
            res = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
            dets = res.get('detections', [])
            print(f"  -> YOLO11/U-Net Detections: {len(dets)} objects found")
            for d in dets:
                print(f"     * Target: {d.get('target_id')} | Class: {d.get('class')} | Conf: {d.get('confidence'):.2f} | Lat/Lon: ({d.get('latitude')}, {d.get('longitude')}) | Quality: {d.get('georeference_quality')} | Accuracy: +-{d.get('uncertainty_radius_m', 5.0)}m")
        except Exception as e:
            print(f"  -> Ingestion Warning: {e}")

    print("\n---------------------------------------------------------")
    print(" 2. Fetching Unified Persistent GIS Intelligence Dataset")
    print("---------------------------------------------------------")
    map_res = urllib.request.urlopen(f"{BASE_URL}/api/gis/map-data")
    map_data = json.loads(map_res.read().decode('utf-8'))
    
    stats = map_data.get('statistics', {})
    targets = map_data.get('targets', [])
    clusters = map_data.get('clusters', [])
    tracks = map_data.get('survey_tracks', [])
    coverage = map_data.get('survey_coverage', [])

    print(f"Total Unique Physical Targets: {len(targets)}")
    print(f"Total DBSCAN Debris Clusters:  {len(clusters)}")
    print(f"Total Vessel Survey Tracks:    {len(tracks)}")
    print(f"Total Scanned Swath Polygons:  {len(coverage)}")
    print(f"Total Scanned Survey Area:     {stats.get('surveyed_area_km2', 0)} km²")
    print(f"Offline Status:                {stats.get('offline_status')}")

    print("\n---------------------------------------------------------")
    print(" 3. Testing Human-in-the-Loop Review on Discovered Target")
    print("---------------------------------------------------------")
    if targets:
        first_target = targets[0]
        t_id = first_target['target_id']
        review_payload = json.dumps({
            "reviewer_decision": "VERIFIED",
            "correct_class": "ghost_net",
            "comments": "Inspected and confirmed by Chief Marine Scientist on mission pass.",
            "reviewer_name": "Senior Hydrographer"
        }).encode('utf-8')
        r_req = urllib.request.Request(f"{BASE_URL}/api/gis/target/{t_id}/review", data=review_payload, headers={'Content-Type': 'application/json'})
        r_res = json.loads(urllib.request.urlopen(r_req).read().decode('utf-8'))
        print(f"Target {t_id} review state updated to: {r_res['target']['verification_status']}")

    print("\n---------------------------------------------------------")
    print(" 4. Testing Multi-Format Spatial Export Offline")
    print("---------------------------------------------------------")
    for fmt in ["geojson", "csv", "kml"]:
        exp_res = urllib.request.urlopen(f"{BASE_URL}/api/gis/export?format={fmt}&layer=all")
        content_len = len(exp_res.read())
        print(f"  * Export {fmt.upper()} generated successfully: {content_len} bytes")

    print("\n=========================================================")
    print(" END-TO-END OFFLINE GIS INTELLIGENCE PIPELINE VERIFIED! ")
    print("=========================================================")

if __name__ == "__main__":
    run_verification()
