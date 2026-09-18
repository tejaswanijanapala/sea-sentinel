"""
Stage 6: End-to-End Pipeline Execution CLI
Runs SIHPipelineAgent across Side-Scan Sonar imagery or survey directories.
Generates:
  - Structured JSON inspection reports
  - Geospatial CSV target export
  - Executive Markdown audit reports
"""

import os
import sys
import argparse
import json
import csv
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.orchestrator import SIHPipelineAgent
from shared.utils.logger import get_logger
logger = get_logger(__name__)




def parse_args():
    parser = argparse.ArgumentParser(description="Run End-to-End SIH57 AI Sonar Pipeline")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to single sonar image (.png, .jpg, .tif) or directory")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "reports"),
                        help="Directory to save inspection reports")
    parser.add_argument("--yolo-checkpoint", type=str, default="", help="Path to YOLO weights (.pt)")
    parser.add_argument("--unet-checkpoint", type=str, default="", help="Path to U-Net weights (.pt)")
    parser.add_argument("--export-csv", action="store_true", default=True, help="Export detections to CSV")
    return parser.parse_args()


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("=" * 75)
    logger.info("SIH26057: AI-POWERED UNDERWATER DEBRIS PIPELINE AGENT")
    logger.info("Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology")
    logger.info("=" * 75)

    config = {}
    if args.yolo_checkpoint:
        config["yolo_checkpoint"] = args.yolo_checkpoint
    if args.unet_checkpoint:
        config["unet_checkpoint"] = args.unet_checkpoint

    agent = SIHPipelineAgent(config=config)

    # Collect images
    if os.path.isdir(args.input):
        image_paths = []
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            for root, _, files in os.walk(args.input):
                for f in files:
                    if f.lower().endswith(ext):
                        image_paths.append(os.path.join(root, f))
        image_paths = sorted(image_paths)
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        logger.error(f"[ERROR] Input path does not exist: {args.input}")
        return

    if not image_paths:
        logger.error(f"[ERROR] No valid sonar images found in {args.input}")
        return

    logger.info(f"Loaded {len(image_paths)} image(s) for end-to-end analysis.\n")

    all_survey_summaries = []

    for idx, img_p in enumerate(image_paths, 1):
        filename = os.path.basename(img_p)
        logger.info(f"[{idx}/{len(image_paths)}] Processing: {filename}...")

        result = agent.analyze_image(img_p)

        if result.get("status") != "success":
            logger.error(f"      Status: {result.get('status').upper()} - {result.get('error', 'Unknown error')}")
            continue

        session_id = result["analysis_id"]
        stats = result["summary_statistics"]
        duration = result["total_duration_ms"]
        georef = result["georeferencing_case"]

        logger.info(f"      Session ID:     {session_id}")
        logger.info(f"      Georeferencing: Case {georef}")
        logger.info(f"      Latency:        {duration:.1f} ms")
        logger.info(f"      Candidates:     {stats.get('total_candidates', 0)} (Confirmed: {stats.get('confirmed_debris', 0)}, Suspicious: {stats.get('suspicious_anomaly', 0)}, Rejected: {stats.get('noise_rejected', 0)})")
        logger.info(f"      High Risk:      {stats.get('high_risk_count', 0)}")

        # Display target table
        if result["detections"]:
            logger.info("\n      " + "-" * 85)
            logger.info(f"      {'Target ID':<12} {'Class':<18} {'Conf':<6} {'Status':<18} {'Risk':<7} {'Coords (Lat, Lon)':<22}")
            logger.info("      " + "-" * 85)
            for d in result["detections"]:
                lat_str = f"{d.get('lat', 0):.5f}" if d.get("lat") is not None else "N/A"
                lon_str = f"{d.get('lon', 0):.5f}" if d.get("lon") is not None else "N/A"
                coord_str = f"({lat_str}, {lon_str})"
                logger.info(f"      {d.get('object_id', ''):<12} {d.get('class', ''):<18} {d.get('calibrated_confidence', 0):<6.2f} {d.get('anomaly_status', ''):<18} {d.get('risk_score', ''):<7} {coord_str:<22}")
            logger.info("      " + "-" * 85 + "\n")

        all_survey_summaries.append(result)

        # Export individual JSON report
        out_json = os.path.join(args.output_dir, f"{session_id}_report.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

    # Export aggregated CSV
    if args.export_csv and all_survey_summaries:
        csv_path = os.path.join(args.output_dir, "survey_detections_summary.csv")
        all_targets = []
        for s in all_survey_summaries:
            for d in s.get("detections", []):
                t_row = {
                    "session_id": s["analysis_id"],
                    "image_path": s["image_path"],
                    "object_id": d.get("object_id"),
                    "class": d.get("class"),
                    "calibrated_confidence": d.get("calibrated_confidence"),
                    "anomaly_status": d.get("anomaly_status"),
                    "risk_score": d.get("risk_score"),
                    "latitude": d.get("lat"),
                    "longitude": d.get("lon"),
                    "length_m": d.get("length_m"),
                    "width_m": d.get("width_m"),
                    "area_sq_m": d.get("area_sq_m"),
                    "reconstruction_error": d.get("reconstruction_error"),
                    "shadow_verified": d.get("shadow_verified"),
                    "is_rock_cluster": d.get("is_rock_cluster"),
                    "executive_narrative": d.get("explanation", {}).get("executive_narrative", "")
                }
                all_targets.append(t_row)

        if all_targets:
            fieldnames = list(all_targets[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_targets)
            logger.info(f"[CSV EXPORTED]: Aggregated detections saved to: {csv_path}")

    logger.info("\n" + "=" * 75)
    logger.info(f"[PIPELINE COMPLETED]: Successfully processed {len(all_survey_summaries)} survey image(s).")
    logger.info(f"Reports directory: {args.output_dir}")
    logger.info("=" * 75)


if __name__ == "__main__":
    args = parse_args()
    run(args)
