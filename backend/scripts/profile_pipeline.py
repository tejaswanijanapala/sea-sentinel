"""
Sea Sentinel Pipeline Profiler & Benchmark
Profiles the end-to-end execution time of every pipeline stage across multiple sample images.
"""
import os
import sys
import time
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.orchestrator import SIHPipelineAgent
from shared.utils.logger import get_logger
logger = get_logger(__name__)




def run_benchmark():
    agent = SIHPipelineAgent()
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "samples")
    
    sample_files = [
        "noaa_h11584_gulf_sample.tif",
        "usgs_14bim05_breton_sample.tif",
        "china_offshore_dongying_pipeline.jpg",
        "china_offshore_dongying_engine.jpg",
        "china_offshore_quanzhou_net.jpg",
        "towfish_mission_case_b.png"
    ]
    
    logger.info("=" * 70)
    logger.info("SEA SENTINEL PIPELINE PERFORMANCE BENCHMARK")
    logger.info(f"Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 70)
    
    results = []
    
    for fname in sample_files:
        fpath = os.path.join(samples_dir, fname)
        if not os.path.exists(fpath):
            continue
            
        logger.info(f"\nEvaluating: {fname}")
        t0 = time.perf_counter()
        res = agent.analyze_image(fpath)
        total_time = time.perf_counter() - t0
        
        trace = {item["stage"]: item["duration_ms"] for item in res.get("execution_trace", [])}
        target_count = len(res.get("detections", []))
        
        logger.info(f"  Total Latency: {total_time:.2f}s ({total_time*1000:.1f} ms) | Targets Fused: {target_count}")
        for stage, dur in trace.items():
            logger.info(f"    - {stage:25s}: {dur:8.2f} ms")
            
        results.append({
            "filename": fname,
            "total_sec": round(total_time, 2),
            "target_count": target_count,
            "trace": trace
        })
        
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    avg_lat = sum(r["total_sec"] for r in results) / max(1, len(results))
    logger.info(f"Average Pipeline Latency: {avg_lat:.2f} seconds across {len(results)} datasets.")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_benchmark()
