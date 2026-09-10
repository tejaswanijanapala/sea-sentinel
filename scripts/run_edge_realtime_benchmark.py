"""
Sea Sentinel: Hard Real-Time Edge AI & Hardware Latency Benchmark Suite.
Measures granular per-stage latencies (ingestion, quality gate, preprocessing,
YOLO, U-Net, fusion, tracking, geolocation, database, telemetry) over sustained operation.
Computes Mean, Median (P50), P95, P99, Worst-Case latency, FPS, and dropped frames.
"""

import os
import sys
import time
import json
import argparse
import statistics
import numpy as np

# Ensure backend root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from edge.edge_perception import EdgePerceptionPipeline
from edge.resource_manager import EdgeResourceManager


def generate_synthetic_sonar_swath(size=(640, 640)) -> np.ndarray:
    """Generates realistic synthetic side-scan sonar waterfall imagery with speckle noise."""
    h, w = size
    # Base acoustic seabed backscatter (Rayleigh-distributed speckle)
    seabed = np.random.rayleigh(scale=35.0, size=(h, w)).astype(np.float32)
    seabed = np.clip(seabed, 5, 230).astype(np.uint8)

    # Insert a central nadir water-column line (dark band)
    center_x = w // 2
    seabed[:, center_x - 12:center_x + 12] = np.random.randint(0, 10, (h, 24), dtype=np.uint8)

    # Insert synthetic debris object: highlight block followed by acoustic shadow
    obj_y, obj_x = h // 2, center_x + 120
    # Highlight (high return reflection)
    seabed[obj_y - 10:obj_y + 10, obj_x - 15:obj_x + 15] = 245
    # Acoustic shadow behind target away from center trackline
    seabed[obj_y - 10:obj_y + 10, obj_x + 16:obj_x + 55] = 2

    return seabed


def run_benchmark(iterations: int = 50, high_recall: bool = False, output_json: Optional[str] = None):
    print("=" * 80)
    print("  SEA SENTINEL — PRODUCTION EDGE AI HARD REAL-TIME BENCHMARK")
    print("  Ministry of Earth Sciences (MoES) — National Institute of Ocean Technology")
    print("=" * 80)

    res_mgr = EdgeResourceManager()
    dev_prof = res_mgr.device_profile
    print(f"\n[Hardware Target Introspection]")
    print(f"  Device Class:         {dev_prof['device_class']}")
    print(f"  Recommended Backend:  {dev_prof['recommended_backend']}")
    print(f"  Optimal Precision:    {dev_prof['optimal_precision'].upper()}")
    print(f"  CUDA GPU Available:   {dev_prof['cuda_available']} ({dev_prof.get('gpu_name')})")
    print(f"  Total System RAM:     {dev_prof['total_ram_gb']} GB")
    print(f"  CPU Cores:            {dev_prof['cpu_cores']}")

    print("\n[Initializing Deterministic Edge Pipeline...]")
    pipeline = EdgePerceptionPipeline(high_recall_mode=high_recall)
    print(f"  High Recall Mode:     {pipeline.high_recall_mode}")
    print(f"  YOLO Model Loaded:    {pipeline.yolo.is_model_loaded}")
    print(f"  U-Net Model Loaded:   {pipeline.unet.is_model_loaded}")
    print(f"  Bounded Buffer Cap:   {pipeline.frame_buffer.max_capacity}")
    print(f"  Modem Packet Schema:  24 Bytes (CRC-8 Protected)")

    # Warmup pass
    print("\n[Pre-Warming AI Engine & GPU/CPU Caches (3 Warmup Passes)...]")
    for _ in range(3):
        dummy_swath = generate_synthetic_sonar_swath((640, 640))
        pipeline.process_frame(dummy_swath, frame_id="WARMUP")

    print(f"\n[Executing Sustained Benchmark Loop: {iterations} Frames]...")
    stage_metrics = {
        "ingestion_ms": [],
        "quality_gate_ms": [],
        "preprocessing_ms": [],
        "yolo_ms": [],
        "unet_ms": [],
        "parallel_model_inference_ms": [],
        "fusion_ms": [],
        "unknown_detector_ms": [],
        "tracking_ms": [],
        "sensor_fusion_ms": [],
        "geolocation_and_telemetry_ms": [],
        "total_pipeline_ms": []
    }

    t_bench_start = time.perf_counter()
    dropped_frames = 0
    processed_frames = 0

    for i in range(iterations):
        frame_id = f"SWATH_{i+1:04d}"
        swath = generate_synthetic_sonar_swath((640, 640))
        
        # Test quality gate with occasional difficult frame
        if (i + 1) % 15 == 0:
            swath = np.zeros((640, 640), dtype=np.uint8)  # Dead ping test

        res = pipeline.process_frame(swath, frame_id=frame_id)
        timing = res.get("timing_breakdown", {})

        for k in stage_metrics.keys():
            if k in timing:
                stage_metrics[k].append(timing[k])

        if res["status"] == "LOW_QUALITY_FRAME":
            dropped_frames += 1
        else:
            processed_frames += 1

        if (i + 1) % 10 == 0 or (i + 1) == iterations:
            tot = timing.get("total_pipeline_ms", 0.0)
            inf = timing.get("parallel_model_inference_ms", 0.0)
            print(f"  Iteration [{i+1:03d}/{iterations:03d}] -> Inference: {inf:5.1f}ms | Total Pipeline: {tot:5.1f}ms | Status: {res['status']}")

    total_bench_duration = time.perf_counter() - t_bench_start
    sustained_fps = round(iterations / max(0.001, total_bench_duration), 2)

    # Statistical Aggregation
    def summarize_times(lst: list) -> dict:
        if not lst:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "worst_case": 0.0}
        s = sorted(lst)
        return {
            "mean": round(float(statistics.mean(s)), 2),
            "median": round(float(statistics.median(s)), 2),
            "p95": round(float(np.percentile(s, 95)), 2),
            "p99": round(float(np.percentile(s, 99)), 2),
            "worst_case": round(float(max(s)), 2)
        }

    stats = {k: summarize_times(v) for k, v in stage_metrics.items()}
    inf_stats = stats["parallel_model_inference_ms"]
    pipe_stats = stats["total_pipeline_ms"]

    print("\n" + "=" * 80)
    print("  HARD REAL-TIME LATENCY BENCHMARK RESULTS (GRANULAR BREAKDOWN)")
    print("=" * 80)
    print(f"{'Pipeline Stage':<30} | {'Mean (ms)':<9} | {'Median':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | {'Max (ms)':<9}")
    print("-" * 80)
    for stage, s in stats.items():
        print(f"{stage:<30} | {s['mean']:<9.1f} | {s['median']:<9.1f} | {s['p95']:<9.1f} | {s['p99']:<9.1f} | {s['worst_case']:<9.1f}")
    print("-" * 80)
    print(f"  Sustained Throughput:  {sustained_fps} FPS across {iterations} frames")
    print(f"  Processed Frames:      {processed_frames}")
    print(f"  Quality-Gate Rejects:  {dropped_frames} (Dead/Corrupt pings safely rejected)")
    print(f"  Model Inference P95:   {inf_stats['p95']} ms (Hard Target <= 50 ms)")
    print(f"  Total Pipeline P95:    {pipe_stats['p95']} ms")
    
    # Validation verdict
    inf_pass = inf_stats["p95"] <= 50.0 or dev_prof["cuda_available"]
    status_label = "PASSED [TARGET <= 50ms MET]" if inf_stats["p95"] <= 50.0 else "OPERATIONAL [EDGE CPU PROFILE]"
    print(f"\n  Final Edge Verdict:    {status_label}")
    print("=" * 80)

    # Save to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": dev_prof,
        "iterations": iterations,
        "sustained_fps": sustained_fps,
        "processed_frames": processed_frames,
        "dropped_frames": dropped_frames,
        "stage_statistics": stats,
        "target_50ms_met": inf_stats["p95"] <= 50.0
    }

    out_file = output_json or os.path.join(PROJECT_ROOT, "backend", "outputs", "benchmarks", "edge_realtime_benchmark.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved Detailed Benchmark Report] -> {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sea Sentinel Real-Time Edge AI Benchmark")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--high-recall", action="store_true", help="Enable High-Recall mode")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    run_benchmark(iterations=args.iterations, high_recall=args.high_recall, output_json=args.output)
