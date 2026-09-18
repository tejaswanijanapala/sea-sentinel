import json
import urllib.request
import time
from shared.utils.logger import get_logger
logger = get_logger(__name__)



def run_benchmark():
    samples_res = urllib.request.urlopen('http://localhost:8000/api/samples')
    samples_data = json.loads(samples_res.read().decode('utf-8'))
    samples = samples_data.get('samples', [])
    
    if not samples:
        logger.info("No samples found!")
        return

    sample_path = samples[0]['path']
    logger.info(f"Testing sample: {samples[0]['name']}")
    logger.info("-" * 60)

    for mode in ['fast', 'balanced', 'high_accuracy']:
        payload = json.dumps({'image_path': sample_path, 'mode': mode}).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:8000/api/analyze',
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        t0 = time.time()
        res = urllib.request.urlopen(req)
        wall_time = time.time() - t0
        
        out = json.loads(res.read().decode('utf-8'))
        prof = out.get('profiling', {})
        b_info = prof.get('bottleneck', {})
        
        logger.info(f"[{mode.upper()} MODE]")
        logger.info(f"  Status:             {out.get('status')}")
        logger.info(f"  Server Profiler:    {prof.get('total_duration_seconds', 0.0):.2f}s")
        logger.info(f"  Wall-Clock Time:    {wall_time:.2f}s")
        logger.info(f"  20s Budget Status:  {prof.get('budget_status', 'PASS')} ({prof.get('headroom_seconds', 0.0):.2f}s headroom)")
        logger.info(f"  Total Detections:   {len(out.get('detections', []))}")
        logger.info(f"  Slowest Component:  {b_info.get('stage')} ({b_info.get('duration_seconds', 0.0):.2f}s, {b_info.get('percentage', 0.0)}%)")
        logger.info(f"  Stage Timings:")
        for stg, ms in prof.get('stages_ms', {}).items():
            logger.info(f"    - {stg:24s}: {ms:7.2f} ms")
        logger.info("-" * 60)

if __name__ == "__main__":
    run_benchmark()
