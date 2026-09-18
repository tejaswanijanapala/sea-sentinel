"""
Script to verify all Adaptive Learning & Error Prevention REST API endpoints:
- GET /api/learning/dashboard
- POST /api/learning/review
- GET /api/learning/active-queue
- GET /api/learning/error-memory
- GET /api/learning/unknown-classes
- POST /api/learning/train
- GET /api/learning/champion-challenger
- POST /api/learning/deploy
- POST /api/learning/rollback
"""

import json
import urllib.request
import urllib.error
from shared.utils.logger import get_logger
logger = get_logger(__name__)



BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, payload=None):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload else None,
        headers={"Content-Type": "application/json"} if payload else {},
        method=method
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            data = json.loads(res_body)
            logger.info(f"[PASS] {method} {path} -> {response.status}")
            return data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        logger.error(f"[FAIL] {method} {path} -> HTTP {e.code}: {err_body}")
        return None
    except Exception as e:
        logger.error(f"[ERROR] {method} {path} -> {e}")
        return None

if __name__ == "__main__":
    logger.info("=== Testing Sea Sentinel Adaptive Learning REST API ===")
    
    # 1. Dashboard
    dash = test_endpoint("GET", "/api/learning/dashboard")
    
    # 2. Structured Review Submission
    review_payload = {
        "analysis_id": "SURVEY_DEMO_01",
        "object_id": "TARGET_101",
        "review_type": "FALSE_POSITIVE",
        "predicted_class": "fishing_net",
        "correct_class": "rock",
        "reviewer_id": "Hydrographer_Alpha",
        "reviewer_confidence": 0.95,
        "reviewer_comment": "Natural granite boulder acoustic texture, not nylon fishing net.",
        "model_name": "yolo_detector",
        "model_version": "v3.2",
        "predicted_confidence": 0.92,
        "bbox": [120, 140, 210, 260]
    }
    rev_res = test_endpoint("POST", "/api/learning/review", review_payload)
    
    # 3. Active Queue
    active_q = test_endpoint("GET", "/api/learning/active-queue?limit=10")
    
    # 4. Error Memory
    err_mem = test_endpoint("GET", "/api/learning/error-memory?limit=10")
    
    # 5. Unknown Classes
    unk_cls = test_endpoint("GET", "/api/learning/unknown-classes")
    
    # 6. Train YOLO Challenger
    train_res = test_endpoint("POST", "/api/learning/train", {"model_type": "yolo", "epochs": 2, "batch_size": 2})
    
    # 7. Champion vs Challenger
    champ_chal = test_endpoint("GET", "/api/learning/champion-challenger?model_type=yolo")
    
    # 8. Deploy Approved Challenger
    deploy_res = test_endpoint("POST", "/api/learning/deploy", {"model_type": "yolo"})
    
    # 9. Rollback to Stable Champion
    rollback_res = test_endpoint("POST", "/api/learning/rollback", {"model_type": "yolo"})
    
    # 10. Dashboard refreshed
    dash_post = test_endpoint("GET", "/api/learning/dashboard")

    logger.info("=== All REST API Tests Completed ===")
