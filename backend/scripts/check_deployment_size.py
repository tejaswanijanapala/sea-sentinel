import os
import sys
from shared.utils.logger import get_logger
logger = get_logger(__name__)



def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

def format_mb(size_bytes):
    return size_bytes / (1024 * 1024)

def main():
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Define categories based on prompt
    categories = {
        "Source code": ["agent", "ai", "api", "app", "authentication", "configs", "database", "edge", "evaluation", "inference", "risk", "ros2", "shared", "tests", "training", "visualization", "scripts", "*.py"],
        "Models": ["models"],
        "Dependencies (Virtual Env)": ["../.venv"], # assuming venv is one level up based on previous context
        "Persistent Data (Uploads/DB/Preprocessed/Datasets)": ["outputs", "datasets", "runs", "scratch", "logs"]
    }
    
    sizes = {"Source code": 0, "Models": 0, "Dependencies (Virtual Env)": 0, "Persistent Data (Uploads/DB/Preprocessed/Datasets)": 0, "Other": 0}
    
    # Calculate sizes
    for root, dirs, files in os.walk(backend_dir):
        # We will do a simpler calculation: just sum known heavy folders vs rest
        pass
        
    models_size = get_dir_size(os.path.join(backend_dir, "models"))
    # The yolo11n.pt is in backend root currently
    for f in os.listdir(backend_dir):
        if f.endswith('.pt'):
            models_size += os.path.getsize(os.path.join(backend_dir, f))
            
    outputs_size = get_dir_size(os.path.join(backend_dir, "outputs"))
    datasets_size = get_dir_size(os.path.join(backend_dir, "datasets"))
    logs_size = get_dir_size(os.path.join(backend_dir, "logs"))
    persistent_size = outputs_size + datasets_size + logs_size
    
    total_backend_size = get_dir_size(backend_dir)
    source_size = total_backend_size - persistent_size - models_size
    
    # Dependencies
    venv_dir = os.path.abspath(os.path.join(backend_dir, "..", ".venv"))
    deps_size = get_dir_size(venv_dir)

    deployable_size = source_size + models_size + deps_size

    logger.info("============================================================")
    logger.info("DEPLOYMENT FOOTPRINT AUDIT")
    logger.info("============================================================")
    logger.info(f"Source code:       {format_mb(source_size):.2f} MB")
    logger.info(f"Dependencies:      {format_mb(deps_size):.2f} MB")
    logger.info(f"Models:            {format_mb(models_size):.2f} MB")
    logger.info("------------------------------------------------------------")
    logger.info(f"DEPLOYABLE TOTAL:  {format_mb(deployable_size):.2f} MB")
    logger.info("============================================================")
    logger.info(f"Persistent Data:   {format_mb(persistent_size):.2f} MB (EXCLUDED FROM DEPLOYMENT)")
    logger.info("============================================================")
    
    if deployable_size < 512:
        logger.info("STATUS: PASS (<512 MB)")
        if deployable_size < 450:
            logger.info("TARGET REACHED: <450 MB Preferred Target achieved!")
    else:
        logger.error("STATUS: FAIL (>=512 MB)")

if __name__ == "__main__":
    main()
