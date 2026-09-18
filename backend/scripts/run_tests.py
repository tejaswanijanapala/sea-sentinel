"""Sea Sentinel Automated Test Runner."""
import sys
import unittest
from pathlib import Path
from shared.utils.logger import get_logger
logger = get_logger(__name__)



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main():
    logger.info("=" * 70)
    logger.info("           SEA SENTINEL 2.0 - AUTOMATED TEST SUITE")
    logger.info("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(BACKEND_DIR / "tests"), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    logger.info("=" * 70)
    if result.wasSuccessful():
        logger.info(f"ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        logger.error(f"TEST RUN COMPLETED: {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
