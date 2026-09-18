import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.tests.test_imo_risk_engine import (
    engine as get_engine,
    test_scenario_1_large_floating_container_near_shipping_lane,
    test_scenario_2_small_plastic_in_marine_protected_area,
    test_scenario_3_large_ghost_fishing_net,
    test_scenario_4_mobile_floating_debris_with_future_exposure,
    test_scenario_5_low_confidence_high_uncertainty_corridor_overlap,
    test_mathematical_determinism_and_base_risk_equation
)

from shared.utils.logger import get_logger
logger = get_logger(__name__)

def run():
    eng = get_engine()
    logger.info("Testing Scenario 1: Large Floating Container...")
    test_scenario_1_large_floating_container_near_shipping_lane(eng)
    logger.info("PASSED Scenario 1!")

    logger.info("Testing Scenario 2: Small Plastic in MPA...")
    test_scenario_2_small_plastic_in_marine_protected_area(eng)
    logger.info("PASSED Scenario 2!")

    logger.info("Testing Scenario 3: Large Ghost Fishing Net...")
    test_scenario_3_large_ghost_fishing_net(eng)
    logger.info("PASSED Scenario 3!")

    logger.info("Testing Scenario 4: Drifting Debris Dynamic Exposure...")
    test_scenario_4_mobile_floating_debris_with_future_exposure(eng)
    logger.info("PASSED Scenario 4!")

    logger.info("Testing Scenario 5: Uncertain Target Corridor Overlap...")
    test_scenario_5_low_confidence_high_uncertainty_corridor_overlap(eng)
    logger.info("PASSED Scenario 5!")

    logger.info("Testing Scenario 6: Determinism & Base Risk Equations...")
    test_mathematical_determinism_and_base_risk_equation(eng)
    logger.info("PASSED Scenario 6!")

    logger.info("\nALL 6 IMO RISK SCENARIO TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run()
