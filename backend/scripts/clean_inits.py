"""Clean package __init__.py files in backend."""
from pathlib import Path
from shared.utils.logger import get_logger
logger = get_logger(__name__)



packages = [
    'debris_detection',
    'sonar_image_processing',
    'debris_risk_scoring',
    'natural_manmade_classification',
    'duplicate_detection',
    'debris_density',
    'sonar_quality',
    'review_intelligence'
]

for pkg in packages:
    init_file = Path('backend') / pkg / '__init__.py'
    init_file.parent.mkdir(parents=True, exist_ok=True)
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(f'"""{pkg} package."""\n')
    logger.info(f'Cleaned {init_file}')
