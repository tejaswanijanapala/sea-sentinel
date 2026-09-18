import sys
from pathlib import Path
from shared.utils.logger import get_logger
logger = get_logger(__name__)



aliases = {
    'debris_detection': 'debris-detection',
    'sonar_image_processing': 'sonar-image-processing',
    'debris_risk_scoring': 'debris-risk-scoring',
    'natural_manmade_classification': 'natural-manmade-classification',
    'duplicate_detection': 'duplicate-detection',
    'debris_density': 'debris-density',
    'sonar_quality': 'sonar-quality'
}

for alias, folder in aliases.items():
    alias_dir = Path('backend') / alias
    alias_dir.mkdir(parents=True, exist_ok=True)
    target_dir = (Path('backend') / folder).resolve()
    with open(alias_dir / '__init__.py', 'w', encoding='utf-8') as f:
        f.write(f'# Package bridge for {folder}\nimport sys\nfrom pathlib import Path\n_tgt = r"{target_dir}"\nif _tgt not in sys.path:\n    sys.path.insert(0, _tgt)\n__path__ = [_tgt]\n')

logger.info('Bridge packages configured.')
