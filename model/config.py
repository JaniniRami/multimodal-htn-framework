"""
Configuration constants and paths for model training.
"""

from pathlib import Path

# Configuration
FOLD_TO_RUN = 0  # Select which fold to run (0-indexed: 0-4 for 5 folds)
RANDOM_SEED = 42  # Fixed seed for reproducibility across all folds
K_FOLDS = 5  # Number of folds
STRATIFY_BY = 'has_cvd'  # Field to use for stratification

# Paths
BASE_PATH = Path(__file__).parent.parent
JSON_PATH = BASE_PATH / 'data_preparation' / 'output' / 'file_mapping_apnea_only.json'
DATA_DIR = BASE_PATH / 'data' / 'Embla_Cleaned'

