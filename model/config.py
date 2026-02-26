"""
Configuration constants and paths for model training.
"""

from pathlib import Path

# Validation mode: 'kfold' or 'loso'
VALIDATION_MODE = 'loso'  # Options: 'kfold' or 'loso' (Leave-One-Subject-Out)

# Configuration for K-Fold (only used if VALIDATION_MODE == 'kfold')
FOLD_TO_RUN = 1  # Select which fold to run (0-indexed: 0-4 for 5 folds)
RANDOM_SEED = 42  # Fixed seed for reproducibility across all folds
K_FOLDS = 5  # Number of folds
STRATIFY_BY = 'has_cvd'  # Field to use for stratification
# Configuration for LOSO (only used if VALIDATION_MO`DE == 'loso')
# Set to None to run all folds, or set to a number (1-indexed) 
# to run only that fold
# Example: PATIENT_TO_RUN = 9  # Runs only fold 9 (the 9th patient)
PATIENT_TO_RUN = None  # Select which patient fold to run (1-indexed, None = run all patients)

# Model architecture selection
MODEL_TYPE = 'ppg'  # Options: 'fusion', 'ecg', 'ppg', 'fusion_pretrained'

# Number of GPUs to use for training (1 = single GPU, no DataParallel; >1 = DataParallel across that many GPUs)
NUM_GPUS = 4

# Paths
BASE_PATH = Path(__file__).parent.parent
JSON_PATH = BASE_PATH / 'data_preparation' / 'output' / 'file_mapping_apnea_only.json'
DATA_DIR = BASE_PATH / 'data' / 'Embla_Cleaned'

# Base directory where all per-run training outputs (checkpoints, text logs, etc.)
# will be organized. Each run will create a subfolder inside this directory.
RESULTS_BASE_DIR = BASE_PATH / 'training_results'
