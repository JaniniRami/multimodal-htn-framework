"""
Model training script with patient-wise stratified k-fold split.
"""

import sys
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np

# Add parent directory to path to import from data_preparation
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.split import get_patient_wise_stratified_kfold_splits


# Configuration
FOLD_TO_RUN = 0  # Select which fold to run (0-indexed: 0-4 for 5 folds)
RANDOM_SEED = 42  # Fixed seed for reproducibility across all folds
K_FOLDS = 5  # Number of folds
STRATIFY_BY = 'has_cvd'  # Field to use for stratification

# Paths
BASE_PATH = Path(__file__).parent.parent
JSON_PATH = BASE_PATH / 'data_preparation' / 'output' / 'file_mapping_apnea_only.json'
DATA_DIR = BASE_PATH / 'data' / 'Embla_Cleaned'


def get_folder_name(entry):
    """
    Get the folder name for a given entry.
    
    Args:
        entry: Dictionary entry with 'patient_id' and 'follow_up'
    
    Returns:
        str: Folder name (patient_id_followup or just patient_id for baseline)
    """
    patient_id = entry.get('patient_id')
    follow_up = entry.get('follow_up')
    
    if follow_up is None:
        return patient_id
    else:
        return f"{patient_id}_{follow_up}"


def load_bags_from_folder(folder_path):
    """
    Load all bag files from a folder.
    
    Args:
        folder_path: Path to the patient folder
    
    Returns:
        list: List of dictionaries, each containing bag data and metadata
    """
    bags = []
    
    if not folder_path.exists():
        return bags
    
    # Find all h5 files in the folder
    h5_files = sorted(folder_path.glob('*.h5'))
    
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                # Load sleep_stages (vlen=str dtype, h5py handles decoding)
                sleep_stages_data = f['sleep_stages'][:]
                sleep_stages = [s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                               for s in sleep_stages_data]
                
                bag_data = {
                    'file_path': str(h5_file),
                    'ecg_segments': np.array(f['ecg_segments']),
                    'ppg_segments': np.array(f['ppg_segments']),
                    'apnea_labels': np.array(f['apnea_labels']),
                    'sleep_stages': sleep_stages
                }
                bags.append(bag_data)
        except Exception as e:
            print(f"Warning: Could not load {h5_file}: {e}")
    
    return bags


def load_data_from_entries(entries, data_dir):
    """
    Load all bags for a list of entries.
    
    Args:
        entries: List of entry dictionaries
        data_dir: Base directory containing patient folders
    
    Returns:
        dict: Dictionary with 'bags', 'baseline_files', 'followup_files', 'total_bags'
    """
    # Group entries by folder to avoid loading duplicates
    folder_to_entries = defaultdict(list)
    for entry in entries:
        folder_name = get_folder_name(entry)
        folder_to_entries[folder_name].append(entry)
    
    all_bags = []
    baseline_count = 0
    followup_count = 0
    
    for folder_name, folder_entries in folder_to_entries.items():
        folder_path = data_dir / folder_name
        
        # Load bags from this folder
        bags = load_bags_from_folder(folder_path)
        all_bags.extend(bags)
        
        # Count baseline vs followup files
        for entry in folder_entries:
            if entry.get('follow_up') is None:
                baseline_count += 1
            else:
                followup_count += 1
    
    return {
        'bags': all_bags,
        'baseline_files': baseline_count,
        'followup_files': followup_count,
        'total_bags': len(all_bags)
    }


def main():
    """Main function to get split and load data."""
    
    # Validate fold selection
    if FOLD_TO_RUN < 0 or FOLD_TO_RUN >= K_FOLDS:
        print(f"Error: FOLD_TO_RUN must be between 0 and {K_FOLDS - 1}")
        return
    
    # Check if JSON file exists
    if not JSON_PATH.exists():
        print(f"Error: JSON file not found at {JSON_PATH}")
        return
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found at {DATA_DIR}")
        return
    
    print(f"Loading splits with seed={RANDOM_SEED}, k={K_FOLDS}, stratify_by={STRATIFY_BY}")
    print(f"Selected fold: {FOLD_TO_RUN} (0-indexed)\n")
    
    # Get all splits
    splits = get_patient_wise_stratified_kfold_splits(
        json_path=str(JSON_PATH),
        k=K_FOLDS,
        random_state=RANDOM_SEED,
        stratify_by=STRATIFY_BY
    )
    
    # Get the selected fold
    selected_split = splits[FOLD_TO_RUN]
    
    # Print patients in the selected fold
    print(f"{'='*80}")
    print(f"FOLD {FOLD_TO_RUN} - PATIENT LISTS")
    print(f"{'='*80}")
    
    print(f"\nTrain Patients ({len(selected_split['train_patient_ids'])}):")
    print(selected_split['train_patient_ids'])
    
    print(f"\nTest Patients ({len(selected_split['test_patient_ids'])}):")
    print(selected_split['test_patient_ids'])
    
    print(f"\nTrain Entries: {len(selected_split['train_entries'])} files")
    print(f"Test Entries: {len(selected_split['test_entries'])} files")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading training data...")
    train_data = load_data_from_entries(selected_split['train_entries'], DATA_DIR)
    
    print("Loading test data...")
    test_data = load_data_from_entries(selected_split['test_entries'], DATA_DIR)
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"DATA STATISTICS - FOLD {FOLD_TO_RUN}")
    print(f"{'='*80}")
    
    print(f"\nTRAINING SET:")
    print(f"  Baseline files: {train_data['baseline_files']}")
    print(f"  Follow-up files: {train_data['followup_files']}")
    print(f"  Total files: {train_data['baseline_files'] + train_data['followup_files']}")
    print(f"  Total bags: {train_data['total_bags']}")
    
    print(f"\nTEST SET:")
    print(f"  Baseline files: {test_data['baseline_files']}")
    print(f"  Follow-up files: {test_data['followup_files']}")
    print(f"  Total files: {test_data['baseline_files'] + test_data['followup_files']}")
    print(f"  Total bags: {test_data['total_bags']}")
    
    print(f"\nTOTAL:")
    total_baseline = train_data['baseline_files'] + test_data['baseline_files']
    total_followup = train_data['followup_files'] + test_data['followup_files']
    total_bags = train_data['total_bags'] + test_data['total_bags']
    print(f"  Baseline files: {total_baseline}")
    print(f"  Follow-up files: {total_followup}")
    print(f"  Total files: {total_baseline + total_followup}")
    print(f"  Total bags: {total_bags}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

