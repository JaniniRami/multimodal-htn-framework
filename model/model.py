"""
Model training script with patient-wise stratified k-fold split.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from data_preparation
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.split import get_patient_wise_stratified_kfold_splits

from config import (
    FOLD_TO_RUN,
    RANDOM_SEED,
    K_FOLDS,
    STRATIFY_BY,
    JSON_PATH,
    DATA_DIR
)
from data_loader import load_data_from_entries
from normalization import (
    calculate_normalization_coefficients,
    apply_normalization
)
from dataset import create_datasets_from_splits
from torch.utils.data import DataLoader


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
    
    # Calculate normalization coefficients from training data only (no data leakage)
    print("\nCalculating normalization coefficients from training data...")
    norm_coefficients = calculate_normalization_coefficients(train_data['bags'])
    
    # Print normalization coefficients
    print(f"\nNormalization Coefficients:")
    print(f"  ECG segments: mean={norm_coefficients['ecg_segments']['mean']:.6f}, std={norm_coefficients['ecg_segments']['std']:.6f}")
    print(f"  PPG segments: mean={norm_coefficients['ppg_segments']['mean']:.6f}, std={norm_coefficients['ppg_segments']['std']:.6f}")
    print(f"  ECG HRV - hrv_hf: mean={norm_coefficients['ecg_hrv']['hrv_hf']['mean']:.6f}, std={norm_coefficients['ecg_hrv']['hrv_hf']['std']:.6f}")
    print(f"  ECG HRV - hrv_lf: mean={norm_coefficients['ecg_hrv']['hrv_lf']['mean']:.6f}, std={norm_coefficients['ecg_hrv']['hrv_lf']['std']:.6f}")
    print(f"  PPG HRV - hrv_hf: mean={norm_coefficients['ppg_hrv']['hrv_hf']['mean']:.6f}, std={norm_coefficients['ppg_hrv']['hrv_hf']['std']:.6f}")
    print(f"  PPG HRV - hrv_lf: mean={norm_coefficients['ppg_hrv']['hrv_lf']['mean']:.6f}, std={norm_coefficients['ppg_hrv']['hrv_lf']['std']:.6f}")
    
    # Apply normalization to training and test data
    print("\nApplying normalization to training data...")
    train_data['bags'] = apply_normalization(train_data['bags'], norm_coefficients)
    
    print("Applying normalization to test data...")
    test_data['bags'] = apply_normalization(test_data['bags'], norm_coefficients)
    
    print("Normalization complete!")
    
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
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, test_dataset = create_datasets_from_splits(
        train_bags=train_data['bags'],
        test_bags=test_data['bags'],
        train_entries=selected_split['train_entries'],
        test_entries=selected_split['test_entries'],
        data_dir=str(DATA_DIR),
        augment_train=True,  # Enable augmentation for training
        max_instances=120,
        min_instances=15,
        ecg_fs=200,
        ppg_fs=100,
    )
    
    print(f"Train dataset: {len(train_dataset)} bags")
    print(f"Test dataset: {len(test_dataset)} bags")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print(f"{'='*80}\n")
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'norm_coefficients': norm_coefficients
    }


if __name__ == '__main__':
    main()

