"""
- patient-wise stratified k-fold split
"""


import json
import csv
from pathlib import Path
from collections import defaultdict
import random


K_FOLDS = 5

def load_file_mapping(json_path):
    """
    Load the file mapping JSON and organize by patient ID.
    
    Args:
        json_path: Path to file_mapping_apnea_only.json
    
    Returns:
        dict: Dictionary keyed by patient_id, values are sets of follow-up numbers
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dictionary to store follow-ups per patient
    # Key: patient_id, Value: set of follow-up numbers (None for baseline, 1, 2, etc.)
    patients_followups = defaultdict(set)
    
    for entry in data.get('valid_mappings', []):
        patient_id = entry.get('patient_id')
        follow_up = entry.get('follow_up')
        
        if patient_id:
            patients_followups[patient_id].add(follow_up)
    
    return patients_followups


def generate_patients_csv(patients_followups, output_path):
    """
    Generate CSV file with patient IDs and their follow-ups.
    
    Args:
        patients_followups: Dictionary keyed by patient_id with sets of follow-up numbers
        output_path: Path to save the CSV file
    """
    # Find the maximum follow-up number across all patients
    max_followup = 0
    has_baseline = False
    
    for followups in patients_followups.values():
        for fu in followups:
            if fu is None:
                has_baseline = True
            elif isinstance(fu, int) and fu > max_followup:
                max_followup = fu
    
    # Create column headers
    headers = ['patient_id']
    if has_baseline:
        headers.append('Baseline')
    
    # Add follow-up columns
    for i in range(1, max_followup + 1):
        headers.append(f'Follow Up {i}')
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Sort patients by ID for consistent output
        for patient_id in sorted(patients_followups.keys()):
            row = [patient_id]
            followups = patients_followups[patient_id]
            
            # Add baseline column
            if has_baseline:
                row.append('Yes' if None in followups else '')
            
            # Add follow-up columns
            for i in range(1, max_followup + 1):
                row.append('Yes' if i in followups else '')
            
            writer.writerow(row)
    
    print(f"CSV file generated: {output_path}")
    print(f"Total patients: {len(patients_followups)}")
    print(f"Maximum follow-up number: {max_followup}")
    print(f"Has baseline entries: {has_baseline}")


def get_patient_wise_stratified_kfold_splits(json_path, k=5, random_state=42, stratify_by='has_cvd'):
    """
    Perform patient-wise stratified k-fold split.
    
    This function groups all files by patient_id and ensures that all files from
    the same patient are in the same fold (train or test). The split is stratified
    based on a specified label (default: has_cvd).
    
    Args:
        json_path: Path to file_mapping_apnea_only.json
        k: Number of folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
        stratify_by: Field to use for stratification. Options: 'has_cvd', 'has_apnea' (default: 'has_cvd')
    
    Returns:
        list: List of k dictionaries, each containing:
              - 'train_patient_ids': List of patient IDs in training set
              - 'test_patient_ids': List of patient IDs in test set
              - 'train_entries': List of all file entries (dicts) in training set
              - 'test_entries': List of all file entries (dicts) in test set
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group entries by patient_id and collect stratification labels
    patient_data = defaultdict(lambda: {'entries': [], 'label': None})
    
    for entry in data.get('valid_mappings', []):
        patient_id = entry.get('patient_id')
        if not patient_id:
            continue
        
        patient_data[patient_id]['entries'].append(entry)
        
        # Get stratification label (use first entry's label for consistency)
        if patient_data[patient_id]['label'] is None:
            label = entry.get(stratify_by, 'unknown')
            patient_data[patient_id]['label'] = label
    
    # Group patients by label for stratification
    patients_by_label = defaultdict(list)
    for pid, data_dict in patient_data.items():
        label = str(data_dict['label']) if data_dict['label'] is not None else 'unknown'
        patients_by_label[label].append(pid)
    
    # Shuffle patients within each label group (for reproducibility with fixed seed)
    for label in patients_by_label:
        random.shuffle(patients_by_label[label])
    
    # Perform stratified k-fold split manually
    splits = []
    
    for fold in range(k):
        train_patient_ids = []
        test_patient_ids = []
        
        # For each label group, assign patients to folds
        for label, patient_list in patients_by_label.items():
            n_patients = len(patient_list)
            fold_size = n_patients // k
            remainder = n_patients % k
            
            # Calculate start and end indices for test set
            start_idx = fold * fold_size + min(fold, remainder)
            end_idx = start_idx + fold_size + (1 if fold < remainder else 0)
            
            # Split into train and test
            test_pids = patient_list[start_idx:end_idx]
            train_pids = patient_list[:start_idx] + patient_list[end_idx:]
            
            train_patient_ids.extend(train_pids)
            test_patient_ids.extend(test_pids)
        
        # Sort for consistency
        train_patient_ids = sorted(train_patient_ids)
        test_patient_ids = sorted(test_patient_ids)
        
        # Collect all entries for train and test sets
        train_entries = []
        test_entries = []
        
        for pid in train_patient_ids:
            train_entries.extend(patient_data[pid]['entries'])
        
        for pid in test_patient_ids:
            test_entries.extend(patient_data[pid]['entries'])
        
        splits.append({
            'train_patient_ids': train_patient_ids,
            'test_patient_ids': test_patient_ids,
            'train_entries': train_entries,
            'test_entries': test_entries
        })
    
    return splits


def print_split_statistics(splits, stratify_by='has_cvd'):
    """
    Print statistics about the k-fold splits.
    
    Args:
        splits: List of split dictionaries from get_patient_wise_stratified_kfold_splits
        stratify_by: Field used for stratification (for display purposes)
    """
    print(f"\n{'='*80}")
    print(f"STRATIFIED K-FOLD SPLIT STATISTICS (stratified by: {stratify_by})")
    print(f"{'='*80}")
    
    for fold_idx, split in enumerate(splits, 1):
        train_pids = split['train_patient_ids']
        test_pids = split['test_patient_ids']
        train_entries = split['train_entries']
        test_entries = split['test_entries']
        
        # Count labels in train and test
        train_labels = [e.get(stratify_by, 'unknown') for e in train_entries]
        test_labels = [e.get(stratify_by, 'unknown') for e in test_entries]
        
        train_label_counts = defaultdict(int)
        test_label_counts = defaultdict(int)
        
        for label in train_labels:
            train_label_counts[label] += 1
        for label in test_labels:
            test_label_counts[label] += 1
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_pids)} patients, {len(train_entries)} files")
        print(f"    Label distribution: {dict(train_label_counts)}")
        print(f"  Test:  {len(test_pids)} patients, {len(test_entries)} files")
        print(f"    Label distribution: {dict(test_label_counts)}")
    
    print(f"{'='*80}\n")


def get_loso_splits(json_path):
    """
    Perform Leave-One-Subject-Out (LOSO) cross-validation.
    
    This function creates one fold per patient, where all files from that patient
    are in the test set and all other patients' files are in the training set.
    All files from the same patient (e.g., 49010018_1, 49010018_2, 49010018_3)
    will always be grouped together in the same fold.
    
    Args:
        json_path: Path to file_mapping_apnea_only.json
    
    Returns:
        list: List of dictionaries (one per patient), each containing:
              - 'patient_id': The patient ID used as test set in this fold
              - 'train_patient_ids': List of all other patient IDs in training set
              - 'test_patient_ids': List containing only the test patient ID
              - 'train_entries': List of all file entries (dicts) in training set
              - 'test_entries': List of all file entries (dicts) in test set
    """
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group entries by patient_id
    patient_data = defaultdict(lambda: {'entries': []})
    
    for entry in data.get('valid_mappings', []):
        patient_id = entry.get('patient_id')
        if not patient_id:
            continue
        
        patient_data[patient_id]['entries'].append(entry)
    
    # Get all unique patient IDs sorted for consistency
    all_patient_ids = sorted(patient_data.keys())
    
    # Create one split per patient
    splits = []
    
    for test_patient_id in all_patient_ids:
        # Test set: all entries from this patient
        test_patient_ids = [test_patient_id]
        test_entries = patient_data[test_patient_id]['entries'].copy()
        
        # Training set: all entries from all other patients
        train_patient_ids = [pid for pid in all_patient_ids if pid != test_patient_id]
        train_entries = []
        
        for pid in train_patient_ids:
            train_entries.extend(patient_data[pid]['entries'])
        
        splits.append({
            'patient_id': test_patient_id,  # The patient left out for this fold
            'train_patient_ids': train_patient_ids,
            'test_patient_ids': test_patient_ids,
            'train_entries': train_entries,
            'test_entries': test_entries
        })
    
    return splits


if __name__ == '__main__':
    # Original CSV generation functionality
    base_path = Path(__file__).parent
    json_path = base_path / 'output' / 'file_mapping_apnea_only.json'
    output_path = base_path / 'output' / 'patients_followups.csv'
    
    # Check if JSON file exists
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        exit(1)
    
    print(f"Loading file mapping from: {json_path}")
    patients_followups = load_file_mapping(json_path)
    
    print(f"\nGenerating CSV file...")
    generate_patients_csv(patients_followups, output_path)
    
    print(f"\nGenerating stratified k-fold splits...")
    splits = get_patient_wise_stratified_kfold_splits(
        json_path, 
        k=K_FOLDS, 
        random_state=42, 
        stratify_by='has_cvd'
    )
    
    print_split_statistics(splits, stratify_by='has_cvd')
    
    print(f"\nDone! CSV saved to: {output_path}")
    print(f"Generated {len(splits)} folds for patient-wise stratified k-fold split.")

