"""
Model training script with LOSO (Leave One Subject Out) cross-validation.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from torch.utils.data import ConcatDataset, Subset
# Add parent directory to path to import from data_preparation
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    JSON_PATH,
    DATA_DIR,
    MODEL_TYPE,
    NUM_GPUS,
    PATIENT_TO_RUN,
    RANDOM_SEED,
    RESULTS_BASE_DIR,
    VALIDATION_MODE,
    ECG_PRETRAINED_RUN_DIR,
    PPG_PRETRAINED_RUN_DIR,
)
from data_loader import load_data_from_entries
from normalization import (
    calculate_normalization_coefficients,
    apply_normalization
)
from dataset import create_datasets_from_splits
from torch.utils.data import DataLoader

# Import architecture based on MODEL_TYPE
if MODEL_TYPE == 'fusion':
    from fusion_architecture import FusionNet
    ModelClass = FusionNet
elif MODEL_TYPE == 'ecg':
    from ecg_architecture import ECGNet
    ModelClass = ECGNet
elif MODEL_TYPE == 'ppg':
    from ppg_architecture import PPGNet
    ModelClass = PPGNet
elif MODEL_TYPE == 'fusion_pretrained':
    from fusion_pretrained import create_fusion_pretrained_for_fold
    ModelClass = None  # Built per fold via create_fusion_pretrained_for_fold
else:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'fusion', 'ecg', 'ppg', or 'fusion_pretrained'.")

from train import train_model_loso, BCEWithLogitsLossLabelSmoothing, FocalLoss
from sam import SAM
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import train_test_split


class Tee:
    """Class to write to both file and terminal simultaneously."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def extract_patient_id_from_record_id(record_id):
    """Extract patient ID from record_id (e.g., '49000123_1' -> '49000123')."""
    record_id_str = str(record_id)
    if '_' in record_id_str:
        return record_id_str.split('_')[0]
    return record_id_str


def extract_file_id_from_bag_id(bag_id):
    """
    Extract file ID (baseline/followup) from bag_id.
    e.g. '49010001_1_bag_0' -> '49010001_1', '49010003_2_bag_5' -> '49010003_2'
    File ID is used for per-file voting (each file = one recording session).
    """
    s = str(bag_id)
    if '_bag_' in s:
        return s.rsplit('_bag_', 1)[0]
    # If no _bag_ suffix, treat whole thing as file_id (single-bag file)
    return s


def main():
    """Main function for LOSO (Leave One Subject Out) validation."""
    
    # Create a per-run output directory under RESULTS_BASE_DIR
    # Example: training_results/fusion_2026-01-28_14-35-10
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RESULTS_BASE_DIR / f"{MODEL_TYPE}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to capture all terminal output
    log_file = run_dir / "training_log.txt"
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    
    # Create Tee to write to both terminal and file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file_handle)
    sys.stderr = Tee(original_stderr, log_file_handle)
    
    # Setup Python logging module
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(original_stdout)
        ]
    )
    
    try:
        print(f"Storing all checkpoints and logs for this run in: {run_dir}")
        print(f"All terminal output is being logged to: {log_file}")
        print(f"{'='*80}\n")
        
        # Check if JSON file exists
        if not JSON_PATH.exists():
            print(f"Error: JSON file not found at {JSON_PATH}")
            return
        
        # Check if data directory exists
        if not DATA_DIR.exists():
            print(f"Error: Data directory not found at {DATA_DIR}")
            return
        
        # Load all entries from JSON
        print(f"Loading entries from {JSON_PATH}...")
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_entries = data.get('valid_mappings', [])
        print(f"Total entries: {len(all_entries)}")
        
        # Group entries by patient_id
        patient_entries = defaultdict(list)
        for entry in all_entries:
            patient_id = entry.get('patient_id')
            if patient_id:
                patient_entries[patient_id].append(entry)
        
        unique_patients = sorted(patient_entries.keys())
        print(f"Total unique patients: {len(unique_patients)}")
        print(f"{'='*80}\n")
        
        # Setup output file for logging inside the run-specific directory
        results_file = run_dir / f"loso_results_{MODEL_TYPE}.txt"
        with open(results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BAG-LEVEL PREDICTIONS\n")
            f.write("="*80 + "\n")
            f.write("patient_id\tbag_id\ttest_probability\ttrue_label\n")
        
        # Store all predictions and labels for final metrics
        all_bag_probs = []
        all_bag_labels = []
        all_bag_record_ids = []
        # File-level: voting per file (baseline/followup), not per patient
        all_file_probs = []   # vote ratio per file
        all_file_labels = [] # true label per file (same for all bags in file)
        all_file_ids = []    # file_id e.g. 49010001_1
        
        # LOSO loop: for each patient (optionally only PATIENT_TO_RUN when set)
        fold_list = list(enumerate(unique_patients, 1))  # [(1, pid1), (2, pid2), ...]
        if VALIDATION_MODE == 'loso' and PATIENT_TO_RUN is not None:
            fold_list = [(f, p) for f, p in fold_list if f == PATIENT_TO_RUN]
            if not fold_list:
                print(f"PATIENT_TO_RUN={PATIENT_TO_RUN} is out of range (1 to {len(unique_patients)}). No fold to run.")
                return
        for fold_idx, test_patient_id in fold_list:
            print(f"\n{'='*80}")
            print(f"LOSO FOLD {fold_idx}/{len(unique_patients)}: Testing Patient {test_patient_id}")
            print(f"{'='*80}")
        
            # ===== 3-WAY SPLIT =====
            # Test Set: Current patient (held out completely)
            test_entries = patient_entries[test_patient_id]
        
            # Get remaining patients (all except test patient)
            remaining_patient_ids = [pid for pid in unique_patients if pid != test_patient_id]
        
            # Get labels for remaining patients for stratified split
            remaining_labels = []
            remaining_patient_id_list = []
            for pid in remaining_patient_ids:
                # Get label from first entry (all entries for same patient have same label)
                label = 1 if patient_entries[pid][0].get('has_cvd', 'no').lower() == 'yes' else 0
                remaining_labels.append(label)
                remaining_patient_id_list.append(pid)
        
            # Split remaining patients into train (~80%) and internal validation (~20%)
            # Use stratified split to maintain class balance
            # random_state ensures reproducibility, but different for each fold by adding fold_idx
        
            # Check if we have enough patients for stratification
            # Stratification requires at least 2 patients per class in both train and val
            cvd_count = sum(remaining_labels)
            non_cvd_count = len(remaining_labels) - cvd_count
        
            if len(remaining_patient_id_list) < 5 or cvd_count < 1 or non_cvd_count < 1:
                # Fallback: simple random split without stratification if not enough patients
                print(f"  Warning: Not enough patients for stratified split. Using random split.")
                print(f"    Total remaining: {len(remaining_patient_id_list)}, CVD: {cvd_count}, non-CVD: {non_cvd_count}")
                train_pids, val_pids = train_test_split(
                    remaining_patient_id_list,
                    test_size=0.15, 
                    stratify=None,  # No stratification
                    random_state=RANDOM_SEED + fold_idx
                )
            else:
                # Use stratified split to maintain class balance
                train_pids, val_pids = train_test_split(
                    remaining_patient_id_list,
                    test_size=0.15,  
                    stratify=remaining_labels,
                    random_state=RANDOM_SEED + fold_idx  # Different seed per fold but reproducible
                )
        
            # Collect entries for train and validation sets
            train_entries = []
            for pid in train_pids:
                train_entries.extend(patient_entries[pid])
        
            val_entries = []
            for pid in val_pids:
                val_entries.extend(patient_entries[pid])
        
            print(f"3-Way Split (stratified by CVD label):")
            print(f"  Test: {len(test_entries)} files from patient {test_patient_id} (1 patient)")
            print(f"  Internal Val: {len(val_entries)} files from {len(val_pids)} patients (~20%)")
            print(f"  Train: {len(train_entries)} files from {len(train_pids)} patients (~80%)")
        
            # Print class distribution for verification
            train_cvd_count = sum(1 for e in train_entries if e.get('has_cvd', 'no').lower() == 'yes')
            val_cvd_count = sum(1 for e in val_entries if e.get('has_cvd', 'no').lower() == 'yes')
            print(f"  Train CVD distribution: {train_cvd_count}/{len(train_entries)} CVD, {len(train_entries) - train_cvd_count}/{len(train_entries)} non-CVD")
            print(f"  Val CVD distribution: {val_cvd_count}/{len(val_entries)} CVD, {len(val_entries) - val_cvd_count}/{len(val_entries)} non-CVD")
        
            # Get true label for test patient (from first entry, all should have same label)
            test_label = 1 if test_entries[0].get('has_cvd', 'no').lower() == 'yes' else 0
        
            # Load data
            print("\nLoading training data...")
            train_data = load_data_from_entries(train_entries, DATA_DIR)
        
            print("Loading internal validation data...")
            val_data = load_data_from_entries(val_entries, DATA_DIR)
        
            print("Loading test data...")
            test_data = load_data_from_entries(test_entries, DATA_DIR)
        
            if train_data['total_bags'] == 0:
                print(f"Warning: No training bags found. Skipping patient {test_patient_id}.")
                continue
        
            if val_data['total_bags'] == 0:
                print(f"Warning: No validation bags found. Skipping patient {test_patient_id}.")
                continue
        
            if test_data['total_bags'] == 0:
                print(f"Warning: No test bags found. Skipping patient {test_patient_id}.")
                continue
        
            # Calculate normalization coefficients from TRAINING data only (not validation or test)
            print("\nCalculating normalization coefficients from training data only...")
            norm_coefficients = calculate_normalization_coefficients(train_data['bags'])
        
            # Apply normalization to all sets using training-only coefficients
            print("Applying normalization...")
            train_data['bags'] = apply_normalization(train_data['bags'], norm_coefficients)
            val_data['bags'] = apply_normalization(val_data['bags'], norm_coefficients)
            test_data['bags'] = apply_normalization(test_data['bags'], norm_coefficients)
        
            # Create datasets
            print("Creating datasets...")
            train_dataset, _ = create_datasets_from_splits(
            train_bags=train_data['bags'],
            test_bags=[],  # Dummy empty list, not used
            train_entries=train_entries,
            test_entries=[],
            data_dir=str(DATA_DIR),
            max_instances=60,
            min_instances=15,
            ecg_fs=200,
            ppg_fs=100,
            )
        
            # Create validation dataset (using test_bags parameter since function expects train/test split)
            _, val_dataset = create_datasets_from_splits(
            train_bags=[],  # Dummy empty list
            test_bags=val_data['bags'],
            train_entries=[],
            test_entries=val_entries,
            data_dir=str(DATA_DIR),
            max_instances=60,
            min_instances=30,
            ecg_fs=200,
            ppg_fs=100,
            )
        
            # Create test dataset (using test_bags parameter)
            _, test_dataset = create_datasets_from_splits(
            train_bags=[],  # Dummy empty list
            test_bags=test_data['bags'],
            train_entries=[],
            test_entries=test_entries,
            data_dir=str(DATA_DIR),
            max_instances=60,
            min_instances=30,
            ecg_fs=200,
            ppg_fs=100,
            )
        
            print(f"Train dataset: {len(train_dataset)} bags")
            print(f"Internal Val dataset: {len(val_dataset)} bags")
            print(f"Test dataset: {len(test_dataset)} bags")
        
            # Use only 10% random sample of test set in validation

            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
            # Initialize model (and optimizer/scheduler) from scratch for this fold.
            # Each LOSO fold trains independently; no state or gradients carry over.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\nInitializing {MODEL_TYPE.upper()} model (fresh for this fold)...")
            if MODEL_TYPE == 'fusion_pretrained':
                # Must match architecture used when training ECG/PPG standalone (checkpoints)
                # ECG: use_hrv (ecg_hf, ecg_lf). PPG: use_prv (ppg_hf, ppg_lf).
                model = create_fusion_pretrained_for_fold(
                    ECG_PRETRAINED_RUN_DIR,
                    PPG_PRETRAINED_RUN_DIR,
                    test_patient_id,
                    device,
                    freeze_encoders=True,
                    embed_dim=128,
                    sleep_embed_dim=8,
                    fusion_dropout=0.3,
                    fusion_hidden_1=64,
                    fusion_hidden_2=32,
                    attn_dim=64,
                    ecg_use_hrv=True,
                    ecg_hrv_dim=2,
                    ecg_use_sleep_stage=True,
                    ecg_use_apnea=True,
                    ppg_use_prv=True,
                    ppg_prv_dim=2,
                    ppg_use_sleep_stage=True,
                    ppg_use_apnea=True,
                )
            elif MODEL_TYPE == 'ecg':
                # ECG model uses only ECG HRV (first 2 dims: ecg_hf, ecg_lf)
                model = ModelClass(
                    embed_dim=128,
                    sleep_embed_dim=8,
                    hrv_dim=2,
                    use_hrv=True,
                    use_sleep_stage=True,
                    use_apnea=True,
                    attn_dim=64,
                )
            elif MODEL_TYPE == 'ppg':
                # PPG model uses only PRV (ppg_hf, ppg_lf from ppg_hrv group); PPG has PRV, not HRV
                model = ModelClass(
                    embed_dim=128,
                    sleep_embed_dim=8,
                    prv_dim=2,
                    use_prv=True,
                    use_sleep_stage=True,
                    use_apnea=True,
                    attn_dim=64,
                )

            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

            if device.type == 'cuda' and NUM_GPUS > 1:
                n_devices = min(NUM_GPUS, torch.cuda.device_count())
                model = nn.DataParallel(model, device_ids=list(range(n_devices)))
                print(f"Using DataParallel on {n_devices} GPU(s)")
            else:
                print(f"Using single device: {device}")


    
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )
            
            # optimizer = SAM(
            #     model.parameters(),
            #     base_optimizer=torch.optim.AdamW,
            #     rho=0.05,
            #     lr=1e-4,
            #     weight_decay=1e-3,
            # )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer.base_optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            # )
        
            # Train model with internal validation set for early stopping
            print(f"\nTraining model for patient {test_patient_id}...")
            print(f"Using internal validation set ({len(val_pids)} patients) for early stopping.")
            print(f"Model selection criterion: Lowest validation loss")
            # ECG: HRV dims 0:2 (ecg_hf, ecg_lf); PPG: dims 2:4 (ppg_hf, ppg_lf); fusion: full 4-dim
            hrv_slice = (0, 2) if MODEL_TYPE == 'ecg' else ((2, 4) if MODEL_TYPE == 'ppg' else None)
            best_epoch, best_val_loss = train_model_loso(
                model=model,
                scaler=None,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=100,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                save_path=str(run_dir / f"best_model_{MODEL_TYPE}_loso_patient_{test_patient_id}.pt"),
                patience=15,
                hrv_slice=hrv_slice,
            )
            print(f"Best model selected at epoch {best_epoch + 1} with validation loss: {best_val_loss:.4f}")
        
            # Test model and get predictions
            print(f"\nTesting on patient {test_patient_id}...")
            model.eval()
            test_probs = []
            test_labels_list = []
            test_record_ids = []
        
            # Load best model from this run's directory
            model_path = run_dir / f"best_model_{MODEL_TYPE}_loso_patient_{test_patient_id}.pt"
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                state_dict = checkpoint['model_state_dict']
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(state_dict, strict=False)
        
            with torch.no_grad():
                for input_dict, y in test_loader:
                    ecg = input_dict.get('ecg', None)
                    ppg = input_dict.get('ppg', None)
                    if ecg is not None:
                        ecg = ecg.to(device)
                    if ppg is not None:
                        ppg = ppg.to(device)
                    
                    mask = input_dict['mask'].to(device)
                    hrv_full = input_dict['hrv'].to(device)
                    # ECG: first 2 dims (ECG HRV); PPG: last 2 dims (PRV); fusion: full 4-dim
                    if MODEL_TYPE == 'ecg':
                        hrv = hrv_full[:, :2]
                    elif MODEL_TYPE == 'ppg':
                        hrv = hrv_full[:, 2:4]
                    else:
                        hrv = hrv_full
                    sleep_stage = input_dict['sleep_stage'].to(device)
                    apnea = input_dict['apnea_label'].to(device)
                    record_ids = input_dict['record_id']  # List of record IDs (bag_id)
                    y = y.view(-1).to(device)
                    
                    output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                    logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    test_probs.extend(probs)
                    test_labels_list.extend(y.cpu().numpy())
                    test_record_ids.extend(record_ids)
        
            # Store bag-level predictions
            for bag_id, prob, label in zip(test_record_ids, test_probs, test_labels_list):
                all_bag_probs.append(prob)
                all_bag_labels.append(label)
                all_bag_record_ids.append(bag_id)
                # Log bag-level predictions to file
                with open(results_file, 'a') as f:
                    f.write(f"{test_patient_id}\t{bag_id}\t{prob:.6f}\t{int(label)}\n")
        
            # Calculate bag-level metrics for this patient
            test_probs_array = np.array(test_probs)
            test_labels_array = np.array(test_labels_list)
            test_preds = (test_probs_array > 0.5).astype(int)
        
            # Calculate metrics for this patient's bags
            if len(set(test_labels_array)) > 1:
                bag_auroc_patient = roc_auc_score(test_labels_array, test_probs_array)
            else:
                bag_auroc_patient = float('nan')
        
            bag_auprc_patient = average_precision_score(test_labels_array, test_probs_array)
            bag_accuracy_patient = accuracy_score(test_labels_array, test_preds)
            bag_sensitivity_patient = recall_score(test_labels_array, test_preds)
        
            # Calculate specificity for this patient
            bag_cm_patient = confusion_matrix(test_labels_array, test_preds)
            if bag_cm_patient.size == 4:  # 2x2 matrix
                tn, fp, fn, tp = bag_cm_patient.ravel()
                bag_specificity_patient = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                bag_specificity_patient = 0.0
        
            bag_mcc_patient = matthews_corrcoef(test_labels_array, test_preds)
        
            # Print bag-level metrics for this patient
            print(f"\nBag-level metrics (Patient {test_patient_id}):")
            print(f"  AUROC: {bag_auroc_patient:.4f}")
            print(f"  AUPRC: {bag_auprc_patient:.4f}")
            print(f"  Accuracy: {bag_accuracy_patient:.4f}")
            print(f"  Sensitivity (Recall): {bag_sensitivity_patient:.4f}")
            print(f"  Specificity: {bag_specificity_patient:.4f}")
            print(f"  MCC: {bag_mcc_patient:.4f}")
            print(f"  Number of bags: {len(test_probs)}")
        
            # Aggregate predictions per FILE (baseline/followup) using VOTING
            # Group bags by file_id (e.g. 49010001_1, 49010001_2), then majority vote per file
            file_probs = defaultdict(list)
            file_labels = defaultdict(list)
            for bag_id, prob, label in zip(test_record_ids, test_probs, test_labels_list):
                fid = extract_file_id_from_bag_id(bag_id)
                file_probs[fid].append(prob)
                file_labels[fid].append(label)
            
            # Per-file voting and store for final metrics
            print(f"\nFile-level voting (Patient {test_patient_id}):")
            for fid in sorted(file_probs.keys()):
                probs_f = np.array(file_probs[fid])
                labels_f = np.array(file_labels[fid])
                bag_votes = (probs_f > 0.5).astype(int)
                positive_votes = int(np.sum(bag_votes))
                n_bags = len(probs_f)
                vote_ratio = positive_votes / n_bags
                file_pred = 1 if vote_ratio >= 0.5 else 0  # 0.5 counts as positive (1)
                true_label_f = int(labels_f[0])  # Same for all bags in file
                all_file_probs.append(vote_ratio)
                all_file_labels.append(true_label_f)
                all_file_ids.append(fid)
                print(f"  File {fid}: {positive_votes}/{n_bags} positive votes -> vote_ratio={vote_ratio:.4f}, pred={file_pred}, true={true_label_f}")
        
            print(f"Completed fold {fold_idx}/{len(unique_patients)}")
            print(f"{'='*80}\n")
    
        # Hierarchy: Bags (raw) -> Files (~68) -> Patients (33)
        # BAG LEVEL: Raw model output (prediction per bag); global metrics for reference
        print(f"\n{'='*80}")
        print("FINAL METRICS - BAG LEVEL (raw model output, per-bag predictions)")
        print(f"{'='*80}")
    
        # Bag-level metrics: use individual bag predictions (not averaged per patient)
        all_bag_probs = np.array(all_bag_probs)
        all_bag_labels = np.array(all_bag_labels)
        bag_preds = (all_bag_probs > 0.5).astype(int)
    
        # Calculate bag-level metrics
        bag_auroc = roc_auc_score(all_bag_labels, all_bag_probs) if len(set(all_bag_labels)) > 1 else float('nan')
        bag_auprc = average_precision_score(all_bag_labels, all_bag_probs)
        bag_accuracy = accuracy_score(all_bag_labels, bag_preds)
        bag_sensitivity = recall_score(all_bag_labels, bag_preds)
    
        # Calculate bag-level specificity
        bag_cm = confusion_matrix(all_bag_labels, bag_preds)
        if bag_cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = bag_cm.ravel()
            bag_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            bag_specificity = 0.0
    
        bag_mcc = matthews_corrcoef(all_bag_labels, bag_preds)
    
        # Print bag-level metrics
        print(f"AUROC: {bag_auroc:.4f}")
        print(f"AUPRC: {bag_auprc:.4f}")
        print(f"Accuracy: {bag_accuracy:.4f}")
        print(f"Sensitivity (Recall): {bag_sensitivity:.4f}")
        print(f"Specificity: {bag_specificity:.4f}")
        print(f"MCC: {bag_mcc:.4f}")
        print(f"Total bags: {len(all_bag_probs)}")
    
        print(f"\n{'='*80}")
        print("FINAL METRICS - FILE LEVEL (majority voting per file)")
        print(f"{'='*80}")
    
        # File-level metrics: one prediction per file (baseline/followup) via majority vote over its bags
        all_file_probs = np.array(all_file_probs)
        all_file_labels = np.array(all_file_labels)
        all_file_preds = (all_file_probs >= 0.5).astype(int)  # 0.5 counts as positive
    
        file_auroc = roc_auc_score(all_file_labels, all_file_probs) if len(set(all_file_labels)) > 1 else float('nan')
        file_auprc = average_precision_score(all_file_labels, all_file_probs)
        file_accuracy = accuracy_score(all_file_labels, all_file_preds)
        file_sensitivity = recall_score(all_file_labels, all_file_preds)
        file_cm = confusion_matrix(all_file_labels, all_file_preds)
        if file_cm.size == 4:
            tn, fp, fn, tp = file_cm.ravel()
            file_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            file_specificity = 0.0
        file_mcc = matthews_corrcoef(all_file_labels, all_file_preds)
    
        print(f"AUROC: {file_auroc:.4f}")
        print(f"AUPRC: {file_auprc:.4f}")
        print(f"Accuracy: {file_accuracy:.4f}")
        print(f"Sensitivity (Recall): {file_sensitivity:.4f}")
        print(f"Specificity: {file_specificity:.4f}")
        print(f"MCC: {file_mcc:.4f}")
        print(f"Total files: {len(all_file_probs)}")
    
        # PATIENT LEVEL: Group files by patient (~33 patients), vote over files per patient, metrics over patients
        patient_to_file_probs = defaultdict(list)
        patient_to_file_labels = defaultdict(list)
        for fid, prob, label in zip(all_file_ids, all_file_probs, all_file_labels):
            pid = extract_patient_id_from_record_id(fid)
            patient_to_file_probs[pid].append(prob)
            patient_to_file_labels[pid].append(label)
        
        all_patient_probs = []
        all_patient_labels = []
        all_patient_ids = []
        for pid in sorted(patient_to_file_probs.keys()):
            file_probs_p = np.array(patient_to_file_probs[pid])
            file_labels_p = np.array(patient_to_file_labels[pid])
            file_preds = (file_probs_p >= 0.5).astype(int)  # 0.5 counts as positive
            positive_files = int(np.sum(file_preds))
            n_files = len(file_probs_p)
            patient_vote_ratio = positive_files / n_files  # Proportion of files that predicted positive
            patient_pred = 1 if patient_vote_ratio >= 0.5 else 0  # 0.5 counts as positive
            patient_true = int(file_labels_p[0])  # Same for all files of this patient
            all_patient_probs.append(patient_vote_ratio)
            all_patient_labels.append(patient_true)
            all_patient_ids.append(pid)
        
        all_patient_probs = np.array(all_patient_probs)
        all_patient_labels = np.array(all_patient_labels)
        all_patient_preds = (all_patient_probs >= 0.5).astype(int)  # 0.5 counts as positive
        
        patient_auroc = roc_auc_score(all_patient_labels, all_patient_probs) if len(set(all_patient_labels)) > 1 else float('nan')
        patient_auprc = average_precision_score(all_patient_labels, all_patient_probs)
        patient_accuracy = accuracy_score(all_patient_labels, all_patient_preds)
        patient_sensitivity = recall_score(all_patient_labels, all_patient_preds)
        patient_cm = confusion_matrix(all_patient_labels, all_patient_preds)
        if patient_cm.size == 4:
            tn, fp, fn, tp = patient_cm.ravel()
            patient_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            patient_specificity = 0.0
        patient_mcc = matthews_corrcoef(all_patient_labels, all_patient_preds)
        
        print(f"\n{'='*80}")
        print("FINAL METRICS - PATIENT LEVEL (majority voting over files per patient)")
        print(f"{'='*80}")
        print(f"AUROC: {patient_auroc:.4f}")
        print(f"AUPRC: {patient_auprc:.4f}")
        print(f"Accuracy: {patient_accuracy:.4f}")
        print(f"Sensitivity (Recall): {patient_sensitivity:.4f}")
        print(f"Specificity: {patient_specificity:.4f}")
        print(f"MCC: {patient_mcc:.4f}")
        print(f"Total patients: {len(all_patient_probs)}")
    
        # Per-patient summary: list each patient's files with their vote results
        print(f"\n{'='*80}")
        print("PER-PATIENT SUMMARY (files with vote results)")
        print(f"{'='*80}")
        patient_to_files = defaultdict(list)
        for fid, prob, label in zip(all_file_ids, all_file_probs, all_file_labels):
            pid = extract_patient_id_from_record_id(fid)
            pred = 1 if prob >= 0.5 else 0  # 0.5 counts as positive
            patient_to_files[pid].append((fid, prob, pred, label))
        for pid in sorted(patient_to_files.keys()):
            rows = patient_to_files[pid]
            print(f"\nPatient {pid} ({len(rows)} file(s)):")
            for fid, vote_ratio, pred, true_label in rows:
                print(f"  File {fid}: vote_ratio={vote_ratio:.4f}, pred={pred}, true={int(true_label)}")
        print(f"{'='*80}\n")
    
        # Append final metrics to results file
        with open(results_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write("FILE-LEVEL PREDICTIONS (majority voting per file)\n")
            f.write(f"{'='*80}\n")
            f.write("file_id\tvote_ratio\ttrue_label\n")
            for fid, prob, label in zip(all_file_ids, all_file_probs, all_file_labels):
                f.write(f"{fid}\t{prob:.6f}\t{int(label)}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("FINAL METRICS - BAG LEVEL (individual bag predictions)\n")
            f.write(f"{'='*80}\n")
            f.write(f"AUROC: {bag_auroc:.4f}\n")
            f.write(f"AUPRC: {bag_auprc:.4f}\n")
            f.write(f"Accuracy: {bag_accuracy:.4f}\n")
            f.write(f"Sensitivity (Recall): {bag_sensitivity:.4f}\n")
            f.write(f"Specificity: {bag_specificity:.4f}\n")
            f.write(f"MCC: {bag_mcc:.4f}\n")
            f.write(f"Total bags: {len(all_bag_probs)}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("FINAL METRICS - FILE LEVEL (majority voting per file)\n")
            f.write(f"{'='*80}\n")
            f.write(f"AUROC: {file_auroc:.4f}\n")
            f.write(f"AUPRC: {file_auprc:.4f}\n")
            f.write(f"Accuracy: {file_accuracy:.4f}\n")
            f.write(f"Sensitivity (Recall): {file_sensitivity:.4f}\n")
            f.write(f"Specificity: {file_specificity:.4f}\n")
            f.write(f"MCC: {file_mcc:.4f}\n")
            f.write(f"Total files: {len(all_file_probs)}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("PATIENT-LEVEL PREDICTIONS (majority voting over files per patient)\n")
            f.write(f"{'='*80}\n")
            f.write("patient_id\tvote_ratio\ttrue_label\n")
            for pid, prob, label in zip(all_patient_ids, all_patient_probs, all_patient_labels):
                f.write(f"{pid}\t{prob:.6f}\t{int(label)}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("FINAL METRICS - PATIENT LEVEL (majority voting over files)\n")
            f.write(f"{'='*80}\n")
            f.write(f"AUROC: {patient_auroc:.4f}\n")
            f.write(f"AUPRC: {patient_auprc:.4f}\n")
            f.write(f"Accuracy: {patient_accuracy:.4f}\n")
            f.write(f"Sensitivity (Recall): {patient_sensitivity:.4f}\n")
            f.write(f"Specificity: {patient_specificity:.4f}\n")
            f.write(f"MCC: {patient_mcc:.4f}\n")
            f.write(f"Total patients: {len(all_patient_probs)}\n")
        
        print(f"Results saved to: {results_file}")
        print(f"Training log saved to: {log_file}")
        
    finally:
        # Always restore original stdout/stderr and close log file, even on errors
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
    
    return {
        'bag_level': {
            'probs': all_bag_probs,
            'labels': all_bag_labels,
            'record_ids': all_bag_record_ids,
            'auroc': bag_auroc,
            'auprc': bag_auprc,
            'accuracy': bag_accuracy,
            'sensitivity': bag_sensitivity,
            'specificity': bag_specificity,
            'mcc': bag_mcc
        },
        'file_level': {
            'file_ids': all_file_ids,
            'vote_ratios': all_file_probs,
            'labels': all_file_labels,
            'auroc': file_auroc,
            'auprc': file_auprc,
            'accuracy': file_accuracy,
            'sensitivity': file_sensitivity,
            'specificity': file_specificity,
            'mcc': file_mcc
        },
        'patient_level': {
            'patient_ids': all_patient_ids,
            'vote_ratios': all_patient_probs,
            'labels': all_patient_labels,
            'auroc': patient_auroc,
            'auprc': patient_auprc,
            'accuracy': patient_accuracy,
            'sensitivity': patient_sensitivity,
            'specificity': patient_specificity,
            'mcc': patient_mcc
        }
    }


if __name__ == '__main__':
    print("Running LOSO (Leave One Subject Out) cross-validation...")
    main()

