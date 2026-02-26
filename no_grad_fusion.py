"""
No-grad fusion: train ONLY the fusion head on top of frozen ECG and PPG backbones.

For each LOSO fold:
  1. Load pre-trained ecg_model (trained on subjects ≠ i)
  2. Load pre-trained ppg_model (trained on subjects ≠ i)
  3. Freeze both backbones (no gradients)
  4. Train ONLY the fusion head (~16K parameters)
  5. Test on subject i

Run from project root. Does not modify the main codebase; uses model/ as a library.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG: Set paths to LOSO result folders (ECG and PPG pre-trained models)
# ---------------------------------------------------------------------------
ECG_RUN_DIR = Path(
    "/data/R.Janini_Work/SA-CVD/ECG_PPG/multimodal-htn-framework/training_results/ecg_2026-02-12_18-47-15"
)
PPG_RUN_DIR = Path(
    "/data/R.Janini_Work/SA-CVD/ECG_PPG/multimodal-htn-framework/training_results/ppg_2026-02-12_16-53-19"
)

# Optional: run only one fold (1-indexed), or None to run all folds
PATIENT_TO_RUN = None  # e.g. 1 for first patient only
RANDOM_SEED = 42
NUM_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 64

# Project paths (same as main pipeline)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR
JSON_PATH = BASE_PATH / "data_preparation" / "output" / "file_mapping_apnea_only.json"
DATA_DIR = BASE_PATH / "data" / "Embla_Cleaned"
RESULTS_BASE_DIR = BASE_PATH / "training_results"

# Add model directory so we can import without changing main code
sys.path.insert(0, str(SCRIPT_DIR / "model"))

from train import train_model_loso
from normalization import calculate_normalization_coefficients, apply_normalization
from dataset import create_datasets_from_splits
from data_loader import load_data_from_entries
from fusion_architecture import FusionNet

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


class Tee:
    """Write to both terminal and file."""

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
    record_id_str = str(record_id)
    if "_" in record_id_str:
        return record_id_str.split("_")[0]
    return record_id_str


def extract_file_id_from_bag_id(bag_id):
    s = str(bag_id)
    if "_bag_" in s:
        return s.rsplit("_bag_", 1)[0]
    return s


def load_encoder_weights(checkpoint_path, prefix, device):
    """Load state_dict from checkpoint and return only keys with given prefix."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Handle DataParallel: strip 'module.' if present
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith(prefix):
            stripped[k] = v
    return stripped


def build_fusion_with_frozen_backbones(
    ecg_run_dir,
    ppg_run_dir,
    patient_id,
    device,
    embed_dim=64,
    hrv_dim=4,
):
    """
    Build FusionNet, load ECG and PPG encoder weights from LOSO checkpoints for this patient,
    then freeze both encoders. Only the fusion head is trainable.
    """
    ecg_ckpt = ecg_run_dir / f"best_model_ecg_loso_patient_{patient_id}.pt"
    ppg_ckpt = ppg_run_dir / f"best_model_ppg_loso_patient_{patient_id}.pt"

    if not ecg_ckpt.exists():
        raise FileNotFoundError(f"ECG checkpoint not found: {ecg_ckpt}")
    if not ppg_ckpt.exists():
        raise FileNotFoundError(f"PPG checkpoint not found: {ppg_ckpt}")

    # FusionNet with same encoder dims as single-modal models (embed_dim=64 -> fusion_dim=128)
    model = FusionNet(
        embed_dim=embed_dim,
        sleep_embed_dim=8,
        hrv_dim=hrv_dim,
        use_ecg_ppg=True,
        use_hrv=True,
        use_ppg_ft=True,
        use_sleep_stage=True,
        use_apnea=True,
        attn_dim=64,
        mil_pooling="attention",
    )

    # Load ECG encoder weights into FusionNet.ecg_encoder
    ecg_sd = load_encoder_weights(ecg_ckpt, "ecg_encoder.", device)
    if ecg_sd:
        model.load_state_dict(ecg_sd, strict=False)
        print(f"  Loaded {len(ecg_sd)} ECG encoder parameters from {ecg_ckpt.name}")
    # Load PPG encoder weights into FusionNet.ppg_encoder
    ppg_sd = load_encoder_weights(ppg_ckpt, "ppg_encoder.", device)
    if ppg_sd:
        model.load_state_dict(ppg_sd, strict=False)
        print(f"  Loaded {len(ppg_sd)} PPG encoder parameters from {ppg_ckpt.name}")

    # Freeze both backbones
    for p in model.ecg_encoder.parameters():
        p.requires_grad = False
    for p in model.ppg_encoder.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Parameters: {n_trainable:,} trainable (fusion head), {total - n_trainable:,} frozen (backbones)")

    return model


def main():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RESULTS_BASE_DIR / f"no_grad_fusion_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training_log.txt"
    log_handle = open(log_file, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(original_stdout),
        ],
    )

    try:
        print(f"No-grad fusion run directory: {run_dir}")
        print(f"ECG run dir: {ECG_RUN_DIR}")
        print(f"PPG run dir: {PPG_RUN_DIR}")
        print(f"{'='*80}\n")

        if not JSON_PATH.exists():
            print(f"Error: JSON file not found at {JSON_PATH}")
            return
        if not DATA_DIR.exists():
            print(f"Error: Data directory not found at {DATA_DIR}")
            return

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_entries = data.get("valid_mappings", [])
        patient_entries = defaultdict(list)
        for entry in all_entries:
            pid = entry.get("patient_id")
            if pid:
                patient_entries[pid].append(entry)
        unique_patients = sorted(patient_entries.keys())
        print(f"Total unique patients: {len(unique_patients)}")

        results_file = run_dir / "loso_results_no_grad_fusion.txt"
        with open(results_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BAG-LEVEL PREDICTIONS (no-grad fusion)\n")
            f.write("=" * 80 + "\n")
            f.write("patient_id\tbag_id\ttest_probability\ttrue_label\n")

        all_bag_probs = []
        all_bag_labels = []
        all_bag_record_ids = []
        all_file_probs = []
        all_file_labels = []
        all_file_ids = []

        fold_list = list(enumerate(unique_patients, 1))
        if PATIENT_TO_RUN is not None:
            fold_list = [(f, p) for f, p in fold_list if f == PATIENT_TO_RUN]
            if not fold_list:
                print(f"PATIENT_TO_RUN={PATIENT_TO_RUN} out of range. Exiting.")
                return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.BCEWithLogitsLoss()

        for fold_idx, test_patient_id in fold_list:
            print(f"\n{'='*80}")
            print(f"LOSO FOLD {fold_idx}/{len(unique_patients)}: Test patient {test_patient_id}")
            print(f"{'='*80}")

            test_entries = patient_entries[test_patient_id]
            remaining_patient_ids = [p for p in unique_patients if p != test_patient_id]
            remaining_labels = [
                1 if patient_entries[pid][0].get("has_cvd", "no").lower() == "yes" else 0
                for pid in remaining_patient_ids
            ]
            cvd_count = sum(remaining_labels)
            non_cvd_count = len(remaining_labels) - cvd_count

            if len(remaining_patient_ids) < 5 or cvd_count < 1 or non_cvd_count < 1:
                print("  Skipping: not enough patients for stratified split.")
                continue

            train_pids, val_pids = train_test_split(
                remaining_patient_ids,
                test_size=0.15,
                stratify=remaining_labels,
                random_state=RANDOM_SEED + fold_idx,
            )
            train_entries = []
            for pid in train_pids:
                train_entries.extend(patient_entries[pid])
            val_entries = []
            for pid in val_pids:
                val_entries.extend(patient_entries[pid])

            print(f"  Train: {len(train_entries)} files, Val: {len(val_entries)} files, Test: {len(test_entries)} files")

            train_data = load_data_from_entries(train_entries, DATA_DIR)
            val_data = load_data_from_entries(val_entries, DATA_DIR)
            test_data = load_data_from_entries(test_entries, DATA_DIR)

            if train_data["total_bags"] == 0 or val_data["total_bags"] == 0 or test_data["total_bags"] == 0:
                print("  Skipping: empty train/val/test.")
                continue

            norm_coefficients = calculate_normalization_coefficients(train_data["bags"])
            train_data["bags"] = apply_normalization(train_data["bags"], norm_coefficients)
            val_data["bags"] = apply_normalization(val_data["bags"], norm_coefficients)
            test_data["bags"] = apply_normalization(test_data["bags"], norm_coefficients)

            train_dataset, _ = create_datasets_from_splits(
                train_bags=train_data["bags"],
                test_bags=[],
                train_entries=train_entries,
                test_entries=[],
                data_dir=str(DATA_DIR),
                max_instances=60,
                min_instances=15,
                ecg_fs=200,
                ppg_fs=100,
            )
            _, val_dataset = create_datasets_from_splits(
                train_bags=[],
                test_bags=val_data["bags"],
                train_entries=[],
                test_entries=val_entries,
                data_dir=str(DATA_DIR),
                max_instances=60,
                min_instances=15,
                ecg_fs=200,
                ppg_fs=100,
            )
            _, test_dataset = create_datasets_from_splits(
                train_bags=[],
                test_bags=test_data["bags"],
                train_entries=[],
                test_entries=test_entries,
                data_dir=str(DATA_DIR),
                max_instances=60,
                min_instances=15,
                ecg_fs=200,
                ppg_fs=100,
            )

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Build fusion model with frozen ECG/PPG backbones
            print("  Building fusion model (frozen backbones)...")
            model = build_fusion_with_frozen_backbones(
                ECG_RUN_DIR,
                PPG_RUN_DIR,
                test_patient_id,
                device,
                embed_dim=64,
                hrv_dim=4,
            )

            # Optimizer only for fusion head parameters
            fusion_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(fusion_params, lr=5e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
            )

            model.to(device)
            save_path = str(run_dir / f"best_model_no_grad_fusion_loso_patient_{test_patient_id}.pt")

            print("  Training fusion head only...")
            best_epoch, best_val_bal = train_model_loso(
                model=model,
                scaler=None,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=NUM_EPOCHS,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                save_path=save_path,
                patience=PATIENCE,
            )
            print(f"  Best epoch: {best_epoch + 1}, best val balanced acc: {best_val_bal:.4f}")

            # Load best checkpoint and test
            if Path(save_path).exists():
                ckpt = torch.load(save_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()

            test_probs = []
            test_labels_list = []
            test_record_ids = []

            with torch.no_grad():
                for input_dict, y in test_loader:
                    ecg = input_dict.get("ecg")
                    ppg = input_dict.get("ppg")
                    if ecg is not None:
                        ecg = ecg.to(device)
                    if ppg is not None:
                        ppg = ppg.to(device)
                    mask = input_dict["mask"].to(device)
                    hrv = input_dict["hrv"].to(device)
                    sleep_stage = input_dict["sleep_stage"].to(device)
                    apnea = input_dict["apnea_label"].to(device)
                    y = y.view(-1).to(device)

                    output = model(
                        ecg=ecg,
                        ppg=ppg,
                        mask=mask,
                        hrv=hrv,
                        sleep_stage=sleep_stage,
                        apnea=apnea,
                    )
                    logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    test_probs.extend(probs)
                    test_labels_list.extend(y.cpu().numpy())
                    test_record_ids.extend(input_dict["record_id"])

            for bag_id, prob, label in zip(test_record_ids, test_probs, test_labels_list):
                all_bag_probs.append(prob)
                all_bag_labels.append(label)
                all_bag_record_ids.append(bag_id)
                with open(results_file, "a") as f:
                    f.write(f"{test_patient_id}\t{bag_id}\t{prob:.6f}\t{int(label)}\n")

            test_probs_array = np.array(test_probs)
            test_labels_array = np.array(test_labels_list)
            test_preds = (test_probs_array > 0.5).astype(int)

            if len(set(test_labels_array)) > 1:
                bag_auroc = roc_auc_score(test_labels_array, test_probs_array)
            else:
                bag_auroc = float("nan")
            bag_auprc = average_precision_score(test_labels_array, test_probs_array)
            bag_acc = accuracy_score(test_labels_array, test_preds)
            bag_sens = recall_score(test_labels_array, test_preds)
            cm = confusion_matrix(test_labels_array, test_preds)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                bag_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                bag_spec = 0.0
            bag_mcc = matthews_corrcoef(test_labels_array, test_preds)

            print(f"\n  Bag-level (patient {test_patient_id}): AUROC={bag_auroc:.4f}, AUPRC={bag_auprc:.4f}, Acc={bag_acc:.4f}, Sens={bag_sens:.4f}, Spec={bag_spec:.4f}, MCC={bag_mcc:.4f}")

            file_probs = defaultdict(list)
            file_labels = defaultdict(list)
            for bag_id, prob, label in zip(test_record_ids, test_probs, test_labels_list):
                fid = extract_file_id_from_bag_id(bag_id)
                file_probs[fid].append(prob)
                file_labels[fid].append(label)
            for fid in sorted(file_probs.keys()):
                probs_f = np.array(file_probs[fid])
                labels_f = np.array(file_labels[fid])
                vote_ratio = (probs_f > 0.5).astype(int).sum() / len(probs_f)
                all_file_probs.append(vote_ratio)
                all_file_labels.append(int(labels_f[0]))
                all_file_ids.append(fid)

        # Summary
        print(f"\n{'='*80}")
        print("NO-GRAD FUSION LOSO SUMMARY")
        print(f"{'='*80}")

        if all_bag_labels:
            bag_auroc_all = roc_auc_score(all_bag_labels, all_bag_probs) if len(set(all_bag_labels)) > 1 else float("nan")
            bag_auprc_all = average_precision_score(all_bag_labels, all_bag_probs)
            bag_preds = (np.array(all_bag_probs) > 0.5).astype(int)
            print(f"Bag-level: AUROC={bag_auroc_all:.4f}, AUPRC={bag_auprc_all:.4f}, Acc={accuracy_score(all_bag_labels, bag_preds):.4f}")

        if all_file_labels:
            file_preds = [1 if p > 0.5 else 0 for p in all_file_probs]
            file_acc = accuracy_score(all_file_labels, file_preds)
            file_auroc = roc_auc_score(all_file_labels, all_file_probs) if len(set(all_file_labels)) > 1 else float("nan")
            print(f"File-level (vote): Acc={file_acc:.4f}, AUROC={file_auroc:.4f}")

        with open(results_file, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("PATIENT-LEVEL PREDICTIONS (file voting)\n")
            f.write("=" * 80 + "\n")
            f.write("patient_id\tvote_ratio\ttrue_label\n")
            for fid, prob, label in zip(all_file_ids, all_file_probs, all_file_labels):
                pid = extract_patient_id_from_record_id(fid)
                f.write(f"{pid}\t{prob:.6f}\t{label}\n")

        print(f"\nResults written to {results_file}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
