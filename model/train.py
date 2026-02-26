"""
Training utilities for the hypertension prediction model.
"""

import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from collections import Counter

from sam import enable_running_stats, disable_running_stats


class BCEWithLogitsLossLabelSmoothing(nn.Module):
    """BCEWithLogitsLoss with label smoothing. Targets 0/1 are smoothed to (smoothing, 1 - smoothing)."""
    def __init__(self, smoothing: float = 0.1, **kwargs):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # y=1 -> 1 - smoothing; y=0 -> smoothing
        smooth = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return self.bce(logits, smooth)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples. It's defined as:
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the predicted probability for the true class
    - alpha is a weighting factor (typically 0.25 for positive class)
    - gamma is the focusing parameter (typically 2.0)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid), shape (N,)
            targets: Ground truth labels (0 or 1), shape (N,)
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Compute p_t: probability of the true class
        # For positive samples (targets=1): p_t = probs
        # For negative samples (targets=0): p_t = 1 - probs
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t: alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy: -log(p_t)
        # Use log(probs) for positive, log(1-probs) for negative
        ce_loss = -torch.log(p_t + 1e-8)  # Add small epsilon to avoid log(0)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    patience: int = 15,
    save_path: str = "best_model.pt",
    scaler: Optional[GradScaler] = None,
) -> Tuple[int, float]:
    """
    Train the model with early stopping based on validation AUPRC.
    
    Args:
        model: The neural network model
        scaler: Gradient scaler for mixed precision training
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run training on (cuda/cpu)
        patience: Number of epochs to wait before early stopping (not used in current implementation)
        save_path: Path to save the best model
    
    Returns:
        Tuple of (best_epoch, best_auprc)
    """
    model.to(device)
    best_auprc = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    
    # Track training losses
    train_losses = []

    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        use_sam = getattr(optimizer, "first_step", None) is not None and getattr(optimizer, "second_step", None) is not None
        # For multi-GPU (DDP): no_sync on first backward so sub-batch SAM gradients are averaged in second step
        no_sync_ctx = model.no_sync() if getattr(model, "no_sync", None) else contextlib.nullcontext()

        # Gradient and shape debugging: check first batch of first epoch
        check_gradients = (epoch == 0)
        check_shapes = (epoch == 0)
        batch_idx = 0

        for input_dict, y in train_loader:
            # Always load ECG/PPG if available, but model will decide whether to use them based on use_ecg_ppg flag
            ecg = input_dict.get('ecg', None)
            ppg = input_dict.get('ppg', None)
            if ecg is not None:
                ecg = ecg.to(device)  # [B, N, L_ecg]
            if ppg is not None:
                ppg = ppg.to(device)  # [B, N, L_ppg]
            
            mask = input_dict['mask'].to(device)  # [B, N]
            hrv = input_dict['hrv'].to(device)  # [B, 4] = [ecg_hf, ecg_lf, ppg_hf, ppg_lf]
            sleep_stage = input_dict['sleep_stage'].to(device)  # [B, N]
            apnea = input_dict['apnea_label'].to(device)  # [B, N]
            y = y.view(-1).to(device)  # [B]

            optimizer.zero_grad()

            if use_sam:
                # SAM: full precision (no AMP) to avoid GradScaler/NaN issues with two-step update
                enable_running_stats(model)
                output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)  # [B]
                
                # Shape debugging: check for broadcasting bugs (first batch of first epoch only)
                if check_shapes and batch_idx == 0:
                    print(f"\n[SHAPE CHECK] Epoch {epoch+1}, Batch {batch_idx} (SAM):")
                    print(f"  logits shape: {logits.shape}")
                    print(f"  y shape: {y.shape}")
                    if logits.shape != y.shape:
                        print(f"  ⚠️ WARNING: Shape mismatch! logits={logits.shape}, y={y.shape}")
                        if logits.shape == (logits.size(0), 1) and y.shape == (y.size(0),):
                            print(f"  → Applying .view(-1) fix to logits")
                            logits = logits.view(-1)
                        elif logits.shape == (logits.size(0),) and y.shape == (y.size(0), 1):
                            print(f"  → Applying .view(-1) fix to y")
                            y = y.view(-1)
                    else:
                        print(f"  ✓ Shapes match correctly")
                    check_shapes = False  # Only check once
                
                loss = criterion(logits, y)
                with no_sync_ctx:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                output2 = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                logits2 = output2[0].squeeze(-1) if isinstance(output2, tuple) else output2.squeeze(-1)
                loss2 = criterion(logits2, y)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step(zero_grad=True)
            else:
                # No SAM: full precision if scaler is None, else AMP
                if scaler is not None:
                    with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                        output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                        logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)  # [B]
                        
                        # Shape debugging: check for broadcasting bugs (first batch of first epoch only)
                        if check_shapes and batch_idx == 0:
                            print(f"\n[SHAPE CHECK] Epoch {epoch+1}, Batch {batch_idx} (AMP):")
                            print(f"  logits shape: {logits.shape}")
                            print(f"  y shape: {y.shape}")
                            if logits.shape != y.shape:
                                print(f"  ⚠️ WARNING: Shape mismatch! logits={logits.shape}, y={y.shape}")
                                if logits.shape == (logits.size(0), 1) and y.shape == (y.size(0),):
                                    print(f"  → Applying .view(-1) fix to logits")
                                    logits = logits.view(-1)
                                elif logits.shape == (logits.size(0),) and y.shape == (y.size(0), 1):
                                    print(f"  → Applying .view(-1) fix to y")
                                    y = y.view(-1)
                            else:
                                print(f"  ✓ Shapes match correctly")
                            check_shapes = False  # Only check once
                        
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                    logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)  # [B]
                    
                    # Shape debugging: check for broadcasting bugs (first batch of first epoch only)
                    if check_shapes and batch_idx == 0:
                        print(f"\n[SHAPE CHECK] Epoch {epoch+1}, Batch {batch_idx}:")
                        print(f"  logits shape: {logits.shape}")
                        print(f"  y shape: {y.shape}")
                        print(f"  logits dtype: {logits.dtype}, y dtype: {y.dtype}")
                        if logits.shape != y.shape:
                            print(f"  ⚠️ WARNING: Shape mismatch! logits={logits.shape}, y={y.shape}")
                            if logits.shape == (logits.size(0), 1) and y.shape == (y.size(0),):
                                print(f"  → Applying .view(-1) fix to logits")
                                logits = logits.view(-1)
                            elif logits.shape == (logits.size(0),) and y.shape == (y.size(0), 1):
                                print(f"  → Applying .view(-1) fix to y")
                                y = y.view(-1)
                        else:
                            print(f"  ✓ Shapes match correctly")
                        check_shapes = False  # Only check once
                    
                    loss = criterion(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += loss.item() * y.size(0)
            probs = torch.sigmoid(logits.detach()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)
            batch_idx += 1

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_bal = balanced_accuracy_score(all_labels, all_preds)
        train_mcc = matthews_corrcoef(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_preds)
        train_auprc = average_precision_score(all_labels, all_preds)
        train_losses.append(train_loss)

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_labels, val_preds = [], []
        val_record_ids = []  # Store record IDs (bag_id/patient_id)
        val_probs = []  # Store probabilities for predictions
        with torch.no_grad():
            for input_dict, y in val_loader:
                # Always load ECG/PPG if available, but model will decide whether to use them based on use_ecg_ppg flag
                ecg = input_dict.get('ecg', None)
                ppg = input_dict.get('ppg', None)
                if ecg is not None:
                    ecg = ecg.to(device)  # [B, N, L_ecg]
                if ppg is not None:
                    ppg = ppg.to(device)  # [B, N, L_ppg]
                
                mask = input_dict['mask'].to(device)  # [B, N]
                hrv = input_dict['hrv'].to(device)  # [B, 4]
                sleep_stage = input_dict['sleep_stage'].to(device)  # [B, N]
                apnea = input_dict['apnea_label'].to(device)  # [B, N]
                record_ids = input_dict['record_id']  # List of record IDs (bag_id)
                y = y.view(-1).to(device)  # [B]

                output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_labels.extend(y.cpu().numpy())
                val_preds.extend(preds)
                val_probs.extend(probs)
                val_record_ids.extend(record_ids)  # record_id is already a list from the batch


        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_bal = balanced_accuracy_score(val_labels, val_preds)
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else float('nan')
        val_auprc = average_precision_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)

        # Save validation results to text file (overwrite each epoch)
        results_file = save_path.replace('.pt', '_best_epoch.txt')
        with open(results_file, 'w') as f:
            # Write header
            f.write("patient_id\tbag_id\ttruth\tprediction\tprobability\n")
            # Write data
            for record_id, truth, pred, prob in zip(val_record_ids, val_labels, val_preds, val_probs):
                # Extract patient_id from record_id
                # Try to extract patient ID from record_id (could be filename or patient_id format)
                record_id_str = str(record_id)
                bag_id = record_id_str
                
                # Try different patterns to extract patient_id
                # Pattern 1: "patient_id_..." -> extract first part
                if '_' in record_id_str:
                    parts = record_id_str.split('_')
                    # If it looks like "patient_id_rest", use first part as patient_id
                    # Otherwise, use the whole thing
                    if len(parts) > 1 and len(parts[0]) > 0:
                        patient_id = parts[0]
                    else:
                        patient_id = record_id_str
                else:
                    # No underscore, use whole record_id as patient_id
                    patient_id = record_id_str
                
                f.write(f"{patient_id}\t{bag_id}\t{int(truth)}\t{pred}\t{prob:.6f}\n")
        print(f"Validation results saved to {results_file}")

        # Update learning rate scheduler
        scheduler.step(val_auprc)

        # === LOG ===
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Train Loss:{train_loss:.4f} | Val Loss:{val_loss:.4f}")
        print(f"Train Acc :{train_acc:.4f} | Val Acc :{val_acc:.4f}")
        print(f"Train Prec:{train_precision:.4f} | Val Prec:{val_precision:.4f}")
        print(f"Train F1  :{train_f1:.4f} | Val F1  :{val_f1:.4f}")
        print(f"Train Bal :{train_bal:.4f} | Val Bal :{val_bal:.4f}")
        print(f"Train MCC :{train_mcc:.4f} | Val MCC :{val_mcc:.4f}")
        print(f"Train AUROC:{train_auc:.4f} | Val AUROC:{val_auc:.4f}")
        print(f"Train AUPRC:{train_auprc:.4f} | Val AUPRC:{val_auprc:.4f}")
        print("-" * 30)

        # Early stopping: save best model based on validation AUPRC
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            epochs_no_improve = 0
            # Handle DataParallel: save model.module.state_dict() if wrapped
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_auprc': best_auprc,
            }, save_path)
            print(f"✔️ Saved model at epoch {epoch+1} (Val AUPRC: {val_auprc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement")
                break

    return best_epoch, best_auprc


def test_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    criterion: nn.Module,
    model_path: str
) -> dict:
    """
    Test the model on the test set by loading the best checkpoint.
    
    Args:
        model: The neural network model (can be wrapped with DataParallel)
        test_loader: DataLoader for test data
        device: Device to run testing on (cuda/cpu)
        criterion: Loss function
        model_path: Path to the saved model checkpoint
    
    Returns:
        Dictionary containing test metrics
    """
    import os
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return {}
    
    print(f"\n{'='*80}")
    print(f"LOADING BEST MODEL FOR TESTING")
    print(f"{'='*80}")
    print(f"Loading checkpoint from: {model_path}")
    
    # Load checkpoint
    # weights_only=False is needed for PyTorch 2.6+ when loading checkpoints with numpy arrays
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel: The checkpoint was saved with model.module.state_dict()
    # So the keys don't have 'module.' prefix. We need to load accordingly.
    # If model is wrapped with DataParallel, load into model.module
    # If model is not wrapped, load directly into model
    try:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Handle case where checkpoint keys don't match (e.g., saved with/without DataParallel)
        # Try loading with strict=False as fallback, or strip 'module.' prefix if needed
        if 'module.' in list(state_dict.keys())[0]:
            # Checkpoint has 'module.' prefix, but model is not wrapped
            # Strip 'module.' prefix
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(new_state_dict, strict=False)
        else:
            # Checkpoint doesn't have 'module.' prefix, but model might be wrapped
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict, strict=False)
            else:
                # Try loading with strict=False as last resort
                model.load_state_dict(state_dict, strict=False)
        print(f"Warning: Some keys in checkpoint did not match. Loaded with strict=False.")
    
    print(f"✓ Model loaded successfully")
    print(f"  Best epoch: {checkpoint.get('epoch', 'N/A') + 1}")
    print(f"  Best AUPRC during training: {checkpoint.get('best_auprc', 'N/A'):.4f}")
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Test on test set
    print(f"\n{'='*80}")
    print(f"TESTING MODEL ON TEST SET")
    print(f"{'='*80}\n")
    
    test_loss = 0.0
    test_labels, test_preds = [], []
    test_record_ids = []
    test_probs = []
    
    with torch.no_grad():
        for input_dict, y in test_loader:
            # Always load ECG/PPG if available
            ecg = input_dict.get('ecg', None)
            ppg = input_dict.get('ppg', None)
            if ecg is not None:
                ecg = ecg.to(device)  # [B, N, L_ecg]
            if ppg is not None:
                ppg = ppg.to(device)  # [B, N, L_ppg]
            
            mask = input_dict['mask'].to(device)  # [B, N]
            hrv = input_dict['hrv'].to(device)  # [B, 4]
            sleep_stage = input_dict['sleep_stage'].to(device)  # [B, N]
            apnea = input_dict['apnea_label'].to(device)  # [B, N]
            record_ids = input_dict['record_id']  # List of record IDs (bag_id)
            y = y.view(-1).to(device)  # [B]
            
            logits = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea).squeeze(-1)
            loss = criterion(logits, y)
            test_loss += loss.item() * y.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            test_labels.extend(y.cpu().numpy())
            test_preds.extend(preds)
            test_probs.extend(probs)
            test_record_ids.extend(record_ids)
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = accuracy_score(test_labels, test_preds)
    test_bal = balanced_accuracy_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else float('nan')
    test_auprc = average_precision_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    
    # Print results
    print(f"{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test Precision : {test_precision:.4f}")
    print(f"Test Recall    : {test_recall:.4f}")
    print(f"Test F1 Score  : {test_f1:.4f}")
    print(f"Test Balanced Acc: {test_bal:.4f}")
    print(f"Test MCC       : {test_mcc:.4f}")
    print(f"Test AUROC     : {test_auc:.4f}")
    print(f"Test AUPRC     : {test_auprc:.4f}")
    print(f"{'='*80}\n")
    
    # Return metrics dictionary
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_balanced_accuracy': test_bal,
        'test_mcc': test_mcc,
        'test_auc': test_auc,
        'test_auprc': test_auprc,
        'test_labels': test_labels,
        'test_predictions': test_preds,
        'test_probabilities': test_probs,
        'test_record_ids': test_record_ids
    }
    
    return results


def train_model_loso(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    save_path: str = "best_model.pt",
    patience: int = 15,
    test_loader: Optional[DataLoader] = None,
    scaler: Optional[GradScaler] = None,
    hrv_slice: Optional[Tuple[int, int]] = None,
) -> Tuple[int, float]:
    """
    Train the model for LOSO validation with early stopping on internal validation set.
    
    This function uses an internal validation set (NOT the test patient) for:
    - Early stopping (based on validation loss)
    - Model selection (saving best checkpoint based on lowest validation loss)
    
    Args:
        model: The neural network model
        scaler: Gradient scaler for mixed precision training
        train_loader: DataLoader for training data
        val_loader: DataLoader for internal validation data (NOT test data)
        num_epochs: Maximum number of training epochs
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (ReduceLROnPlateau, monitors validation loss; use mode='min')
        device: Device to run training on (cuda/cpu)
        save_path: Path to save the best model
        patience: Number of epochs to wait before early stopping
    
    Returns:
        Tuple of (best_epoch, best_val_loss)
    """
    model.to(device)
    best_val_loss = float('inf')  # Minimize validation loss
    best_epoch = -1
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        use_sam = getattr(optimizer, "first_step", None) is not None and getattr(optimizer, "second_step", None) is not None
        no_sync_ctx = model.no_sync() if getattr(model, "no_sync", None) else contextlib.nullcontext()

        # Gradient debugging: check first batch of first epoch
        check_gradients = (epoch == 0)
        check_shapes = (epoch == 0)
        batch_idx = 0

        for input_dict, y in train_loader:
            ecg = input_dict.get('ecg', None)
            ppg = input_dict.get('ppg', None)
            if ecg is not None:
                ecg = ecg.to(device)
            if ppg is not None:
                ppg = ppg.to(device)
            
            mask = input_dict['mask'].to(device)
            hrv_full = input_dict['hrv'].to(device)  # [B, 4] = [ecg_hf, ecg_lf, ppg_hf, ppg_lf]
            # Slice: ECG uses dims 0:2 (ECG HRV), PPG uses dims 2:4 (PRV), fusion uses full 4-dim
            if hrv_slice is not None:
                hrv = hrv_full[:, hrv_slice[0]:hrv_slice[1]]
            else:
                hrv = hrv_full
            sleep_stage = input_dict['sleep_stage'].to(device)
            apnea = input_dict['apnea_label'].to(device)
            y = y.view(-1).to(device)

            optimizer.zero_grad()

            if use_sam:
                # SAM: full precision (no AMP) to avoid GradScaler/NaN issues with two-step update
                enable_running_stats(model)
                output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                
                # Shape debugging: check for broadcasting bugs (first batch of first epoch only)
                if check_shapes and batch_idx == 0:
                    print(f"\n[SHAPE CHECK] Epoch {epoch+1}, Batch {batch_idx} (LOSO SAM):")
                    print(f"  logits shape: {logits.shape}")
                    print(f"  y shape: {y.shape}")
                    if logits.shape != y.shape:
                        print(f"  ⚠️ WARNING: Shape mismatch! logits={logits.shape}, y={y.shape}")
                        if logits.shape == (logits.size(0), 1) and y.shape == (y.size(0),):
                            print(f"  → Applying .view(-1) fix to logits")
                            logits = logits.view(-1)
                        elif logits.shape == (logits.size(0),) and y.shape == (y.size(0), 1):
                            print(f"  → Applying .view(-1) fix to y")
                            y = y.view(-1)
                    else:
                        print(f"  ✓ Shapes match correctly")
                    check_shapes = False  # Only check once
                
                loss = criterion(logits, y)
                with no_sync_ctx:
                    loss.backward()
                
                # Gradient debugging: check if gradients are flowing (first batch of first epoch only)
                if check_gradients and batch_idx == 0:
                    grad_info = []
                    total_params = 0
                    params_with_grad = 0
                    zero_grad_count = 0
                    for name, param in model.named_parameters():
                        total_params += 1
                        if param.grad is not None:
                            params_with_grad += 1
                            grad_norm = param.grad.norm().item()
                            if grad_norm == 0.0:
                                zero_grad_count += 1
                            grad_info.append((name, grad_norm))
                        else:
                            grad_info.append((name, None))
                    
                    print(f"\n[GRADIENT CHECK] Epoch {epoch+1}, Batch {batch_idx} (SAM first step):")
                    print(f"  Total parameters: {total_params}")
                    print(f"  Parameters with gradients: {params_with_grad}")
                    print(f"  Parameters with zero gradients: {zero_grad_count}")
                    if params_with_grad > 0:
                        print(f"  Sample gradients (first 5):")
                        for name, grad_norm in grad_info[:5]:
                            if grad_norm is not None:
                                print(f"    {name}: {grad_norm:.6f}")
                    else:
                        print(f"  ⚠️ WARNING: No gradients found! Backpropagation may be broken.")
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                output2 = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                logits2 = output2[0].squeeze(-1) if isinstance(output2, tuple) else output2.squeeze(-1)
                loss2 = criterion(logits2, y)
                loss2.backward()
                
                # Check gradients after second backward (SAM)
                if check_gradients and batch_idx == 0:
                    grad_info = []
                    total_params = 0
                    params_with_grad = 0
                    zero_grad_count = 0
                    for name, param in model.named_parameters():
                        total_params += 1
                        if param.grad is not None:
                            params_with_grad += 1
                            grad_norm = param.grad.norm().item()
                            if grad_norm == 0.0:
                                zero_grad_count += 1
                            grad_info.append((name, grad_norm))
                        else:
                            grad_info.append((name, None))
                    
                    print(f"[GRADIENT CHECK] Epoch {epoch+1}, Batch {batch_idx} (SAM second step):")
                    print(f"  Total parameters: {total_params}")
                    print(f"  Parameters with gradients: {params_with_grad}")
                    print(f"  Parameters with zero gradients: {zero_grad_count}")
                    if params_with_grad > 0:
                        print(f"  Sample gradients (first 5):")
                        for name, grad_norm in grad_info[:5]:
                            if grad_norm is not None:
                                print(f"    {name}: {grad_norm:.6f}")
                    check_gradients = False  # Only check once
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step(zero_grad=True)
            else:
                # No SAM: full precision if scaler is None, else AMP
                if scaler is not None:
                    with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                        output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                        logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                    logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                    
                    # Shape debugging: check for broadcasting bugs (first batch of first epoch only)
                    if check_shapes and batch_idx == 0:
                        print(f"\n[SHAPE CHECK] Epoch {epoch+1}, Batch {batch_idx} (LOSO):")
                        print(f"  logits shape: {logits.shape}")
                        print(f"  y shape: {y.shape}")
                        if logits.shape != y.shape:
                            print(f"  ⚠️ WARNING: Shape mismatch! logits={logits.shape}, y={y.shape}")
                            if logits.shape == (logits.size(0), 1) and y.shape == (y.size(0),):
                                print(f"  → Applying .view(-1) fix to logits")
                                logits = logits.view(-1)
                            elif logits.shape == (logits.size(0),) and y.shape == (y.size(0), 1):
                                print(f"  → Applying .view(-1) fix to y")
                                y = y.view(-1)
                        else:
                            print(f"  ✓ Shapes match correctly")
                        check_shapes = False  # Only check once
                    
                    loss = criterion(logits, y)
                    loss.backward()
                    
                    # Gradient debugging: check if gradients are flowing (first batch of first epoch only)
                    if check_gradients and batch_idx == 0:
                        grad_info = []
                        total_params = 0
                        params_with_grad = 0
                        zero_grad_count = 0
                        for name, param in model.named_parameters():
                            total_params += 1
                            if param.grad is not None:
                                params_with_grad += 1
                                grad_norm = param.grad.norm().item()
                                if grad_norm == 0.0:
                                    zero_grad_count += 1
                                grad_info.append((name, grad_norm))
                            else:
                                grad_info.append((name, None))
                        
                        print(f"\n[GRADIENT CHECK] Epoch {epoch+1}, Batch {batch_idx}:")
                        print(f"  Total parameters: {total_params}")
                        print(f"  Parameters with gradients: {params_with_grad}")
                        print(f"  Parameters with zero gradients: {zero_grad_count}")
                        if params_with_grad > 0:
                            # Show first few layers with gradients
                            print(f"  Sample gradients (first 5):")
                            for name, grad_norm in grad_info[:5]:
                                if grad_norm is not None:
                                    print(f"    {name}: {grad_norm:.6f}")
                        else:
                            print(f"  ⚠️ WARNING: No gradients found! Backpropagation may be broken.")
                        check_gradients = False  # Only check once
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += loss.item() * y.size(0)
            batch_idx += 1
            probs = torch.sigmoid(logits.detach()).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_labels, val_preds, val_probs = [], [], []
        
        with torch.no_grad():
            for input_dict, y in val_loader:
                ecg = input_dict.get('ecg', None)
                ppg = input_dict.get('ppg', None)
                if ecg is not None:
                    ecg = ecg.to(device)
                if ppg is not None:
                    ppg = ppg.to(device)
                
                mask = input_dict['mask'].to(device)
                hrv_full = input_dict['hrv'].to(device)
                if hrv_slice is not None:
                    hrv = hrv_full[:, hrv_slice[0]:hrv_slice[1]]
                else:
                    hrv = hrv_full
                sleep_stage = input_dict['sleep_stage'].to(device)
                apnea = input_dict['apnea_label'].to(device)
                y = y.view(-1).to(device)

                output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_labels.extend(y.cpu().numpy())
                val_preds.extend(preds)
                val_probs.extend(probs)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_bal = balanced_accuracy_score(val_labels, val_preds)
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        val_probs_arr = np.asarray(val_probs, dtype=np.float64)
        nan_count = int(np.isnan(val_probs_arr).sum())
        if nan_count > 0:
            print(f"[WARNING] Validation probs contain {nan_count} NaN(s) at epoch {epoch + 1}. Replacing with 0.5 for metrics.")
            val_probs_arr = np.nan_to_num(val_probs_arr, nan=0.5, posinf=1.0, neginf=0.0)
        elif epoch == 0:
            print("Validation probs: no NaN.")
        val_auprc = average_precision_score(val_labels, val_probs_arr)

        # Calculate test accuracy for monitoring (optional, does not influence training)
        test_acc = None
        if test_loader is not None:
            model.eval()
            test_labels, test_preds = [], []
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
                    if hrv_slice is not None:
                        hrv = hrv_full[:, hrv_slice[0]:hrv_slice[1]]
                    else:
                        hrv = hrv_full
                    sleep_stage = input_dict['sleep_stage'].to(device)
                    apnea = input_dict['apnea_label'].to(device)
                    y = y.view(-1).to(device)

                    output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
                    logits = output[0].squeeze(-1) if isinstance(output, tuple) else output.squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    test_labels.extend(y.cpu().numpy())
                    test_preds.extend(preds)
            test_acc = accuracy_score(test_labels, test_preds)

        # Update learning rate scheduler (ReduceLROnPlateau uses step(val_loss); use mode='min')
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # === LOG - Print every epoch ===
        log_str = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        log_str += f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val BalAcc: {val_bal:.4f} | Val AUPRC: {val_auprc:.4f}"
        if test_acc is not None:
            log_str += f" | Test Acc: {test_acc:.4f}"
        log_str += f" | LR: {current_lr:.6f}"
        print(log_str)

        # Save best model based on lowest validation loss (minimize)
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_balanced_accuracy': val_bal,
                'best_val_auprc': val_auprc,
                'best_val_mcc': val_mcc,
            }, save_path)
            improvement_str = f"  ✓ Saved best model (epoch {epoch+1}, val_loss: {val_loss:.4f}, improvement: {improvement:.4f}, val_bal: {val_bal:.4f}, val_auprc: {val_auprc:.4f})"
            if test_acc is not None:
                improvement_str += f", test_acc: {test_acc:.4f}"
            print(improvement_str)
        else:
            epochs_no_improve += 1
            if epochs_no_improve > 0 and (epoch + 1) % 10 == 0:
                print(f"  No improvement for {epochs_no_improve} epochs (best: epoch {best_epoch + 1}, val_loss: {best_val_loss:.4f})")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epochs_no_improve} epochs without improvement.")
                print(f"Best model was at epoch {best_epoch + 1} with validation loss: {best_val_loss:.4f}")
                break

    return best_epoch, best_val_loss

