"""
PyTorch Dataset class for ECG-PPG bag-level data.

This module implements a Dataset class for Multiple Instance Learning (MIL)
on ECG and PPG signals, where each bag contains multiple segments (instances).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing import Dict, List, Optional, Tuple


# Sleep stage mapping: string -> integer
# Matches the embedding layer which expects 6 classes (0-5) with padding_idx=0
SLEEP_STAGE_MAPPING = {
    'SLEEP-S0': 0,
    'SLEEP-S1': 1,
    'SLEEP-S2': 2,
    'SLEEP-S3': 3,
    'SLEEP-S4': 4,
    'SLEEP-REM': 5
}

# Default value for unknown/missing sleep stages (will be treated as padding)
UNKNOWN_SLEEP_STAGE = 0


class RandomShift:
    """Random time shift augmentation for 1D signals."""
    
    def __init__(self, max_shift_seconds: float = 2.0, sampling_rate: int = 200):
        """
        Args:
            max_shift_seconds: Maximum shift in seconds
            sampling_rate: Sampling rate of the signal (Hz)
        """
        self.max_shift = int(max_shift_seconds * sampling_rate)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random circular shift to the signal."""
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
        return torch.roll(x, shifts=shift, dims=-1)


class GaussianNoise:
    """Add Gaussian noise to the signal."""
    
    def __init__(self, noise_level: float = 0.01):
        """
        Args:
            noise_level: Standard deviation of noise as fraction of signal std
        """
        self.noise_level = noise_level
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the signal."""
        std = x.std(unbiased=False)
        return x + torch.randn_like(x) * (std * self.noise_level)


class AmplitudeScale:
    """Random amplitude scaling augmentation."""
    
    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1):
        """
        Args:
            min_scale: Minimum scaling factor
            max_scale: Maximum scaling factor
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling."""
        factor = torch.empty(1).uniform_(self.min_scale, self.max_scale)
        return x * factor


class ECGPPGBagDataset(Dataset):
    """
    Dataset class for ECG-PPG bag-level data with Multiple Instance Learning.
    
    Each bag contains multiple segments (instances) of ECG and PPG signals,
    along with sleep stage labels, apnea labels, and HRV features.
    
    The dataset expects bags in the format:
    {
        'ecg_segments': np.ndarray of shape (N, L_ecg),
        'ppg_segments': np.ndarray of shape (N, L_ppg),
        'apnea_labels': np.ndarray of shape (N,),
        'sleep_stages': List[str] of length N,
        'ecg_hrv': Optional[Dict] with 'hrv_hf' and 'hrv_lf' keys,
        'ppg_hrv': Optional[Dict] with 'hrv_hf' and 'hrv_lf' keys,
        'file_path': str
    }
    """
    
    def __init__(
        self,
        bags: List[Dict],
        cvd_labels: Optional[Dict[str, int]] = None,
        augment: bool = False,
        max_instances: int = 120,
        min_instances: int = 15,
        ecg_fs: int = 200,
        ppg_fs: int = 100,
    ):
        """
        Initialize the dataset.
        
        Args:
            bags: List of bag dictionaries from load_data_from_entries()
            cvd_labels: Optional dict mapping bag_id (file_path) to CVD label (0 or 1).
                       If None, CVD labels will be inferred from the data structure.
            augment: Whether to apply data augmentation during training
            max_instances: Maximum number of segments per bag (for padding/truncation)
            min_instances: Minimum number of segments required to include a bag
            ecg_fs: ECG sampling rate (Hz) for augmentation
            ppg_fs: PPG sampling rate (Hz) for augmentation
        """
        self.max_instances = max_instances
        self.min_instances = min_instances
        self.augment = augment
        self.ecg_fs = ecg_fs
        self.ppg_fs = ppg_fs
        
        # Filter bags by minimum instances and create bag_id mapping
        self.bag_ids = []
        self.bags = {}
        self.cvd_labels = {}
        
        for bag in bags:
            # Use file_path as bag_id (or generate one if not present)
            bag_id = bag.get('file_path', f"bag_{len(self.bag_ids)}")
            
            # Get number of segments
            n_segments = len(bag['ecg_segments']) if isinstance(bag['ecg_segments'], np.ndarray) else 0
            
            # Filter by minimum instances
            if n_segments < self.min_instances:
                continue
            
            self.bag_ids.append(bag_id)
            self.bags[bag_id] = bag
            
            # Assign CVD label
            if cvd_labels is not None:
                self.cvd_labels[bag_id] = cvd_labels.get(bag_id, 0)
            else:
                # Default: assume all bags are from the same class (will be set externally)
                self.cvd_labels[bag_id] = 0
        
        # Setup augmentation pipeline
        if self.augment:
            self.ecg_augment = Compose([
                GaussianNoise(noise_level=0.01),
                AmplitudeScale(min_scale=0.9, max_scale=1.1),
                RandomShift(max_shift_seconds=2.0, sampling_rate=ecg_fs),
            ])
            
            self.ppg_augment = Compose([
                GaussianNoise(noise_level=0.01),
                AmplitudeScale(min_scale=0.9, max_scale=1.1),
                RandomShift(max_shift_seconds=2.0, sampling_rate=ppg_fs),
            ])
    
    def __len__(self) -> int:
        """Return the number of bags in the dataset."""
        return len(self.bag_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single bag from the dataset.
        
        Args:
            idx: Index of the bag
            
        Returns:
            Tuple of (input_dict, cvd_label) where:
            - input_dict contains:
                - 'ecg': torch.Tensor of shape (max_instances, L_ecg)
                - 'ppg': torch.Tensor of shape (max_instances, L_ppg)
                - 'sleep_stage': torch.Tensor of shape (max_instances,) with integer labels
                - 'apnea_label': torch.Tensor of shape (max_instances,) with binary labels
                - 'mask': torch.Tensor of shape (max_instances,) indicating valid segments
                - 'hrv': torch.Tensor of shape (hrv_dim,) with HRV features
                - 'record_id': str (patient/record identifier)
            - cvd_label: torch.Tensor scalar (0 or 1)
        """
        bag_id = self.bag_ids[idx]
        bag = self.bags[bag_id]
        
        # Extract segments
        ecg_segments = bag['ecg_segments']  # Shape: (N, L_ecg)
        ppg_segments = bag['ppg_segments']  # Shape: (N, L_ppg)
        apnea_labels = bag['apnea_labels']  # Shape: (N,)
        sleep_stages = bag['sleep_stages']  # List[str] of length N
        
        # Ensure arrays are numpy arrays
        if not isinstance(ecg_segments, np.ndarray):
            ecg_segments = np.array(ecg_segments)
        if not isinstance(ppg_segments, np.ndarray):
            ppg_segments = np.array(ppg_segments)
        if not isinstance(apnea_labels, np.ndarray):
            apnea_labels = np.array(apnea_labels)
        
        n = len(ecg_segments)
        
        # Down-sample if too many instances
        if n > self.max_instances:
            indices = np.random.choice(n, self.max_instances, replace=False)
            ecg_segments = ecg_segments[indices]
            ppg_segments = ppg_segments[indices]
            apnea_labels = apnea_labels[indices]
            sleep_stages = [sleep_stages[i] for i in indices]
            n = self.max_instances
        
        # Convert sleep stages from strings to integers
        sleep_stage_ints = []
        for stage in sleep_stages:
            stage_int = SLEEP_STAGE_MAPPING.get(stage, UNKNOWN_SLEEP_STAGE)
            sleep_stage_ints.append(stage_int)
        sleep_stage_ints = np.array(sleep_stage_ints, dtype=np.int64)
        
        # Pad to max_instances
        pad_n = self.max_instances - n
        
        if pad_n > 0:
            # Get shapes for padding
            ecg_pad_shape = (pad_n,) + ecg_segments.shape[1:]
            ppg_pad_shape = (pad_n,) + ppg_segments.shape[1:]
            
            # Pad arrays
            ecg_segments = np.concatenate([
                ecg_segments,
                np.zeros(ecg_pad_shape, dtype=ecg_segments.dtype)
            ], axis=0)
            
            ppg_segments = np.concatenate([
                ppg_segments,
                np.zeros(ppg_pad_shape, dtype=ppg_segments.dtype)
            ], axis=0)
            
            apnea_labels = np.concatenate([
                apnea_labels,
                np.zeros(pad_n, dtype=apnea_labels.dtype)
            ], axis=0)
            
            sleep_stage_ints = np.concatenate([
                sleep_stage_ints,
                np.zeros(pad_n, dtype=sleep_stage_ints.dtype)
            ], axis=0)
        
        # Create mask for valid segments
        mask = torch.zeros(self.max_instances, dtype=torch.bool)
        mask[:n] = True
        
        # Convert to tensors
        ecg_t = torch.from_numpy(ecg_segments).float()
        ppg_t = torch.from_numpy(ppg_segments).float()
        apnea_t = torch.from_numpy(apnea_labels).float()
        sleep_stage_t = torch.from_numpy(sleep_stage_ints).long()
        
        # Apply augmentation if enabled
        if self.augment:
            ecg_t = torch.stack([self.ecg_augment(x) for x in ecg_t])
            ppg_t = torch.stack([self.ppg_augment(x) for x in ppg_t])
        
        # Extract HRV features
        # Combine ECG and PPG HRV features into a single vector
        # Format: [ecg_hrv_hf, ecg_hrv_lf, ppg_hrv_hf, ppg_hrv_lf]
        hrv_features = []
        
        if bag.get('ecg_hrv') is not None:
            hrv_features.append(bag['ecg_hrv']['hrv_hf'])
            hrv_features.append(bag['ecg_hrv']['hrv_lf'])
        else:
            # Use zeros if missing
            hrv_features.extend([0.0, 0.0])
        
        if bag.get('ppg_hrv') is not None:
            hrv_features.append(bag['ppg_hrv']['hrv_hf'])
            hrv_features.append(bag['ppg_hrv']['hrv_lf'])
        else:
            # Use zeros if missing
            hrv_features.extend([0.0, 0.0])
        
        hrv_t = torch.tensor(hrv_features, dtype=torch.float32)
        
        # Extract record_id from file_path if possible
        record_id = bag.get('file_path', bag_id)
        # Try to extract patient ID from path
        if isinstance(record_id, str):
            # Extract filename without extension
            import os
            record_id = os.path.splitext(os.path.basename(record_id))[0]
        
        # CVD label
        cvd_t = torch.tensor(self.cvd_labels[bag_id], dtype=torch.float32)
        
        return {
            'ecg': ecg_t,
            'ppg': ppg_t,
            'sleep_stage': sleep_stage_t,
            'apnea_label': apnea_t,
            'mask': mask,
            'hrv': hrv_t,
            'record_id': record_id
        }, cvd_t


def create_cvd_label_mapping(
    bags: List[Dict],
    entries: List[Dict],
    data_dir: str,
) -> Dict[str, int]:
    """
    Create a mapping from bag file_path to CVD label (0 or 1).
    
    This function maps each bag's file_path back to its corresponding entry
    to extract the has_cvd label. The mapping is done by matching the folder
    structure and file naming convention.
    
    Args:
        bags: List of bag dictionaries with 'file_path' keys
        entries: List of entry dictionaries with 'has_cvd', 'patient_id', 'follow_up' keys
        data_dir: Base directory where bags are stored
    
    Returns:
        Dictionary mapping bag file_path to CVD label (0 or 1)
    """
    from pathlib import Path
    from data_loader import get_folder_name
    
    # Create mapping from folder name to has_cvd label
    folder_to_cvd = {}
    for entry in entries:
        folder_name = get_folder_name(entry)
        has_cvd = entry.get('has_cvd', 'no')
        
        # Convert has_cvd to integer (0 or 1)
        if isinstance(has_cvd, str):
            cvd_label = 1 if has_cvd.lower() == 'yes' else 0
        else:
            cvd_label = int(has_cvd) if has_cvd else 0
        
        folder_to_cvd[folder_name] = cvd_label
    
    # Map each bag's file_path to CVD label
    cvd_labels = {}
    data_dir_path = Path(data_dir)
    
    for bag in bags:
        file_path = bag.get('file_path', '')
        if not file_path:
            continue
        
        # Extract folder name from file path
        # Path structure: data_dir / folder_name / bag_file.h5
        file_path_obj = Path(file_path)
        
        # Get parent folder name
        folder_name = file_path_obj.parent.name
        
        # Look up CVD label
        cvd_label = folder_to_cvd.get(folder_name, 0)
        cvd_labels[file_path] = cvd_label
    
    return cvd_labels


def create_datasets_from_splits(
    train_bags: List[Dict],
    test_bags: List[Dict],
    train_entries: Optional[List[Dict]] = None,
    test_entries: Optional[List[Dict]] = None,
    train_cvd_labels: Optional[Dict[str, int]] = None,
    test_cvd_labels: Optional[Dict[str, int]] = None,
    data_dir: Optional[str] = None,
    augment_train: bool = True,
    max_instances: int = 120,
    min_instances: int = 15,
    ecg_fs: int = 200,
    ppg_fs: int = 100,
) -> Tuple[ECGPPGBagDataset, ECGPPGBagDataset]:
    """
    Create train and test datasets from bag lists.
    
    Args:
        train_bags: List of training bag dictionaries
        test_bags: List of test bag dictionaries
        train_entries: Optional list of training entries (for automatic CVD label extraction)
        test_entries: Optional list of test entries (for automatic CVD label extraction)
        train_cvd_labels: Optional dict mapping bag file_path to CVD label for training.
                         If None and train_entries provided, will be auto-generated.
        test_cvd_labels: Optional dict mapping bag file_path to CVD label for testing.
                         If None and test_entries provided, will be auto-generated.
        data_dir: Base data directory (required if auto-generating CVD labels)
        augment_train: Whether to apply augmentation to training set
        max_instances: Maximum number of segments per bag
        min_instances: Minimum number of segments required
        ecg_fs: ECG sampling rate
        ppg_fs: PPG sampling rate
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Auto-generate CVD labels if not provided but entries are available
    if train_cvd_labels is None and train_entries is not None and data_dir is not None:
        train_cvd_labels = create_cvd_label_mapping(train_bags, train_entries, data_dir)
    
    if test_cvd_labels is None and test_entries is not None and data_dir is not None:
        test_cvd_labels = create_cvd_label_mapping(test_bags, test_entries, data_dir)
    
    train_dataset = ECGPPGBagDataset(
        bags=train_bags,
        cvd_labels=train_cvd_labels,
        augment=augment_train,
        max_instances=max_instances,
        min_instances=min_instances,
        ecg_fs=ecg_fs,
        ppg_fs=ppg_fs,
    )
    
    test_dataset = ECGPPGBagDataset(
        bags=test_bags,
        cvd_labels=test_cvd_labels,
        augment=False,  # Never augment test set
        max_instances=max_instances,
        min_instances=min_instances,
        ecg_fs=ecg_fs,
        ppg_fs=ppg_fs,
    )
    
    return train_dataset, test_dataset

