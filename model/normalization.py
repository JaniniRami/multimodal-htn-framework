"""
Normalization utilities for z-score normalization of data.
"""

import numpy as np


def calculate_normalization_coefficients(train_bags):
    """
    Calculate z-score normalization coefficients (mean and std) from training data only.
    
    Args:
        train_bags: List of bag dictionaries from training set
    
    Returns:
        dict: Dictionary with normalization coefficients:
            - 'ecg_segments': {'mean': float, 'std': float}
            - 'ppg_segments': {'mean': float, 'std': float}
            - 'ecg_hrv': {'hrv_hf': {'mean': float, 'std': float}, 'hrv_lf': {'mean': float, 'std': float}}
            - 'ppg_hrv': {'hrv_hf': {'mean': float, 'std': float}, 'hrv_lf': {'mean': float, 'std': float}}
    """
    # Collect all ECG segments
    all_ecg_segments = []
    for bag in train_bags:
        all_ecg_segments.append(bag['ecg_segments'])
    
    # Collect all PPG segments
    all_ppg_segments = []
    for bag in train_bags:
        all_ppg_segments.append(bag['ppg_segments'])
    
    # Calculate coefficients for segments
    if len(all_ecg_segments) > 0:
        ecg_concatenated = np.concatenate([seg.flatten() for seg in all_ecg_segments])
        ecg_segments_mean = np.mean(ecg_concatenated)
        ecg_segments_std = np.std(ecg_concatenated)
        # Avoid division by zero
        if ecg_segments_std == 0:
            ecg_segments_std = 1.0
    else:
        ecg_segments_mean = 0.0
        ecg_segments_std = 1.0
    
    if len(all_ppg_segments) > 0:
        ppg_concatenated = np.concatenate([seg.flatten() for seg in all_ppg_segments])
        ppg_segments_mean = np.mean(ppg_concatenated)
        ppg_segments_std = np.std(ppg_concatenated)
        # Avoid division by zero
        if ppg_segments_std == 0:
            ppg_segments_std = 1.0
    else:
        ppg_segments_mean = 0.0
        ppg_segments_std = 1.0
    
    # Collect HRV features
    ecg_hrv_hf_values = []
    ecg_hrv_lf_values = []
    ppg_hrv_hf_values = []
    ppg_hrv_lf_values = []
    
    for bag in train_bags:
        if bag['ecg_hrv'] is not None:
            ecg_hrv_hf_values.append(bag['ecg_hrv']['hrv_hf'])
            ecg_hrv_lf_values.append(bag['ecg_hrv']['hrv_lf'])
        if bag['ppg_hrv'] is not None:
            ppg_hrv_hf_values.append(bag['ppg_hrv']['hrv_hf'])
            ppg_hrv_lf_values.append(bag['ppg_hrv']['hrv_lf'])
    
    # Calculate coefficients for ECG HRV features
    if len(ecg_hrv_hf_values) > 0:
        ecg_hrv_hf_mean = np.mean(ecg_hrv_hf_values)
        ecg_hrv_hf_std = np.std(ecg_hrv_hf_values)
        if ecg_hrv_hf_std == 0:
            ecg_hrv_hf_std = 1.0
    else:
        ecg_hrv_hf_mean = 0.0
        ecg_hrv_hf_std = 1.0
    
    if len(ecg_hrv_lf_values) > 0:
        ecg_hrv_lf_mean = np.mean(ecg_hrv_lf_values)
        ecg_hrv_lf_std = np.std(ecg_hrv_lf_values)
        if ecg_hrv_lf_std == 0:
            ecg_hrv_lf_std = 1.0
    else:
        ecg_hrv_lf_mean = 0.0
        ecg_hrv_lf_std = 1.0
    
    # Calculate coefficients for PPG HRV features
    if len(ppg_hrv_hf_values) > 0:
        ppg_hrv_hf_mean = np.mean(ppg_hrv_hf_values)
        ppg_hrv_hf_std = np.std(ppg_hrv_hf_values)
        if ppg_hrv_hf_std == 0:
            ppg_hrv_hf_std = 1.0
    else:
        ppg_hrv_hf_mean = 0.0
        ppg_hrv_hf_std = 1.0
    
    if len(ppg_hrv_lf_values) > 0:
        ppg_hrv_lf_mean = np.mean(ppg_hrv_lf_values)
        ppg_hrv_lf_std = np.std(ppg_hrv_lf_values)
        if ppg_hrv_lf_std == 0:
            ppg_hrv_lf_std = 1.0
    else:
        ppg_hrv_lf_mean = 0.0
        ppg_hrv_lf_std = 1.0
    
    return {
        'ecg_segments': {'mean': ecg_segments_mean, 'std': ecg_segments_std},
        'ppg_segments': {'mean': ppg_segments_mean, 'std': ppg_segments_std},
        'ecg_hrv': {
            'hrv_hf': {'mean': ecg_hrv_hf_mean, 'std': ecg_hrv_hf_std},
            'hrv_lf': {'mean': ecg_hrv_lf_mean, 'std': ecg_hrv_lf_std}
        },
        'ppg_hrv': {
            'hrv_hf': {'mean': ppg_hrv_hf_mean, 'std': ppg_hrv_hf_std},
            'hrv_lf': {'mean': ppg_hrv_lf_mean, 'std': ppg_hrv_lf_std}
        }
    }


def apply_normalization(bags, norm_coefficients):
    """
    Apply z-score normalization to bags using pre-computed coefficients.
    
    Args:
        bags: List of bag dictionaries
        norm_coefficients: Dictionary with normalization coefficients from calculate_normalization_coefficients()
    
    Returns:
        list: List of normalized bag dictionaries (modified in-place)
    """
    for bag in bags:
        # Normalize ECG segments
        bag['ecg_segments'] = (bag['ecg_segments'] - norm_coefficients['ecg_segments']['mean']) / norm_coefficients['ecg_segments']['std']
        
        # Normalize PPG segments
        bag['ppg_segments'] = (bag['ppg_segments'] - norm_coefficients['ppg_segments']['mean']) / norm_coefficients['ppg_segments']['std']
        
        # Normalize ECG HRV features
        if bag['ecg_hrv'] is not None:
            bag['ecg_hrv']['hrv_hf'] = (bag['ecg_hrv']['hrv_hf'] - norm_coefficients['ecg_hrv']['hrv_hf']['mean']) / norm_coefficients['ecg_hrv']['hrv_hf']['std']
            bag['ecg_hrv']['hrv_lf'] = (bag['ecg_hrv']['hrv_lf'] - norm_coefficients['ecg_hrv']['hrv_lf']['mean']) / norm_coefficients['ecg_hrv']['hrv_lf']['std']
        
        # Normalize PPG HRV features
        if bag['ppg_hrv'] is not None:
            bag['ppg_hrv']['hrv_hf'] = (bag['ppg_hrv']['hrv_hf'] - norm_coefficients['ppg_hrv']['hrv_hf']['mean']) / norm_coefficients['ppg_hrv']['hrv_hf']['std']
            bag['ppg_hrv']['hrv_lf'] = (bag['ppg_hrv']['hrv_lf'] - norm_coefficients['ppg_hrv']['hrv_lf']['mean']) / norm_coefficients['ppg_hrv']['hrv_lf']['std']
    
    return bags

