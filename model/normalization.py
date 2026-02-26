"""
Normalization utilities for signal processing.

- Instance Normalization: Each 30-second segment is normalized independently to have mean=0 and std=1.
  This is applied to ECG and PPG segments.
- Robust Scaling: Used for HRV features (scalar values) using median and IQR.
"""

import numpy as np


def calculate_normalization_coefficients(train_bags):
    """
    Calculate robust scaling normalization coefficients (median and IQR) for HRV features only.
    
    For ECG and PPG segments, we use Instance Normalization (each segment normalized independently),
    so no global coefficients are needed. Only HRV features need pre-computed coefficients.
    
    Robust scaling uses:
    - Median instead of mean (resistant to outliers)
    - IQR (Interquartile Range = Q3 - Q1) instead of std (describes normal variation)
    
    Args:
        train_bags: List of bag dictionaries from training set
    
    Returns:
        dict: Dictionary with normalization coefficients for HRV features:
            - 'ecg_hrv': {'hrv_hf': {'median': float, 'iqr': float}, 'hrv_lf': {'median': float, 'iqr': float}}
            - 'ppg_hrv': {'hrv_hf': {'median': float, 'iqr': float}, 'hrv_lf': {'median': float, 'iqr': float}}
    """
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
    
    # Calculate robust scaling coefficients for ECG HRV features
    if len(ecg_hrv_hf_values) > 0:
        ecg_hrv_hf_median = np.median(ecg_hrv_hf_values)
        q1_ecg_hf = np.percentile(ecg_hrv_hf_values, 25)
        q3_ecg_hf = np.percentile(ecg_hrv_hf_values, 75)
        ecg_hrv_hf_iqr = q3_ecg_hf - q1_ecg_hf
        if ecg_hrv_hf_iqr == 0:
            ecg_hrv_hf_iqr = 1.0
    else:
        ecg_hrv_hf_median = 0.0
        ecg_hrv_hf_iqr = 1.0
    
    if len(ecg_hrv_lf_values) > 0:
        ecg_hrv_lf_median = np.median(ecg_hrv_lf_values)
        q1_ecg_lf = np.percentile(ecg_hrv_lf_values, 25)
        q3_ecg_lf = np.percentile(ecg_hrv_lf_values, 75)
        ecg_hrv_lf_iqr = q3_ecg_lf - q1_ecg_lf
        if ecg_hrv_lf_iqr == 0:
            ecg_hrv_lf_iqr = 1.0
    else:
        ecg_hrv_lf_median = 0.0
        ecg_hrv_lf_iqr = 1.0
    
    # Calculate robust scaling coefficients for PPG HRV features
    if len(ppg_hrv_hf_values) > 0:
        ppg_hrv_hf_median = np.median(ppg_hrv_hf_values)
        q1_ppg_hf = np.percentile(ppg_hrv_hf_values, 25)
        q3_ppg_hf = np.percentile(ppg_hrv_hf_values, 75)
        ppg_hrv_hf_iqr = q3_ppg_hf - q1_ppg_hf
        if ppg_hrv_hf_iqr == 0:
            ppg_hrv_hf_iqr = 1.0
    else:
        ppg_hrv_hf_median = 0.0
        ppg_hrv_hf_iqr = 1.0
    
    if len(ppg_hrv_lf_values) > 0:
        ppg_hrv_lf_median = np.median(ppg_hrv_lf_values)
        q1_ppg_lf = np.percentile(ppg_hrv_lf_values, 25)
        q3_ppg_lf = np.percentile(ppg_hrv_lf_values, 75)
        ppg_hrv_lf_iqr = q3_ppg_lf - q1_ppg_lf
        if ppg_hrv_lf_iqr == 0:
            ppg_hrv_lf_iqr = 1.0
    else:
        ppg_hrv_lf_median = 0.0
        ppg_hrv_lf_iqr = 1.0
    
    return {
        'ecg_hrv': {
            'hrv_hf': {'median': ecg_hrv_hf_median, 'iqr': ecg_hrv_hf_iqr},
            'hrv_lf': {'median': ecg_hrv_lf_median, 'iqr': ecg_hrv_lf_iqr}
        },
        'ppg_hrv': {
            'hrv_hf': {'median': ppg_hrv_hf_median, 'iqr': ppg_hrv_hf_iqr},
            'hrv_lf': {'median': ppg_hrv_lf_median, 'iqr': ppg_hrv_lf_iqr}
        }
    }


def apply_normalization(bags, norm_coefficients):
    """
    Apply normalization to bags.
    
    - Instance Normalization for ECG and PPG segments: Each segment is normalized independently
      to have mean=0 and std=1. Formula: (x - mean) / std
    - Robust Scaling for HRV features: Uses pre-computed coefficients (median and IQR).
      Formula: (x - median) / IQR
    
    Args:
        bags: List of bag dictionaries
        norm_coefficients: Dictionary with normalization coefficients from calculate_normalization_coefficients()
                          Contains 'median' and 'iqr' keys for HRV features only
    
    Returns:
        list: List of normalized bag dictionaries (modified in-place)
    """
    for bag in bags:
        # Instance Normalization for ECG segments: normalize each segment independently
        ecg_segments = bag['ecg_segments']  # Shape: (N, L_ecg)
        if len(ecg_segments) > 0:
            ecg_segments = np.array(ecg_segments)
            # Compute mean and std along axis 1 (time dimension) for each segment
            # Keep dimensions for broadcasting: (N, 1)
            segment_means = np.mean(ecg_segments, axis=1, keepdims=True)
            segment_stds = np.std(ecg_segments, axis=1, keepdims=True)
            # Avoid division by zero
            segment_stds = np.where(segment_stds == 0, 1.0, segment_stds)
            # Normalize: (x - mean) / std
            ecg_segments = (ecg_segments - segment_means) / segment_stds
            bag['ecg_segments'] = ecg_segments
        
        # Instance Normalization for PPG segments: normalize each segment independently
        ppg_segments = bag['ppg_segments']  # Shape: (N, L_ppg)
        if len(ppg_segments) > 0:
            ppg_segments = np.array(ppg_segments)
            # Compute mean and std along axis 1 (time dimension) for each segment
            # Keep dimensions for broadcasting: (N, 1)
            segment_means = np.mean(ppg_segments, axis=1, keepdims=True)
            segment_stds = np.std(ppg_segments, axis=1, keepdims=True)
            # Avoid division by zero
            segment_stds = np.where(segment_stds == 0, 1.0, segment_stds)
            # Normalize: (x - mean) / std
            ppg_segments = (ppg_segments - segment_means) / segment_stds
            bag['ppg_segments'] = ppg_segments
        
        # Robust scaling for ECG HRV features
        if bag['ecg_hrv'] is not None:
            bag['ecg_hrv']['hrv_hf'] = (bag['ecg_hrv']['hrv_hf'] - norm_coefficients['ecg_hrv']['hrv_hf']['median']) / norm_coefficients['ecg_hrv']['hrv_hf']['iqr']
            bag['ecg_hrv']['hrv_lf'] = (bag['ecg_hrv']['hrv_lf'] - norm_coefficients['ecg_hrv']['hrv_lf']['median']) / norm_coefficients['ecg_hrv']['hrv_lf']['iqr']
        
        # Robust scaling for PPG HRV features
        if bag['ppg_hrv'] is not None:
            bag['ppg_hrv']['hrv_hf'] = (bag['ppg_hrv']['hrv_hf'] - norm_coefficients['ppg_hrv']['hrv_hf']['median']) / norm_coefficients['ppg_hrv']['hrv_hf']['iqr']
            bag['ppg_hrv']['hrv_lf'] = (bag['ppg_hrv']['hrv_lf'] - norm_coefficients['ppg_hrv']['hrv_lf']['median']) / norm_coefficients['ppg_hrv']['hrv_lf']['iqr']
    
    return bags

