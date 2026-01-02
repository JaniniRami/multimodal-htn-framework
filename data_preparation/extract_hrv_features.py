"""
HRV feature extraction module.

This module provides functions to:
- Merge overlapping segments back into continuous signals
- Extract HRV features from merged signals
"""
import time
import numpy as np
import neurokit2 as nk
from typing import Tuple, Dict


def merge_segments_with_overlap_removal(segments: list, segment_duration: int, signal_fs: int) -> np.ndarray:
    """
    Merge overlapping segments back into a continuous signal by removing the overlap.
    
    The segments were created with 30% overlap. This function:
    1. Takes the first segment fully
    2. For subsequent segments, removes the overlapping portion (first 30%) 
       and concatenates only the non-overlapping part (last 70%)
    
    Args:
        segments: List of numpy arrays, each representing a 30-second segment
        segment_duration: Duration of each segment in seconds (default: 30)
        signal_fs: Sampling frequency in Hz (200 for ECG, 100 for PPG)
    
    Returns:
        Merged signal as a numpy array (approximately 30 minutes long for 60 segments)
    """
    if len(segments) == 0:
        return np.array([])
    
    # Calculate segment size and overlap
    segment_size = segment_duration * signal_fs
    overlap_size = int(segment_size * 0.3)  # 30% overlap
    non_overlap_size = segment_size - overlap_size  # 70% non-overlapping part
    
    # Start with the first segment (take it fully)
    merged_signal = segments[0].copy()
    
    # For subsequent segments, take only the non-overlapping part (skip first 30%)
    for i in range(1, len(segments)):
        segment = segments[i]
        
        # Ensure segment has expected size
        if len(segment) != segment_size:
            # If last segment is shorter, take what we can
            if i == len(segments) - 1:
                # For the last segment, take everything after the overlap
                if len(segment) > overlap_size:
                    merged_signal = np.concatenate([merged_signal, segment[overlap_size:]])
                # If it's too short, just append what's left
                elif len(segment) > 0:
                    merged_signal = np.concatenate([merged_signal, segment])
            else:
                raise ValueError(f"Segment {i} has unexpected size: {len(segment)} (expected {segment_size})")
        else:
            # Take only the non-overlapping part (last 70% of the segment)
            non_overlap_part = segment[overlap_size:]
            merged_signal = np.concatenate([merged_signal, non_overlap_part])
    
    return merged_signal


def merge_ecg_ppg_segments(ecg_segments: list, ppg_segments: list, 
                          ecg_fs: float = 200.0, ppg_fs: float = 100.0,
                          segment_duration: int = 30) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Merge ECG and PPG segments back into continuous 30-minute signals and extract HRV features.
    
    Args:
        ecg_segments: List of ECG segments (each 30 seconds at 200 Hz)
        ppg_segments: List of PPG segments (each 30 seconds at 100 Hz)
        ecg_fs: ECG sampling frequency in Hz (default: 200.0)
        ppg_fs: PPG sampling frequency in Hz (default: 100.0)
        segment_duration: Duration of each segment in seconds (default: 30)
    
    Returns:
        Tuple of (merged_ecg_signal, merged_ppg_signal, ecg_hrv_features, ppg_hrv_features)
        where hrv_features are dictionaries with 'hrv_hf' and 'hrv_lf' keys
    """
    # Merge ECG segments
    merged_ecg = merge_segments_with_overlap_removal(ecg_segments, segment_duration, int(ecg_fs))
    
    # Merge PPG segments
    merged_ppg = merge_segments_with_overlap_removal(ppg_segments, segment_duration, int(ppg_fs))

    start_time = time.time()
    ecg_peaks, _ = nk.ecg_peaks(merged_ecg, sampling_rate=200.0)
    end_time = time.time()
    print(f"ECG peaks extraction time: {end_time - start_time} seconds")

    start_time = time.time()
    ecg_hrv = nk.hrv_frequency(ecg_peaks, sampling_rate=200.0, show=False, psd_method="welch")
    end_time = time.time()
    print(f"ECG HRV extraction time: {end_time - start_time} seconds")
    
    ecg_hrv_features = {
        "hrv_hf": float(ecg_hrv.get("HRV_HF", [0])[0]),
        "hrv_lf": float(ecg_hrv.get("HRV_LF", [0])[0]),
    }
    start_time = time.time()
    ppg_peaks, info = nk.ppg_peaks(merged_ppg, sampling_rate=100,  method="elgendi", show=False)
    end_time = time.time()
    print(f"PPG peaks extraction time: {end_time - start_time} seconds")

    start_time = time.time()
    ppg_hrv = nk.hrv_frequency(ppg_peaks, sampling_rate=100, show=False)
    end_time = time.time()
    print(f"PPG HRV extraction time: {end_time - start_time} seconds")

    ppg_hrv_features = {
        "hrv_hf": float(ppg_hrv.get("HRV_HF", [0])[0]),
        "hrv_lf": float(ppg_hrv.get("HRV_LF", [0])[0]),
    }

    return merged_ecg, merged_ppg, ecg_hrv_features, ppg_hrv_features

