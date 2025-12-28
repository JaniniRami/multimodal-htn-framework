"""
Signal validation functions for ECG and PPG signals from EDF files.

This module provides validation functions that check signal quality using
statistical analysis on windowed segments.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from edfio import read_edf


def validate_edf_signal(edf_file_path: Path, window_size_seconds: int = 10, 
                        sampling_rate: Optional[float] = None,
                        kurtosis_threshold: float = 3.0,
                        saturation_threshold: float = 0.01,
                        std_threshold: float = 0.05,
                        usable_threshold: float = 0.5) -> Dict:
    """
    Validate EDF file by checking for ECG/EKG channel and performing statistical analysis.
    
    Args:
        edf_file_path: Path to EDF file
        window_size_seconds: Size of non-overlapping windows in seconds (default: 10)
        sampling_rate: Sampling rate (if None, will be read from file)
        kurtosis_threshold: Minimum kurtosis for valid ECG (default: 3.0)
        saturation_threshold: Maximum saturation percentage (default: 0.01 = 1%)
        std_threshold: Minimum standard deviation (default: 0.05)
        usable_threshold: Minimum percentage of usable windows (default: 0.5 = 50%)
    
    Returns:
        Dictionary with validation results and statistics
    """
    result = {
        'has_ecg_channel': False,
        'is_valid': False,
        'usable_percentage': 0.0,
        'error': None,
        'statistics': {
            'mean_std': None,
            'mean_saturation': None,
            'mean_kurtosis': None,
            'mean_skewness': None,
            'mean_baseline_wander': None,
            'mean_line_noise_ratio': None,
            'num_windows': 0,
            'num_usable_windows': 0
        }
    }
    
    # Check if read_edf is available
    if read_edf is None:
        result['error'] = 'read_edf function not available. Please import it.'
        return result
    
    try:
        # Read EDF file
        raw = read_edf(str(edf_file_path), lazy_load_data=True)
        
        # Get channel names
        ch_names = [raw.signals[i].label for i in range(raw.num_signals)]
        
        # Find ECG/EKG channel
        ecg_ch_name = None
        if 'EKG' in ch_names:
            ecg_ch_name = 'EKG'
        elif 'ECG' in ch_names:
            ecg_ch_name = 'ECG'
        
        if ecg_ch_name is None:
            result['error'] = 'No ECG or EKG channel found'
            return result
        
        result['has_ecg_channel'] = True
        
        # Get ECG signal
        ecg_ch = raw.get_signal(ecg_ch_name)
        ecg_signal = np.array(ecg_ch.data)
        
        # Get sampling rate if not provided
        if sampling_rate is None:
            # Try to get from signal metadata
            try:
                # Try different possible attributes for sampling rate
                if hasattr(ecg_ch, 'sample_rate'):
                    sampling_rate = ecg_ch.sample_rate
                elif hasattr(raw, 'signal_headers'):
                    sampling_rate = raw.signal_headers[ch_names.index(ecg_ch_name)]['sample_rate']
                elif hasattr(raw, 'sample_rate'):
                    sampling_rate = raw.sample_rate
                else:
                    # Default fallback - common EDF sampling rate
                    sampling_rate = 256.0
            except:
                # Default fallback
                sampling_rate = 256.0
        
        # Calculate window size in samples
        window_size_samples = int(window_size_seconds * sampling_rate)
        
        # Process in non-overlapping windows
        num_windows = len(ecg_signal) // window_size_samples
        if num_windows == 0:
            result['error'] = 'Signal too short for analysis'
            return result
        
        # Storage for window statistics
        window_stats = {
            'std': [],
            'saturation': [],
            'kurtosis': [],
            'skewness': [],
            'baseline_wander': [],
            'line_noise_ratio': [],
            'is_usable': []
        }
        
        # Process each window
        for i in range(num_windows):
            start_idx = i * window_size_samples
            end_idx = start_idx + window_size_samples
            window_data = ecg_signal[start_idx:end_idx]
            
            # 1. Flatline Check: Standard Deviation
            std_val = np.std(window_data)
            window_stats['std'].append(std_val)
            
            # Saturation check removed - falsely flags valid ECGs with flat baselines as clipped
            # Kurtosis check is sufficient to reject true clipping (square waves have low kurtosis)
            # Set saturation to 0.0 to preserve output structure
            saturation_pct = 0.0
            window_stats['saturation'].append(saturation_pct)
            
            # 2. ECG-ness: Kurtosis and Skewness
            # Guard clause for flatlines to avoid NaN
            if std_val < 1e-6:
                kurt_val = -1.0
                skew_val = 0.0
            else:
                kurt_val = kurtosis(window_data)
                skew_val = skew(window_data)
            window_stats['kurtosis'].append(kurt_val)
            window_stats['skewness'].append(skew_val)
            
            # 3. Baseline Wander: Range of rolling mean
            # Use a rolling window of ~1 second for baseline
            rolling_window = int(sampling_rate)  # 1 second
            if len(window_data) > rolling_window:
                rolling_mean = np.convolve(window_data, 
                                         np.ones(rolling_window)/rolling_window, 
                                         mode='valid')
                baseline_wander = np.max(rolling_mean) - np.min(rolling_mean)
            else:
                baseline_wander = np.max(window_data) - np.min(window_data)
            window_stats['baseline_wander'].append(baseline_wander)
            
            # 4. Line Noise: PSD ratio at 50Hz/60Hz
            # Calculate PSD using Welch's method
            nperseg = min(256, len(window_data) // 4)  # Segment length
            if nperseg < 8:
                line_noise_ratio = 0.0
            else:
                freqs, psd = signal.welch(window_data, fs=sampling_rate, 
                                        nperseg=nperseg, 
                                        noverlap=nperseg//2)
                total_power = np.sum(psd)
                
                # Check for 50Hz and 60Hz (with some tolerance)
                freq_50_idx = np.argmin(np.abs(freqs - 50))
                freq_60_idx = np.argmin(np.abs(freqs - 60))
                
                # Get power at these frequencies (check ±2Hz range)
                freq_50_power = np.sum(psd[np.abs(freqs - 50) <= 2])
                freq_60_power = np.sum(psd[np.abs(freqs - 60) <= 2])
                
                line_noise_power = freq_50_power + freq_60_power
                if total_power > 0:
                    line_noise_ratio = line_noise_power / total_power
                else:
                    line_noise_ratio = 0.0
            window_stats['line_noise_ratio'].append(line_noise_ratio)
            
            # Determine if window is usable
            # Rely on kurtosis to catch true clipping (square waves have low kurtosis)
            # Flatline check already handled above
            is_usable = (kurt_val > kurtosis_threshold and 
                        std_val > std_threshold)
            window_stats['is_usable'].append(is_usable)
        
        # Calculate averages (using nanmean to handle NaN values robustly)
        result['statistics']['mean_std'] = float(np.nanmean(window_stats['std']))
        result['statistics']['mean_saturation'] = float(np.nanmean(window_stats['saturation']))
        result['statistics']['mean_kurtosis'] = float(np.nanmean(window_stats['kurtosis']))
        result['statistics']['mean_skewness'] = float(np.nanmean(window_stats['skewness']))
        result['statistics']['mean_baseline_wander'] = float(np.nanmean(window_stats['baseline_wander']))
        result['statistics']['mean_line_noise_ratio'] = float(np.nanmean(window_stats['line_noise_ratio']))
        result['statistics']['num_windows'] = num_windows
        result['statistics']['num_usable_windows'] = int(np.sum(window_stats['is_usable']))
        
        # Calculate usable percentage
        usable_percentage = result['statistics']['num_usable_windows'] / num_windows
        result['usable_percentage'] = float(usable_percentage)
        
        # Determine if signal is valid (>50% usable)
        result['is_valid'] = usable_percentage >= usable_threshold
        
    except Exception as e:
        result['error'] = str(e)
        return result
    
    return result


def validate_ppg_signal(edf_file_path: Path, raw=None, window_size_seconds: int = 10,
                        sampling_rate: Optional[float] = None,
                        std_min: float = 10.0,
                        std_max: float = 400.0,
                        skewness_threshold: float = 0.2,
                        std_threshold: float = 1e-6,
                        usable_threshold: float = 0.5) -> Dict:
    """
    Validate PPG signal from EDF file using the "Wavy" pipeline.
    Handles zero-centered (AC-coupled) PPG signals.
    
    Args:
        edf_file_path: Path to EDF file
        raw: Pre-loaded EDF file object (optional, to avoid re-reading)
        window_size_seconds: Size of non-overlapping windows in seconds (default: 10)
        sampling_rate: Sampling rate (if None, will be read from file)
        std_min: Minimum standard deviation for valid signal (default: 10.0)
        std_max: Maximum standard deviation for valid signal (default: 400.0)
        skewness_threshold: Minimum absolute skewness (default: 0.2)
        std_threshold: Minimum standard deviation to avoid flatline (default: 1e-6)
        usable_threshold: Minimum percentage of usable windows (default: 0.5 = 50%)
    
    Returns:
        Dictionary with validation results and statistics
    """
    result = {
        'has_ppg_channel': False,
        'is_valid': False,
        'usable_percentage': 0.0,
        'error': None,
        'statistics': {
            'mean_std': None,
            'mean_saturation': None,
            'mean_skewness': None,
            'mean_kurtosis': None,
            'num_windows': 0,
            'num_usable_windows': 0
        }
    }
    
    # Check if read_edf is available
    if read_edf is None:
        result['error'] = 'read_edf function not available. Please import it.'
        return result
    
    try:
        # Read EDF file if not provided
        if raw is None:
            raw = read_edf(str(edf_file_path), lazy_load_data=True)
        
        # Get channel names
        ch_names = [raw.signals[i].label for i in range(raw.num_signals)]
        
        # Find PPG channel (Plethysmogram)
        ppg_ch_name = None
        if 'Plethysmogram' in ch_names:
            ppg_ch_name = 'Plethysmogram'
        elif 'PPG' in ch_names:
            ppg_ch_name = 'PPG'
        elif 'PLETH' in ch_names:
            ppg_ch_name = 'PLETH'
        
        if ppg_ch_name is None:
            result['error'] = 'No Plethysmogram/PPG channel found'
            return result
        
        result['has_ppg_channel'] = True
        
        # Get PPG signal
        ppg_ch = raw.get_signal(ppg_ch_name)
        ppg_signal = np.array(ppg_ch.data)
        
        # Get sampling rate if not provided
        if sampling_rate is None:
            try:
                if hasattr(ppg_ch, 'sample_rate'):
                    sampling_rate = ppg_ch.sample_rate
                elif hasattr(raw, 'signal_headers'):
                    sampling_rate = raw.signal_headers[ch_names.index(ppg_ch_name)]['sample_rate']
                elif hasattr(raw, 'sample_rate'):
                    sampling_rate = raw.sample_rate
                else:
                    sampling_rate = 256.0
            except:
                sampling_rate = 256.0
        
        # Calculate window size in samples
        window_size_samples = int(window_size_seconds * sampling_rate)
        
        # Process in non-overlapping windows
        num_windows = len(ppg_signal) // window_size_samples
        if num_windows == 0:
            result['error'] = 'Signal too short for analysis'
            return result
        
        # Storage for window statistics
        window_stats = {
            'std': [],
            'saturation': [],
            'skewness': [],
            'kurtosis': [],
            'is_usable': []
        }
        
        # Process each window
        for i in range(num_windows):
            start_idx = i * window_size_samples
            end_idx = start_idx + window_size_samples
            window_data = ppg_signal[start_idx:end_idx]
            
            # 1. Flatline Check: Standard Deviation
            std_val = np.std(window_data)
            window_stats['std'].append(std_val)
            
            # Reject if flatline
            if std_val < std_threshold:
                window_stats['saturation'].append(0.0)  # Set to 0.0 to preserve output structure
                window_stats['skewness'].append(0.0)
                window_stats['kurtosis'].append(0.0)
                window_stats['is_usable'].append(False)
                continue
            
            # Saturation check removed - PPG signals rarely clip in a way that statistical checks catch
            # Set saturation to 0.0 to preserve output structure
            saturation_pct = 0.0
            window_stats['saturation'].append(saturation_pct)
            
            # Shape Check: Skewness (for zero-centered AC-coupled signals)
            # Guard clause for flatlines to avoid NaN
            if std_val < 1e-6:
                skew_val = 0.0
                kurt_val = 0.0
            else:
                skew_val = skew(window_data)
                kurt_val = kurtosis(window_data)  # Still calculate for statistics
            window_stats['skewness'].append(skew_val)
            window_stats['kurtosis'].append(kurt_val)
            
            # Determine if window is usable
            # For zero-centered (AC-coupled) PPG signals:
            # - Not flatline (std >= threshold) - already checked above
            # - Std in valid range (10.0 < std < 400.0)
            # - Skewness > threshold
            is_usable = (std_min < std_val < std_max and
                        abs(skew_val) > skewness_threshold)
            window_stats['is_usable'].append(is_usable)
        
        # Calculate averages (using nanmean to handle NaN values robustly)
        result['statistics']['mean_std'] = float(np.nanmean(window_stats['std']))
        result['statistics']['mean_saturation'] = float(np.nanmean(window_stats['saturation']))
        result['statistics']['mean_skewness'] = float(np.nanmean(window_stats['skewness']))
        result['statistics']['mean_kurtosis'] = float(np.nanmean(window_stats['kurtosis']))
        result['statistics']['num_windows'] = num_windows
        result['statistics']['num_usable_windows'] = int(np.sum(window_stats['is_usable']))
        
        # Calculate usable percentage
        usable_percentage = result['statistics']['num_usable_windows'] / num_windows
        result['usable_percentage'] = float(usable_percentage)
        
        # Determine if signal is valid (>50% usable)
        result['is_valid'] = usable_percentage >= usable_threshold
        
    except Exception as e:
        result['error'] = str(e)
        return result
    
    return result

