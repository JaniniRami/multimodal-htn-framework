"""
ECG and PPG signal cleaning pipeline.

This module processes EDF files from file_mapping_apnea_only.json,
cleans the signals, and saves them to data/Embla_Cleaned.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal
import pywt
import neurokit2 as nk
from edfio import read_edf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import h5py
from segment_signal import get_segments
from extract_hrv_features import merge_ecg_ppg_segments


APNEA_EVENTS_NAMES = ["APNEA", "APNEA-CENTRAL", "APNEA-MIXED", "APNEA-OBSTRUCTIVE", "HYPOPNEA"]

sts_guide = {
    'SLEEP-S0': 0,
    'SLEEP-S1': 1,
    'SLEEP-S2': 2,
    'SLEEP-S3': 3,
    'SLEEP-S4': 4,
    'SLEEP-REM': 5
}


def get_edf_header(edf_file):
    with open(edf_file, "rb") as binary_file:
        header = binary_file.read()[:256].decode("latin-1")
        header_dict = {
            "version": header[:8].strip(),
            "patient_id": header[8:88].strip(),
            "recording_id": header[88:168].strip(),
            "start_date": header[168:176].strip(),
            "start_time": header[176:184].strip(),
            "header_bytes": header[184:192].strip(),
            "num_records": header[236:244].strip(),
            "duration": header[244:252].strip(),
            "num_signals": int(header[252:256].strip()),
        }

    return header_dict

def load_signals(edf_file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], Optional[str]]:
    """
    Load ECG and PPG signals from EDF file.
    
    Args:
        edf_file_path: Path to the EDF file
    
    Returns:
        tuple: (ecg_signal, ppg_signal, ecg_sampling_rate, ppg_sampling_rate, error_message)
               - ecg_signal: numpy array of ECG data, or None if error
               - ppg_signal: numpy array of PPG data, or None if error
               - ecg_sampling_rate: ECG sampling rate in Hz (fixed at 200.0), or None if error
               - ppg_sampling_rate: PPG sampling rate in Hz (fixed at 100.0), or None if error
               - error_message: error message string, or None if successful
    """
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
            return None, None, None, None, 'No ECG or EKG channel found'
        
        # Find PPG channel (Plethysmogram)
        ppg_ch_name = None
        if 'Plethysmogram' in ch_names:
            ppg_ch_name = 'Plethysmogram'
        
        if ppg_ch_name is None:
            return None, None, None, None, 'No Plethysmogram channel found'
        
        # Get ECG signal
        ecg_ch = raw.get_signal(ecg_ch_name)
        ecg_signal = np.array(ecg_ch.data)
        
        # Get PPG signal
        ppg_ch = raw.get_signal(ppg_ch_name)
        ppg_signal = np.array(ppg_ch.data)
        
        # Use fixed sampling rates
        ecg_sampling_rate = 200.0  # Fixed ECG sampling rate
        ppg_sampling_rate = 100.0  # Fixed PPG sampling rate
        
        return ecg_signal, ppg_signal, ecg_sampling_rate, ppg_sampling_rate, None
        
    except Exception as e:
        return None, None, None, None, f"Error loading signals: {str(e)}"


def wavelet_denoise(x: np.ndarray,
                    wavelet: str = 'db4',
                    level: int = 4,
                    method: str = 'soft') -> np.ndarray:
    """
    Level-dependent wavelet thresholding (VisuShrink).
    
    Args:
        x: Input signal array
        wavelet: Wavelet type (default: 'db6')
        level: Decomposition level (None = maximum)
        method: Thresholding method ('soft' or 'hard', default: 'soft')
    
    Returns:
        Denoised signal array
    """
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    #  Use a fixed multiplier (e.g., 3.0 or 3.5) instead of VisuShrink 
    # to prevents over-smoothing on long signals. 
    effective_len = min(len(x), 2000) 
    uthresh = sigma * np.sqrt(2 * np.log(effective_len))
    
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode=method) for c in coeffs[1:]]
    
    return pywt.waverec(coeffs, wavelet)


def clean_ecg_signal(ecg_signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Clean ECG signal.
    
    Steps:
    1. Band-pass filter: Zero-phase 4th-order Butterworth band-pass, 0.5–45 Hz
    2. Notch filter: 50 Hz powerline notch filter
    3. Wavelet denoising: Level-dependent wavelet thresholding (VisuShrink)
    
    Args:
        ecg_signal: Raw ECG signal array
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Cleaned ECG signal array
    """
    # Step 1: Band-pass filter (0.5-45 Hz) to remove baseline wander and high-frequency noise
    # Zero-phase 4th-order Butterworth band-pass filter
    lowcut = 0.5  # Hz
    highcut = 45.0  # Hz
    
    # Check Nyquist frequency
    nyquist = sampling_rate / 2.0
    if highcut >= nyquist:
        # If highcut is too close to Nyquist, adjust it
        highcut = nyquist * 0.95
    
    # Design Butterworth band-pass filter
    # For 4th-order, use order=2 (filtfilt applies filter twice, so 2*2=4)
    order = 2
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)
    
    # Apply zero-phase filtering (forward and backward)
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    
    # Step 2: Notch filter at 50 Hz to remove powerline interference
    # Design notch filter (band-stop filter with narrow stopband)
    notch_freq = 50.0  # Hz
    quality_factor = 30.0  # Quality factor determines the width of the notch (higher = narrower)
    
    # Check if notch frequency is within valid range
    if notch_freq < nyquist:
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
        # Apply zero-phase notch filtering
        filtered_signal = signal.filtfilt(b_notch, a_notch, filtered_signal)
    
    # Step 3: Wavelet denoising using VisuShrink thresholding
    filtered_signal = wavelet_denoise(filtered_signal, wavelet='db6', level=None, method='soft')
    
    return filtered_signal


def clean_ppg_signal(ppg_signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """
    Clean PPG signal.
    
    Steps:
    1. Band-pass filter: Zero-phase 4th-order Butterworth band-pass, 0.5–18 Hz
       Note: 18 Hz cutoff handles powerline noise, so no notch filter is needed.
    
    Args:
        ppg_signal: Raw PPG signal array
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Cleaned PPG signal array
    """
    # Band-pass filter (0.5-18 Hz) to remove baseline wander and high-frequency noise
    # Zero-phase 4th-order Butterworth band-pass filter
    lowcut = 0.5  # Hz
    highcut = 18.0  # Hz
    
    # Check Nyquist frequency
    nyquist = sampling_rate / 2.0
    if highcut >= nyquist:
        # If highcut is too close to Nyquist, adjust it
        highcut = nyquist * 0.95
    
    # Design Butterworth band-pass filter
    # For 4th-order, use order=2 (filtfilt applies filter twice, so 2*2=4)
    order = 2
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)
    
    # Apply zero-phase filtering (forward and backward)
    filtered_signal = signal.filtfilt(b, a, ppg_signal)
    
    return filtered_signal


def plot_signals_before_after(raw_ecg: np.ndarray, cleaned_ecg: np.ndarray,
                              raw_ppg: np.ndarray, cleaned_ppg: np.ndarray,
                              ecg_sampling_rate: float, ppg_sampling_rate: float,
                              output_path: Path, segment_start_time: float = 0.0,
                              sample_duration_seconds: int = 30):
    """
    Plot raw and cleaned ECG and PPG signals for a 30-second sample.
    
    Creates a 4-row subplot with:
    - Row 1: Raw ECG
    - Row 2: Cleaned ECG
    - Row 3: Raw PPG
    - Row 4: Cleaned PPG
    
    Args:
        raw_ecg: Raw ECG signal array
        cleaned_ecg: Cleaned ECG signal array
        raw_ppg: Raw PPG signal array
        cleaned_ppg: Cleaned PPG signal array
        ecg_sampling_rate: ECG sampling rate in Hz
        ppg_sampling_rate: PPG sampling rate in Hz
        output_path: Path to save the plot
        segment_start_time: Start time of the segment in seconds (for random selection within segment)
        sample_duration_seconds: Duration of sample to plot in seconds (default: 30)
    """
    # Determine available length for each signal
    ecg_min_length = min(len(raw_ecg), len(cleaned_ecg))
    ppg_min_length = min(len(raw_ppg), len(cleaned_ppg))
    
    # Calculate total duration and segment boundaries
    ecg_total_duration = ecg_min_length / ecg_sampling_rate
    ppg_total_duration = ppg_min_length / ppg_sampling_rate
    total_duration = min(ecg_total_duration, ppg_total_duration)
    
    # Calculate sample length in samples
    ecg_sample_length = int(sample_duration_seconds * ecg_sampling_rate)
    ppg_sample_length = int(sample_duration_seconds * ppg_sampling_rate)
    
    # Select random start time within the segment (30-minute window)
    segment_duration = 1800.0  # 30 minutes in seconds
    segment_end_time = min(segment_start_time + segment_duration, total_duration)
    available_duration = segment_end_time - segment_start_time
    
    if available_duration < sample_duration_seconds:
        actual_duration = available_duration
        if actual_duration < 1.0:  # Skip if less than 1 second available
            print(f"Warning: Not enough data in segment. Skipping plot.")
            return
        print(f"Warning: Segment shorter than {sample_duration_seconds}s. Using {actual_duration:.1f}s sample.")
        sample_start_time = segment_start_time  # Use start of segment if too short
    else:
        actual_duration = sample_duration_seconds
        # Random start within the segment (leave room for the sample)
        max_start_in_segment = available_duration - actual_duration
        random_offset = np.random.uniform(0, max_start_in_segment)
        sample_start_time = segment_start_time + random_offset
    
    # Calculate indices
    ecg_start_idx = int(sample_start_time * ecg_sampling_rate)
    ppg_start_idx = int(sample_start_time * ppg_sampling_rate)
    
    ecg_end_idx = ecg_start_idx + int(actual_duration * ecg_sampling_rate)
    ppg_end_idx = ppg_start_idx + int(actual_duration * ppg_sampling_rate)
    
    # Ensure indices don't exceed signal length
    ecg_start_idx = min(ecg_start_idx, ecg_min_length)
    ppg_start_idx = min(ppg_start_idx, ppg_min_length)
    ecg_end_idx = min(ecg_end_idx, ecg_min_length)
    ppg_end_idx = min(ppg_end_idx, ppg_min_length)
    
    # Extract samples
    raw_ecg_sample = raw_ecg[ecg_start_idx:ecg_end_idx]
    cleaned_ecg_sample = cleaned_ecg[ecg_start_idx:ecg_end_idx]
    raw_ppg_sample = raw_ppg[ppg_start_idx:ppg_end_idx]
    cleaned_ppg_sample = cleaned_ppg[ppg_start_idx:ppg_end_idx]
    
    # Create time axis in seconds for each signal (relative to segment start)
    ecg_time_axis = np.arange(len(raw_ecg_sample)) / ecg_sampling_rate
    ppg_time_axis = np.arange(len(raw_ppg_sample)) / ppg_sampling_rate
    
    # Create figure with 4 subplots stacked vertically
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Signal Cleaning Comparison - {actual_duration:.1f}s Sample at {sample_start_time/60:.1f} min', 
                 fontsize=16, fontweight='bold')
    
    # Plot raw ECG
    axes[0].plot(ecg_time_axis, raw_ecg_sample, 'b-', linewidth=1.5)
    axes[0].set_title('Raw ECG Signal', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, ecg_time_axis[-1] if len(ecg_time_axis) > 0 else actual_duration])
    
    # Plot cleaned ECG
    axes[1].plot(ecg_time_axis, cleaned_ecg_sample, 'g-', linewidth=1.5)
    axes[1].set_title('Cleaned ECG Signal', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, ecg_time_axis[-1] if len(ecg_time_axis) > 0 else actual_duration])
    
    # Plot raw PPG
    axes[2].plot(ppg_time_axis, raw_ppg_sample, 'r-', linewidth=1.5)
    axes[2].set_title('Raw PPG Signal', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, ppg_time_axis[-1] if len(ppg_time_axis) > 0 else actual_duration])
    
    # Plot cleaned PPG
    axes[3].plot(ppg_time_axis, cleaned_ppg_sample, 'm-', linewidth=1.5)
    axes[3].set_title('Cleaned PPG Signal', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Time (seconds)', fontsize=12)
    axes[3].set_ylabel('Amplitude', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([0, ppg_time_axis[-1] if len(ppg_time_axis) > 0 else actual_duration])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_segments(raw_ecg: np.ndarray, cleaned_ecg: np.ndarray,
                      raw_ppg: np.ndarray, cleaned_ppg: np.ndarray,
                      ecg_sampling_rate: float, ppg_sampling_rate: float,
                      plot_output_dir: Path, folder_name: str,
                      segment_duration_minutes: int = 30,
                      sample_duration_seconds: int = 30) -> list:
    """
    Plot random 30-second samples from each 30-minute segment of the signals.
    
    Args:
        raw_ecg: Raw ECG signal array
        cleaned_ecg: Cleaned ECG signal array
        raw_ppg: Raw PPG signal array
        cleaned_ppg: Cleaned PPG signal array
        ecg_sampling_rate: ECG sampling rate in Hz
        ppg_sampling_rate: PPG sampling rate in Hz
        plot_output_dir: Directory to save plots
        folder_name: Name for the plot files (patient identifier)
        segment_duration_minutes: Duration of each segment in minutes (default: 30)
        sample_duration_seconds: Duration of sample to plot in seconds (default: 30)
    
    Returns:
        List of plot file paths
    """
    plot_paths = []
    
    # Determine total duration
    ecg_min_length = min(len(raw_ecg), len(cleaned_ecg))
    ppg_min_length = min(len(raw_ppg), len(cleaned_ppg))
    ecg_total_duration = ecg_min_length / ecg_sampling_rate
    ppg_total_duration = ppg_min_length / ppg_sampling_rate
    total_duration = min(ecg_total_duration, ppg_total_duration)
    
    # Calculate segment duration in seconds
    segment_duration_seconds = segment_duration_minutes * 60.0
    
    # Calculate number of segments
    num_segments = int(total_duration / segment_duration_seconds)
    
    if num_segments == 0:
        # Signal is shorter than one segment, create one plot
        plot_file = plot_output_dir / f"{folder_name}_segment_0.png"
        plot_signals_before_after(raw_ecg, cleaned_ecg, raw_ppg, cleaned_ppg,
                                 ecg_sampling_rate, ppg_sampling_rate,
                                 plot_file, segment_start_time=0.0,
                                 sample_duration_seconds=sample_duration_seconds)
        plot_paths.append(str(plot_file))
    else:
        # Create a plot for each segment
        for segment_idx in range(num_segments):
            segment_start_time = segment_idx * segment_duration_seconds
            plot_file = plot_output_dir / f"{folder_name}_segment_{segment_idx:02d}.png"
            plot_signals_before_after(raw_ecg, cleaned_ecg, raw_ppg, cleaned_ppg,
                                     ecg_sampling_rate, ppg_sampling_rate,
                                     plot_file, segment_start_time=segment_start_time,
                                     sample_duration_seconds=sample_duration_seconds)
            plot_paths.append(str(plot_file))
            print(f"  Plot {segment_idx + 1}/{num_segments} saved: {plot_file.name}")
        
        # Handle remaining time (last partial segment if any)
        remaining_time = total_duration - (num_segments * segment_duration_seconds)
        if remaining_time >= sample_duration_seconds:
            segment_start_time = num_segments * segment_duration_seconds
            plot_file = plot_output_dir / f"{folder_name}_segment_{num_segments:02d}.png"
            plot_signals_before_after(raw_ecg, cleaned_ecg, raw_ppg, cleaned_ppg,
                                     ecg_sampling_rate, ppg_sampling_rate,
                                     plot_file, segment_start_time=segment_start_time,
                                     sample_duration_seconds=sample_duration_seconds)
            plot_paths.append(str(plot_file))
            print(f"  Plot {num_segments + 1} (final segment) saved: {plot_file.name}")
    
    return plot_paths


def plot_segment_before_after(raw_ecg_segment: np.ndarray, cleaned_ecg_segment: np.ndarray,
                              raw_ppg_segment: np.ndarray, cleaned_ppg_segment: np.ndarray,
                              ecg_sampling_rate: float, ppg_sampling_rate: float,
                              output_path: Path, patient_id: str, segment_idx: int,
                              apnea_label: int, sleep_stage: str):
    """
    Plot raw and cleaned ECG and PPG signals for a 30-second segment in 4 rows.
    
    Creates a 4-row subplot with:
    - Row 1: Raw ECG
    - Row 2: Cleaned ECG
    - Row 3: Raw PPG
    - Row 4: Cleaned PPG
    
    Args:
        raw_ecg_segment: Raw ECG segment array (30 seconds)
        cleaned_ecg_segment: Cleaned ECG segment array (30 seconds)
        raw_ppg_segment: Raw PPG segment array (30 seconds)
        cleaned_ppg_segment: Cleaned PPG segment array (30 seconds)
        ecg_sampling_rate: ECG sampling rate in Hz
        ppg_sampling_rate: PPG sampling rate in Hz
        output_path: Path to save the plot
        patient_id: Patient identifier
        segment_idx: Segment index
        apnea_label: Apnea label (0 or 1)
        sleep_stage: Sleep stage string
    """
    # Create time axis in seconds for each signal
    ecg_time_axis = np.arange(len(raw_ecg_segment)) / ecg_sampling_rate
    ppg_time_axis = np.arange(len(raw_ppg_segment)) / ppg_sampling_rate
    
    # Create figure with 4 rows (4 rows, 1 column) - wide enough for 30 seconds
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Patient: {patient_id} | Segment: {segment_idx} | Apnea: {apnea_label} | Sleep Stage: {sleep_stage}', 
                 fontsize=16, fontweight='bold')
    
    # Plot raw ECG
    axes[0].plot(ecg_time_axis, raw_ecg_segment, 'b-', linewidth=1.5)
    axes[0].set_title('Raw ECG Signal', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 30])
    
    # Plot cleaned ECG
    axes[1].plot(ecg_time_axis, cleaned_ecg_segment, 'g-', linewidth=1.5)
    axes[1].set_title('Cleaned ECG Signal', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 30])
    
    # Plot raw PPG
    axes[2].plot(ppg_time_axis, raw_ppg_segment, 'r-', linewidth=1.5)
    axes[2].set_title('Raw PPG Signal', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 30])
    
    # Plot cleaned PPG
    axes[3].plot(ppg_time_axis, cleaned_ppg_segment, 'm-', linewidth=1.5)
    axes[3].set_title('Cleaned PPG Signal', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Time (seconds)', fontsize=12)
    axes[3].set_ylabel('Amplitude', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([0, 30])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def zero_percentage(signal):
    zero_count = np.sum(signal == 0)
    total_count = len(signal)
    zero_percentage = (zero_count / total_count) * 100
    return zero_percentage

def detect_low_quality_ppg(ppg, fs,
                           min_ptp=0.02,
                           min_std=0.005,
                           min_slope_energy=1e-4):
    """
    Returns True if signal is low-quality (flat or unusable)
    """

    # Remove DC
    ppg = ppg - np.mean(ppg)

    # 1. Peak-to-peak amplitude
    ptp = np.ptp(ppg)
    if ptp < min_ptp:
        return True

    # 2. Global variability
    if np.std(ppg) < min_std:
        return True

    # 3. First-derivative (slope) energy
    dppg = np.diff(ppg)
    slope_energy = np.mean(dppg ** 2)

    if slope_energy < min_slope_energy:
        return True

    return False


def process_file(entry: Dict, output_base_dir: Path, save_plot: bool = False, plot_output_dir: Optional[Path] = None) -> Dict:
    """
    Process a single file entry from the mapping.
    
    Args:
        entry: Dictionary entry from file_mapping_apnea_only.json
        output_base_dir: Base directory (currently unused, kept for compatibility)
        save_plot: Whether to save a plot of before/after signals
        plot_output_dir: Directory to save plots (if save_plot is True)
    
    Returns:
        Dictionary with processing results
    """
    edf_path = Path(entry['edf_path'])
    
    result = {
        'edf_path': str(edf_path),
        'patient_id': entry.get('patient_id'),
        'follow_up': entry.get('follow_up'),
        'success': False,
        'error': None
    }
    
    # Check if EDF file exists
    if not edf_path.exists():
        result['error'] = f"EDF file not found: {edf_path}"
        return result
    
    # Determine patient_followup_id early to check for existing bags before loading EDF
    patient_id = entry.get('patient_id', 'unknown')
    follow_up = entry.get('follow_up')
    if follow_up is None:
        patient_followup_id = patient_id
    else:
        patient_followup_id = f"{patient_id}_{follow_up}"
    
    # Check if bag_0 exists - if it does, we'll still process but skip existing bags
    # This allows us to resume processing if interrupted
    patient_dir = output_base_dir / patient_followup_id
    bag_0_file = patient_dir / f"{patient_followup_id}_bag_0.h5"
    if bag_0_file.exists():
        print(f"  Note: Bag 0 already exists for {patient_followup_id}, will skip existing bags during processing")
    
    # Load ECG and PPG signals
    ecg_signal, ppg_signal, ecg_sampling_rate, ppg_sampling_rate, error = load_signals(edf_path)
    if error:
        result['error'] = error
        return result
    
    # Check if FLIP is enabled (FLIP == 1 means "yes", invert ECG signal)
    flip_enabled = entry.get('FLIP', 0) == 1
    if flip_enabled:
        # Invert ECG signal (multiply by -1)
        ecg_signal = -ecg_signal
    
    # Store raw signals for plotting if needed (after potential inversion)
    raw_ecg = ecg_signal.copy()
    raw_ppg = ppg_signal.copy()
    
    # Get EDF header for segmentation
    header = get_edf_header(edf_path)
    
    # Get annotation file path from entry
    txt_path = entry.get('txt_path')
    if txt_path is None:
        result['error'] = "No txt_path found in entry"
        return result
    
    txt_path = Path(txt_path)
    if not txt_path.exists():
        result['error'] = f"Annotation file not found: {txt_path}"
        return result
    
    ecg_report, ecg_signal_segments, ecg_labels, ecg_sleep_stages, ecg_n_apnea, ecg_n_non_apnea = get_segments(
        str(txt_path), raw_ecg, header, APNEA_EVENTS_NAMES, segment_duration=30, signal_fs=200
    )
    
    ppg_report, ppg_signal_segments, ppg_labels, ppg_sleep_stages, ppg_n_apnea, ppg_n_non_apnea = get_segments(
        str(txt_path), raw_ppg, header, APNEA_EVENTS_NAMES, segment_duration=30, signal_fs=100
    )

    # Get patient_id for error messages (already determined above)
    assert ecg_n_apnea == ppg_n_apnea, f"Apnea count mismatch for {patient_id}: ECG {ecg_n_apnea}, PPG {ppg_n_apnea}"
    assert ecg_labels == ppg_labels, f"Labels mismatch for {patient_id}"
    assert ecg_n_non_apnea == ppg_n_non_apnea, f"Non-apnea count mismatch for {patient_id}: ECG {ecg_n_non_apnea}, PPG {ppg_n_non_apnea}"
    assert len(ecg_signal_segments) == len(ppg_signal_segments), "Signal segments length mismatch"

    n_apnea = ecg_n_apnea
    n_non_apnea = ecg_n_non_apnea
    print(f"Apnea count: {n_apnea}, Non-apnea count: {n_non_apnea}")

    assert n_apnea > 0, "No apnea events found"
        
    # Store segment counts in result for tracking
    result['n_apnea'] = n_apnea
    result['n_non_apnea'] = n_non_apnea
    result['total_segments'] = n_apnea + n_non_apnea

    apnea_labels = ecg_labels 
    sleep_stages = ecg_sleep_stages
    
    # For plots (if needed)
    if follow_up is None:
        folder_name = patient_id
    else:
        folder_name = f"{patient_id} ({follow_up})"
    
    # Initialize lists to accumulate segments that pass quality checks
    valid_ecg_segments = []
    valid_ppg_segments = []
    valid_apnea_labels = []
    valid_sleep_stages = []
    bag_idx = 0
    
    # Loop through each segment and clean individually
    for i in range(len(ecg_signal_segments)):
        ecg_segment = ecg_signal_segments[i]
        ppg_segment = ppg_signal_segments[i]
        apnea_label = apnea_labels[i]
        sleep_stage = sleep_stages[i]
        
        if zero_percentage(ecg_segment) > 25 or zero_percentage(ppg_segment) > 25:
            continue

        # Clean each segment individually
        cleaned_ecg_segment = clean_ecg_signal(ecg_segment, ecg_sampling_rate)
        cleaned_ppg_segment = clean_ppg_signal(ppg_segment, ppg_sampling_rate)

        if detect_low_quality_ppg(cleaned_ppg_segment, ppg_sampling_rate):
            print(f"Skipping segment {i} for {patient_id} due to low quality PPG signal")
            continue


        try:
            _, ecg_info = nk.ecg_peaks(cleaned_ecg_segment, sampling_rate=ecg_sampling_rate, correct_artifacts=False, show=False)
        except IndexError:
                print(f"Skipping segment {i} for {patient_id} due to IndexError in ECG peaks detection")
                continue
        ecg_peaks = ecg_info['ECG_R_Peaks']
        if len(ecg_peaks) <= 20 or len(ecg_peaks) >= 80:
            print(f"Skipping segment {i} for {patient_id} due to ECG peaks count out of range, len(ecg_peaks): {len(ecg_peaks)}")
            continue

        try:
            _, ppg_info = nk.ppg_peaks(cleaned_ppg_segment, sampling_rate=ppg_sampling_rate, method="elgendi", show=False)
        except IndexError:
            print(f"Skipping segment {i} for {patient_id} due to IndexError in PPG peaks detection")
            continue
        ppg_peaks = ppg_info['PPG_Peaks']
        if len(ppg_peaks) <= 20 or len(ppg_peaks) >= 80:
            print(f"Skipping segment {i} for {patient_id} due to PPG peaks count out of range, len(ppg_peaks): {len(ppg_peaks)}")
            continue

        # All checks passed - add to valid segments list
        valid_ecg_segments.append(cleaned_ecg_segment)
        valid_ppg_segments.append(cleaned_ppg_segment)
        valid_apnea_labels.append(apnea_label)
        valid_sleep_stages.append(sleep_stage)
        
        # Plot the segment if enabled
        if save_plot and plot_output_dir is not None:
            plot_file = plot_output_dir / folder_name / f"segment_{i:04d}_apnea_{apnea_label}_stage_{sleep_stage}.png"
            plot_segment_before_after(
                ecg_segment, cleaned_ecg_segment,
                ppg_segment, cleaned_ppg_segment,
                ecg_sampling_rate, ppg_sampling_rate,
                plot_file, patient_id, i, apnea_label, sleep_stage
            )
        
        # When we have 60 segments, save them as a batch
        if len(valid_ecg_segments) == 60:
            # Create output directory structure: Embla_Cleaned/patient_id_followup_id/
            patient_dir = output_base_dir / patient_followup_id
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as h5py file directly in patient folder
            h5_file = patient_dir / f"{patient_followup_id}_bag_{bag_idx}.h5"
            
            # Check if file already exists, skip if it does
            if h5_file.exists():
                print(f"  Skipping bag {bag_idx} - file already exists: {h5_file}")
                # Reset lists for next batch
                valid_ecg_segments = []
                valid_ppg_segments = []
                valid_apnea_labels = []
                valid_sleep_stages = []
                bag_idx += 1
                continue
            
            # Merge segments back into continuous 30-minute signals and extract HRV features
            _, _, ecg_hrv_features, ppg_hrv_features = merge_ecg_ppg_segments(
                valid_ecg_segments, 
                valid_ppg_segments,
                ecg_fs=ecg_sampling_rate,
                ppg_fs=ppg_sampling_rate,
                segment_duration=30
            )
            
            # Check for NaN values in HRV features - skip bag if any are NaN
            has_nan = False
            if ecg_hrv_features is not None:
                if (np.isnan(ecg_hrv_features.get('hrv_hf', np.nan)) or 
                    np.isnan(ecg_hrv_features.get('hrv_lf', np.nan))):
                    has_nan = True
            else:
                has_nan = True
            
            if ppg_hrv_features is not None:
                if (np.isnan(ppg_hrv_features.get('hrv_hf', np.nan)) or 
                    np.isnan(ppg_hrv_features.get('hrv_lf', np.nan))):
                    has_nan = True
            else:
                has_nan = True
            
            if has_nan:
                print(f"  Skipping bag {bag_idx} for {patient_id} - NaN values in HRV features")
                # Reset lists for next batch
                valid_ecg_segments = []
                valid_ppg_segments = []
                valid_apnea_labels = []
                valid_sleep_stages = []
                bag_idx += 1
                continue
            
            with h5py.File(h5_file, 'w') as f:
                # Convert lists to numpy arrays
                f.create_dataset('ecg_segments', data=np.array(valid_ecg_segments))
                f.create_dataset('ppg_segments', data=np.array(valid_ppg_segments))
                f.create_dataset('apnea_labels', data=np.array(valid_apnea_labels))
                # Sleep stages are strings, store as variable-length strings
                f.create_dataset('sleep_stages', 
                                data=[s.encode('utf-8') for s in valid_sleep_stages],
                                dtype=h5py.special_dtype(vlen=str))
                # Save HRV features
                ecg_hrv_group = f.create_group('ecg_hrv')
                ecg_hrv_group.create_dataset('hrv_hf', data=ecg_hrv_features['hrv_hf'])
                ecg_hrv_group.create_dataset('hrv_lf', data=ecg_hrv_features['hrv_lf'])
                ppg_hrv_group = f.create_group('ppg_hrv')
                ppg_hrv_group.create_dataset('hrv_hf', data=ppg_hrv_features['hrv_hf'])
                ppg_hrv_group.create_dataset('hrv_lf', data=ppg_hrv_features['hrv_lf'])
            
            print(f"  Saved bag {bag_idx} with 60 segments and HRV features to {h5_file}")
            
            # Reset lists for next batch
            valid_ecg_segments = []
            valid_ppg_segments = []
            valid_apnea_labels = []
            valid_sleep_stages = []
            bag_idx += 1
    
    # Save any remaining segments (< 60) as the final bag
    if len(valid_ecg_segments) > 0:
        # Create output directory structure: Embla_Cleaned/patient_id_followup_id/
        patient_dir = output_base_dir / patient_followup_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as h5py file directly in patient folder
        h5_file = patient_dir / f"{patient_followup_id}_bag_{bag_idx}.h5"
        
        # Check if file already exists, skip if it does
        if h5_file.exists():
            print(f"  Skipping final bag {bag_idx} - file already exists: {h5_file}")
        else:
            # Merge segments back into continuous signals and extract HRV features
            merged_ecg, merged_ppg, ecg_hrv_features, ppg_hrv_features = merge_ecg_ppg_segments(
                valid_ecg_segments, 
                valid_ppg_segments,
                ecg_fs=ecg_sampling_rate,
                ppg_fs=ppg_sampling_rate,
                segment_duration=30
            )
            
            # Check for NaN values in HRV features - skip bag if any are NaN
            has_nan = False
            if ecg_hrv_features is not None:
                if (np.isnan(ecg_hrv_features.get('hrv_hf', np.nan)) or 
                    np.isnan(ecg_hrv_features.get('hrv_lf', np.nan))):
                    has_nan = True
            else:
                has_nan = True
            
            if ppg_hrv_features is not None:
                if (np.isnan(ppg_hrv_features.get('hrv_hf', np.nan)) or 
                    np.isnan(ppg_hrv_features.get('hrv_lf', np.nan))):
                    has_nan = True
            else:
                has_nan = True
            
            if has_nan:
                print(f"  Skipping final bag {bag_idx} for {patient_id} - NaN values in HRV features")
            else:
                with h5py.File(h5_file, 'w') as f:
                    f.create_dataset('ecg_segments', data=np.array(valid_ecg_segments))
                    f.create_dataset('ppg_segments', data=np.array(valid_ppg_segments))
                    f.create_dataset('apnea_labels', data=np.array(valid_apnea_labels))
                    f.create_dataset('sleep_stages',
                                    data=[s.encode('utf-8') for s in valid_sleep_stages],
                                    dtype=h5py.special_dtype(vlen=str))
                    # Save HRV features
                    ecg_hrv_group = f.create_group('ecg_hrv')
                    ecg_hrv_group.create_dataset('hrv_hf', data=ecg_hrv_features['hrv_hf'])
                    ecg_hrv_group.create_dataset('hrv_lf', data=ecg_hrv_features['hrv_lf'])
                    ppg_hrv_group = f.create_group('ppg_hrv')
                    ppg_hrv_group.create_dataset('hrv_hf', data=ppg_hrv_features['hrv_hf'])
                    ppg_hrv_group.create_dataset('hrv_lf', data=ppg_hrv_features['hrv_lf'])
                
                print(f"  Saved final bag {bag_idx} with {len(valid_ecg_segments)} segments and HRV features to {h5_file}")
    
    result['success'] = True
    
    return result


def main():
    """
    Main function to process all files from file_mapping_apnea_only.json.
    Set PLOT_ENABLED=True to generate plots (one per 30-minute segment for each file).
    """
    # Set to True to generate plots, False to skip plotting
    PLOT_ENABLED = False
    
    # Define paths
    base_path = Path(__file__).parent
    mapping_json_path = base_path / 'output' / 'file_mapping_apnea_only.json'
    output_base_dir = base_path.parent / 'data' / 'Embla_Cleaned'
    plot_output_dir = base_path / 'output' / 'plots' if PLOT_ENABLED else None
    
    # Check if mapping file exists
    if not mapping_json_path.exists():
        print(f"Error: Mapping file not found at {mapping_json_path}")
        return
    
    # Load file mapping
    print(f"Loading file mapping from: {mapping_json_path}")
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    valid_mappings = mapping_data.get('valid_mappings', [])
    print(f"Found {len(valid_mappings)} files to process")
    
    if PLOT_ENABLED:
        print("Plotting enabled: Generating plots for each 30-minute segment")
    else:
        print("Plotting disabled: Skipping plot generation")
    
    # Create output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_base_dir}")
    if PLOT_ENABLED:
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plot output directory: {plot_output_dir}")
    
    # Initialize segment counters for CVD tracking
    total_non_cvd_segments_train = 0
    total_cvd_segments_train = 0
    
    # Process each file
    results = []
    for idx, entry in enumerate(valid_mappings, 1):
        print(f"\nProcessing {idx}/{len(valid_mappings)}: {entry.get('edf_path')}")
        result = process_file(entry, output_base_dir, save_plot=PLOT_ENABLED, plot_output_dir=plot_output_dir)
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Success")
            if PLOT_ENABLED and 'plot_paths' in result:
                print(f"  Generated {len(result['plot_paths'])} plot(s)")
            
            # Track segments by CVD status
            has_cvd = entry.get('has_cvd', 'no')
            # Convert "yes"/"no" to 0/1, or handle if already numeric
            if isinstance(has_cvd, str):
                has_cvd_value = 1 if has_cvd.lower() == 'yes' else 0
            else:
                has_cvd_value = int(has_cvd) if has_cvd else 0
            
            total_segments = result.get('total_segments', 0)
            if has_cvd_value == 0:
                total_non_cvd_segments_train += total_segments
            elif has_cvd_value == 1:
                total_cvd_segments_train += total_segments
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_plots = sum(len(r.get('plot_paths', [])) for r in results if r['success'])
    
    print(f"\n{'='*80}")
    print(f"Processing Summary:")
    print(f"  Total files: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if PLOT_ENABLED:
        print(f"  Total plots generated: {total_plots}")
    print(f"\nSegment Counts by CVD Status:")
    print(f"  Non-CVD segments (has_cvd=0): {total_non_cvd_segments_train}")
    print(f"  CVD segments (has_cvd=1): {total_cvd_segments_train}")
    print(f"  Total segments: {total_non_cvd_segments_train + total_cvd_segments_train}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

