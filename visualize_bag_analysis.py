"""
Comprehensive bag-level visualization showing attention, apnea events, and sleep stages.

This script creates detailed visualizations for each bag, showing:
- Attention weights across segments
- Apnea event timeline
- Sleep stage transitions
- Highlighted important segments with heatmaps
- Zoomed-in view of highest attention segment with context (30s before and after)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import importlib.util
import json
from collections import defaultdict

# ============================================================================
# CONFIGURATION VARIABLES - EDIT THESE
# ============================================================================
# Set to None to process all patients, or specify a patient ID to process only that patient
PATIENT_ID_FILTER = None  # e.g., "49010001" or None for all
# Set to None to process all follow-ups, or specify a follow-up number
FOLLOW_UP_FILTER = None  # e.g., 1 or None for all
# JSON file to use (file_mapping_apnea_only.json or file_mapping.json)
JSON_FILE = "file_mapping_apnea_only.json"  # or "file_mapping.json"
# ============================================================================

# Add parent directory and model directory to path
project_root = Path(__file__).parent
model_dir = project_root / 'model'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(model_dir))

# Import directly from files using importlib to avoid package issues
# Import config
config_spec = importlib.util.spec_from_file_location("config", model_dir / "config.py")
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
# JSON_PATH will be determined dynamically based on JSON_FILE variable
# Keep original for fallback
JSON_PATH = config_module.JSON_PATH
DATA_DIR = config_module.DATA_DIR

# Import data_loader
data_loader_spec = importlib.util.spec_from_file_location("data_loader", model_dir / "data_loader.py")
data_loader_module = importlib.util.module_from_spec(data_loader_spec)
data_loader_spec.loader.exec_module(data_loader_module)
load_data_from_entries = data_loader_module.load_data_from_entries
get_folder_name = data_loader_module.get_folder_name

# Import normalization
normalization_spec = importlib.util.spec_from_file_location("normalization", model_dir / "normalization.py")
normalization_module = importlib.util.module_from_spec(normalization_spec)
normalization_spec.loader.exec_module(normalization_module)
calculate_normalization_coefficients = normalization_module.calculate_normalization_coefficients
apply_normalization = normalization_module.apply_normalization

# Import dataset
dataset_spec = importlib.util.spec_from_file_location("dataset", model_dir / "dataset.py")
dataset_module = importlib.util.module_from_spec(dataset_spec)
dataset_spec.loader.exec_module(dataset_module)
create_datasets_from_splits = dataset_module.create_datasets_from_splits

# Import architecture
architecture_spec = importlib.util.spec_from_file_location("architecture", model_dir / "architecture.py")
architecture_module = importlib.util.module_from_spec(architecture_spec)
architecture_spec.loader.exec_module(architecture_module)
FUSEDNet = architecture_module.FUSEDNet

from torch.utils.data import DataLoader

# Sleep stage mapping for display
SLEEP_STAGE_NAMES = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM',
    5: 'Artifact'
}

SLEEP_STAGE_COLORS = {
    0: '#FF6B6B',  # Red - Wake
    1: '#4ECDC4',  # Teal - N1
    2: '#45B7D1',  # Blue - N2
    3: '#96CEB4',  # Green - N3
    4: '#FFEAA7',  # Yellow - REM
    5: '#DDA0DD'   # Plum - Artifact
}

# Signal parameters
ECG_FS = 200  # ECG sampling rate
PPG_FS = 100  # PPG sampling rate
SEGMENT_DURATION = 30  # seconds
OVERLAP_RATIO = 0.3  # 30% overlap


def load_specific_followup_data(patient_id: str, follow_up: int, json_path: Path, data_dir: Path):
    """
    Load data for a specific patient and follow-up.
    
    Args:
        patient_id: Patient ID (e.g., '49010001')
        follow_up: Follow-up number (1, 2, 3, etc.) or None for baseline
        json_path: Path to file_mapping JSON
        data_dir: Path to data directory
    
    Returns:
        Tuple of (normalized_bags, original_bags, entries, norm_coefficients)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find entries for this specific patient and follow-up
    target_entries = []
    for entry in data.get('valid_mappings', []):
        if entry.get('patient_id') == patient_id:
            entry_follow_up = entry.get('follow_up')
            if follow_up is None:
                # Looking for baseline
                if entry_follow_up is None:
                    target_entries.append(entry)
            else:
                # Looking for specific follow-up
                if entry_follow_up == follow_up:
                    target_entries.append(entry)
    
    if not target_entries:
        follow_up_str = "baseline" if follow_up is None else f"follow-up {follow_up}"
        raise ValueError(f"No entries found for patient {patient_id}, {follow_up_str}")
    
    print(f"Found {len(target_entries)} entries for patient {patient_id}, follow-up {follow_up}")
    
    # Load data
    patient_data = load_data_from_entries(target_entries, data_dir)
    import copy
    original_bags = copy.deepcopy(patient_data['bags'])
    
    norm_coefficients = calculate_normalization_coefficients(patient_data['bags'])
    normalized_bags = apply_normalization(patient_data['bags'], norm_coefficients)
    
    return normalized_bags, original_bags, target_entries, norm_coefficients


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model = FUSEDNet(
        embed_dim=64,
        sleep_embed_dim=8,
        hrv_dim=4,
        use_ecg_ppg=True,
        use_hrv=True,
        use_ppg_ft=False,
        use_sleep_stage=True,
        use_apnea=True,
        attn_dim=32,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    if 'module.' in list(state_dict.keys())[0]:
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.module
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(device)
    print(f"Model loaded from {checkpoint_path}")
    return model


def merge_segments_with_overlap_removal(segments: list, segment_duration: int, signal_fs: int) -> np.ndarray:
    """
    Merge overlapping segments back into a continuous signal by removing the overlap.
    
    Segments have 30% overlap. This function removes the overlap to create a continuous signal.
    """
    if len(segments) == 0:
        return np.array([])
    
    segment_size = segment_duration * signal_fs
    overlap_size = int(segment_size * OVERLAP_RATIO)
    non_overlap_size = segment_size - overlap_size
    
    merged_signal = segments[0].copy()
    
    for i in range(1, len(segments)):
        segment = segments[i]
        if len(segment) >= overlap_size:
            merged_signal = np.concatenate([merged_signal, segment[overlap_size:]])
        else:
            merged_signal = np.concatenate([merged_signal, segment])
    
    return merged_signal


def get_90_second_context(
    ecg_segments: list,
    ppg_segments: list,
    segment_idx: int,
    attention_weights: np.ndarray
):
    """
    Get 90 seconds of signal (30s before + 30s middle + 30s after) for a given segment.
    
    Args:
        ecg_segments: List of ECG segments
        ppg_segments: List of PPG segments
        segment_idx: Index of the middle segment (highest attention)
        attention_weights: Array of attention weights for all segments
    
    Returns:
        Tuple of (ecg_90s, ppg_90s, time_ecg, time_ppg, segment_boundaries, attention_map)
    """
    num_segments = len(ecg_segments)
    
    # Get indices for before, middle, and after segments
    before_idx = segment_idx - 1 if segment_idx > 0 else None
    middle_idx = segment_idx
    after_idx = segment_idx + 1 if segment_idx < num_segments - 1 else None
    
    # Collect segments to merge
    ecg_segs_to_merge = []
    ppg_segs_to_merge = []
    seg_indices = []
    
    if before_idx is not None:
        ecg_segs_to_merge.append(ecg_segments[before_idx])
        ppg_segs_to_merge.append(ppg_segments[before_idx])
        seg_indices.append(before_idx)
    
    ecg_segs_to_merge.append(ecg_segments[middle_idx])
    ppg_segs_to_merge.append(ppg_segments[middle_idx])
    seg_indices.append(middle_idx)
    
    if after_idx is not None:
        ecg_segs_to_merge.append(ecg_segments[after_idx])
        ppg_segs_to_merge.append(ppg_segments[after_idx])
        seg_indices.append(after_idx)
    
    # Merge segments removing overlap
    ecg_90s = merge_segments_with_overlap_removal(ecg_segs_to_merge, SEGMENT_DURATION, ECG_FS)
    ppg_90s = merge_segments_with_overlap_removal(ppg_segs_to_merge, SEGMENT_DURATION, PPG_FS)
    
    # Create time arrays
    time_ecg = np.arange(len(ecg_90s)) / ECG_FS
    time_ppg = np.arange(len(ppg_90s)) / PPG_FS
    
    # Calculate segment boundaries in the merged signal
    # First segment: full 30s
    # Second segment: starts after removing overlap from first
    segment_size_ecg = SEGMENT_DURATION * ECG_FS
    segment_size_ppg = SEGMENT_DURATION * PPG_FS
    overlap_size_ecg = int(segment_size_ecg * OVERLAP_RATIO)
    overlap_size_ppg = int(segment_size_ppg * OVERLAP_RATIO)
    
    boundaries_ecg = []
    boundaries_ppg = []
    current_pos_ecg = 0
    current_pos_ppg = 0
    
    for i, seg_idx in enumerate(seg_indices):
        if i == 0:
            # First segment: take full length
            boundaries_ecg.append((current_pos_ecg, current_pos_ecg + segment_size_ecg))
            boundaries_ppg.append((current_pos_ppg, current_pos_ppg + segment_size_ppg))
            current_pos_ecg += segment_size_ecg
            current_pos_ppg += segment_size_ppg
        else:
            # Subsequent segments: skip overlap
            start_ecg = current_pos_ecg
            start_ppg = current_pos_ppg
            end_ecg = start_ecg + (segment_size_ecg - overlap_size_ecg)
            end_ppg = start_ppg + (segment_size_ppg - overlap_size_ppg)
            boundaries_ecg.append((start_ecg, end_ecg))
            boundaries_ppg.append((start_ppg, end_ppg))
            current_pos_ecg = end_ecg
            current_pos_ppg = end_ppg
    
    # Create attention map for the 90-second window
    attention_map_ecg = np.zeros(len(ecg_90s))
    attention_map_ppg = np.zeros(len(ppg_90s))
    
    for i, seg_idx in enumerate(seg_indices):
        attn = attention_weights[seg_idx]
        start_ecg, end_ecg = boundaries_ecg[i]
        start_ppg, end_ppg = boundaries_ppg[i]
        
        # Ensure indices are within bounds
        end_ecg = min(end_ecg, len(attention_map_ecg))
        end_ppg = min(end_ppg, len(attention_map_ppg))
        
        attention_map_ecg[start_ecg:end_ecg] = attn
        attention_map_ppg[start_ppg:end_ppg] = attn
    
    return ecg_90s, ppg_90s, time_ecg, time_ppg, boundaries_ecg, boundaries_ppg, attention_map_ecg, attention_map_ppg, seg_indices


def analyze_bags(
    model: nn.Module,
    bags: list,
    original_bags: list,
    entries: list,
    data_dir: Path,
    device: torch.device
):
    """
    Analyze all bags and collect attention, apnea, and sleep stage data.
    
    Returns:
        List of bag analysis dictionaries
    """
    dataset, _ = create_datasets_from_splits(
        train_bags=bags,
        test_bags=[],
        train_entries=entries,
        test_entries=[],
        data_dir=str(data_dir),
        max_instances=60,
        min_instances=15,
        ecg_fs=ECG_FS,
        ppg_fs=PPG_FS,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No valid bags found")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    bag_analyses = []
    
    print(f"Analyzing {len(dataloader)} bags...")
    with torch.no_grad():
        for bag_idx, (input_dict, label) in enumerate(dataloader):
            ecg = input_dict.get('ecg', None)
            ppg = input_dict.get('ppg', None)
            if ecg is not None:
                ecg = ecg.to(device)
            if ppg is not None:
                ppg = ppg.to(device)
            
            mask = input_dict['mask'].to(device)
            hrv = input_dict['hrv'].to(device)
            sleep_stage = input_dict['sleep_stage'].to(device)
            apnea = input_dict['apnea_label'].to(device)
            record_id = input_dict.get('record_id', [f'bag_{bag_idx}'])[0]
            
            # Forward pass
            output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
            
            if isinstance(output, tuple):
                logits, attn_weights = output
            else:
                print(f"Warning: Bag {bag_idx} did not return attention weights")
                continue
            
            # Get valid segments
            valid_mask = mask[0].cpu().bool().numpy()
            attn_np = attn_weights[0].cpu().numpy()
            sleep_stage_np = sleep_stage[0].cpu().numpy()
            apnea_np = apnea[0].cpu().numpy()
            
            # Extract only valid segments
            valid_attn = attn_np[valid_mask]
            valid_sleep = sleep_stage_np[valid_mask].astype(int)
            valid_apnea = apnea_np[valid_mask].astype(int)
            
            # Get raw ECG and PPG signals from original bags
            original_bag = None
            for orig_bag in original_bags:
                bag_file_path = orig_bag.get('file_path', '')
                if str(record_id) in bag_file_path or bag_file_path.endswith(str(record_id)):
                    original_bag = orig_bag
                    break
            
            if original_bag is None and bag_idx < len(original_bags):
                original_bag = original_bags[bag_idx]
            
            # Extract raw ECG and PPG segments
            raw_ecg_segments = None
            raw_ppg_segments = None
            if original_bag:
                ecg_segs = original_bag.get('ecg_segments', None)
                ppg_segs = original_bag.get('ppg_segments', None)
                if ecg_segs is not None and isinstance(ecg_segs, np.ndarray):
                    raw_ecg_segments = ecg_segs[valid_mask] if len(ecg_segs) >= len(valid_mask) else None
                if ppg_segs is not None and isinstance(ppg_segs, np.ndarray):
                    raw_ppg_segments = ppg_segs[valid_mask] if len(ppg_segs) >= len(valid_mask) else None
            
            # Get prediction
            prob = torch.sigmoid(logits[0]).cpu().item()
            pred = 1 if prob > 0.5 else 0
            true_label = int(label[0].cpu().item())
            
            bag_analysis = {
                'bag_idx': bag_idx,
                'record_id': str(record_id),
                'attention_weights': valid_attn,
                'sleep_stages': valid_sleep,
                'apnea_labels': valid_apnea,
                'ecg_segments': raw_ecg_segments,
                'ppg_segments': raw_ppg_segments,
                'prediction_prob': prob,
                'prediction': pred,
                'true_label': true_label,
                'num_segments': len(valid_attn)
            }
            bag_analyses.append(bag_analysis)
    
    return bag_analyses


def visualize_highest_attention_segment(
    bag_analysis: dict,
    patient_id: str,
    follow_up: int,
    output_path: Path
):
    """
    Create visualization for the highest attention segment with 30s before and after.
    
    Args:
        bag_analysis: Dictionary with bag analysis data
        patient_id: Patient ID
        follow_up: Follow-up number
        output_path: Path to save the figure
    """
    attn_weights = bag_analysis['attention_weights']
    sleep_stages = bag_analysis['sleep_stages']
    apnea_labels = bag_analysis['apnea_labels']
    ecg_segments = bag_analysis.get('ecg_segments', None)
    ppg_segments = bag_analysis.get('ppg_segments', None)
    bag_idx = bag_analysis['bag_idx']
    record_id = bag_analysis['record_id']
    prob = bag_analysis['prediction_prob']
    pred = bag_analysis['prediction']
    true_label = bag_analysis['true_label']
    
    if ecg_segments is None or ppg_segments is None:
        print(f"Warning: No raw signals available for bag {bag_idx}")
        return
    
    # Find segment with highest attention
    highest_attn_idx = np.argmax(attn_weights)
    highest_attn = attn_weights[highest_attn_idx]
    
    # Debug: Print attention values for the 3 segments we'll visualize
    print(f"Bag {bag_idx}: Attention range: [{attn_weights.min():.6f}, {attn_weights.max():.6f}], "
          f"Highest at idx {highest_attn_idx} = {highest_attn:.6f}")
    
    # Get 90-second context (30s before + 30s middle + 30s after)
    ecg_90s, ppg_90s, time_ecg, time_ppg, boundaries_ecg, boundaries_ppg, \
    attention_map_ecg, attention_map_ppg, seg_indices = get_90_second_context(
        ecg_segments, ppg_segments, highest_attn_idx, attn_weights
    )
    
    # Get sleep stages and apnea labels for the 3 segments
    sleep_stages_90s = [int(sleep_stages[idx]) for idx in seg_indices]
    apnea_labels_90s = [int(apnea_labels[idx]) for idx in seg_indices]
    
    # Get attention weights for the 3 segments
    attn_weights_90s = [attn_weights[idx] for idx in seg_indices]
    
    # Normalize attention weights for printing (min-max normalization across full bag)
    attn_min = attn_weights.min()
    attn_max = attn_weights.max()
    attn_range = attn_max - attn_min
    if attn_range > 0:
        norm_attn_90s = [(aw - attn_min) / attn_range for aw in attn_weights_90s]
    else:
        norm_attn_90s = [0.5] * len(attn_weights_90s)
    
    # Debug output with normalized attention values
    print(f"  Segments indices: {seg_indices}")
    if len(norm_attn_90s) == 3:
        print(f"  Normalized Attention: Before={norm_attn_90s[0]:.6f} (idx {seg_indices[0]}), "
              f"Middle={norm_attn_90s[1]:.6f} (idx {seg_indices[1]}), "
              f"After={norm_attn_90s[2]:.6f} (idx {seg_indices[2]})")
    else:
        print(f"  Normalized Attention: {norm_attn_90s}")
    print(f"  Sleep stages (raw): {[sleep_stages[idx] for idx in seg_indices]}")
    print(f"  Sleep stages (int): {sleep_stages_90s}")
    print(f"  Apnea labels: {apnea_labels_90s}")
    
    # Normalize attention map relative to FULL BAG's attention range (min-max normalization)
    # This ensures we see the relative importance within the full context
    attn_min = attn_weights.min()
    attn_max = attn_weights.max()
    attn_range = attn_max - attn_min
    
    if attn_range > 0:
        # Normalize to 0-1 range based on full bag's attention
        norm_attn_ecg = (attention_map_ecg - attn_min) / attn_range
        norm_attn_ppg = (attention_map_ppg - attn_min) / attn_range
    else:
        # All attention values are the same
        norm_attn_ecg = np.ones_like(attention_map_ecg) * 0.5
        norm_attn_ppg = np.ones_like(attention_map_ppg) * 0.5
    
    # Clip to [0, 1] to ensure valid colormap values
    norm_attn_ecg = np.clip(norm_attn_ecg, 0, 1)
    norm_attn_ppg = np.clip(norm_attn_ppg, 0, 1)
    
    # Create figure with 4 rows
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.5, 3.5, 1, 1], hspace=0.4)
    
    # Use a colormap from blue (low) to red (high)
    cmap = plt.cm.get_cmap('RdYlBu_r')  # Reversed: blue=low, red=high
    
    # Row 1: ECG Signal with Attention Heatmap
    ax1 = fig.add_subplot(gs[0])
    
    # Calculate y-limits from data
    y_min_ecg, y_max_ecg = ecg_90s.min(), ecg_90s.max()
    y_range_ecg = y_max_ecg - y_min_ecg
    y_min_ecg -= y_range_ecg * 0.1
    y_max_ecg += y_range_ecg * 0.1
    ax1.set_ylim(y_min_ecg, y_max_ecg)
    ax1.set_xlim(time_ecg[0], time_ecg[-1])
    
    # Create a background heatmap using imshow for better gradient
    # Resample attention map for visualization
    num_samples = 500  # Number of heatmap segments
    indices = np.linspace(0, len(norm_attn_ecg) - 1, num_samples, dtype=int)
    attn_sampled = norm_attn_ecg[indices]
    time_sampled = time_ecg[indices]
    
    # Create 2D array for imshow (1 row)
    attn_2d = attn_sampled.reshape(1, -1)
    
    # Use imshow to create heatmap background
    im1 = ax1.imshow(attn_2d, aspect='auto', extent=[time_ecg[0], time_ecg[-1], y_min_ecg, y_max_ecg],
                     cmap=cmap, alpha=0.5, interpolation='bilinear', zorder=0, origin='lower')
    
    # Plot ECG signal on top
    ax1.plot(time_ecg, ecg_90s, 'b-', linewidth=2.5, alpha=0.95, label='ECG Signal', zorder=2)
    
    # Draw red vertical lines at segment boundaries and label segments with attention weights
    for i, (start, end) in enumerate(boundaries_ecg):
        start_time = start / ECG_FS
        end_time = end / ECG_FS
        ax1.axvline(x=start_time, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        ax1.axvline(x=end_time, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        # Label each segment with attention weight
        mid_time = (start_time + end_time) / 2
        seg_idx = seg_indices[i]
        attn_val = attn_weights[seg_idx]
        if i == 1:  # Middle segment (highest attention)
            ax1.text(mid_time, ax1.get_ylim()[1] * 0.95, f'Seg {seg_idx}\n(Attn: {attn_val:.4f})',
                    ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:  # Before or after segment
            label = 'Before' if i == 0 else 'After'
            ax1.text(mid_time, ax1.get_ylim()[1] * 0.88, f'{label}\nSeg {seg_idx}\n(Attn: {attn_val:.4f})',
                    ha='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax1.set_ylabel('ECG (mV)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title(f'ECG Signal with Attention Heatmap (Blue=Low, Red=High)', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add colorbar for attention scale
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02)
    cbar1.set_label('Attention (Normalized)', fontsize=10, rotation=270, labelpad=15)
    
    # Row 2: PPG Signal with Attention Heatmap
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate y-limits from data
    y_min_ppg, y_max_ppg = ppg_90s.min(), ppg_90s.max()
    y_range_ppg = y_max_ppg - y_min_ppg
    y_min_ppg -= y_range_ppg * 0.1
    y_max_ppg += y_range_ppg * 0.1
    ax2.set_ylim(y_min_ppg, y_max_ppg)
    ax2.set_xlim(time_ppg[0], time_ppg[-1])
    
    # Create background heatmap using imshow
    num_samples_ppg = 250  # Number of heatmap segments for PPG
    indices_ppg = np.linspace(0, len(norm_attn_ppg) - 1, num_samples_ppg, dtype=int)
    attn_ppg_sampled = norm_attn_ppg[indices_ppg]
    time_ppg_sampled = time_ppg[indices_ppg]
    
    # Create 2D array for imshow
    attn_ppg_2d = attn_ppg_sampled.reshape(1, -1)
    
    # Use imshow to create heatmap background
    im2 = ax2.imshow(attn_ppg_2d, aspect='auto', extent=[time_ppg[0], time_ppg[-1], y_min_ppg, y_max_ppg],
                     cmap=cmap, alpha=0.5, interpolation='bilinear', zorder=0, origin='lower')
    
    # Plot PPG signal on top
    ax2.plot(time_ppg, ppg_90s, 'r-', linewidth=2.5, alpha=0.95, label='PPG Signal', zorder=2)
    
    # Draw red vertical lines at segment boundaries and label segments with attention weights
    for i, (start, end) in enumerate(boundaries_ppg):
        start_time = start / PPG_FS
        end_time = end / PPG_FS
        ax2.axvline(x=start_time, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        ax2.axvline(x=end_time, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        # Label each segment with attention weight
        mid_time = (start_time + end_time) / 2
        seg_idx = seg_indices[i]
        attn_val = attn_weights[seg_idx]
        if i == 1:  # Middle segment (highest attention)
            ax2.text(mid_time, ax2.get_ylim()[1] * 0.95, f'Seg {seg_idx}\n(Attn: {attn_val:.4f})',
                    ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:  # Before or after segment
            label = 'Before' if i == 0 else 'After'
            ax2.text(mid_time, ax2.get_ylim()[1] * 0.88, f'{label}\nSeg {seg_idx}\n(Attn: {attn_val:.4f})',
                    ha='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax2.set_ylabel('PPG (au)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'PPG Signal with Attention Heatmap (Blue=Low, Red=High)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add colorbar for attention scale
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02)
    cbar2.set_label('Attention (Normalized)', fontsize=10, rotation=270, labelpad=15)
    
    # Row 3: Apnea Timeline
    ax3 = fig.add_subplot(gs[2])
    
    # Create timeline for 90 seconds
    # Map each segment to its time range
    apnea_timeline = np.zeros(len(time_ecg))
    for i, seg_idx in enumerate(seg_indices):
        start, end = boundaries_ecg[i]
        end = min(end, len(apnea_timeline))
        apnea_timeline[start:end] = apnea_labels_90s[i]
    
    # Create bar chart
    apnea_colors = ['#90EE90' if a == 0 else '#FF6B6B' for a in apnea_timeline[::100]]  # Sample every 100 points
    ax3.bar(time_ecg[::100], apnea_timeline[::100], width=time_ecg[1]*100, color=apnea_colors, 
            edgecolor='black', linewidth=0.3, alpha=0.7)
    
    # Draw segment boundaries
    for start, end in boundaries_ecg:
        start_time = start / ECG_FS
        end_time = end / ECG_FS
        ax3.axvline(x=start_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.axvline(x=end_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax3.set_ylabel('Apnea Event', fontsize=12, fontweight='bold')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Normal', 'Apnea'])
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Row 4: Sleep Stage Timeline
    ax4 = fig.add_subplot(gs[3])
    
    # Debug: Print sleep stages for the 3 segments
    print(f"  Sleep stages for 3 segments: {sleep_stages_90s} (indices: {seg_indices})")
    
    # Create timeline for sleep stages - use a more visible approach
    # Instead of sampling, create continuous blocks for each segment
    sleep_timeline_values = []
    sleep_timeline_times = []
    sleep_timeline_colors = []
    
    for i, seg_idx in enumerate(seg_indices):
        start, end = boundaries_ecg[i]
        start_time = start / ECG_FS
        end_time = end / ECG_FS
        
        # Get sleep stage for this segment
        sleep_stage_val = int(sleep_stages_90s[i])
        sleep_stage_name = SLEEP_STAGE_NAMES.get(sleep_stage_val, f'Unknown({sleep_stage_val})')
        sleep_stage_color = SLEEP_STAGE_COLORS.get(sleep_stage_val, '#CCCCCC')
        
        # Create a bar for this entire segment
        sleep_timeline_values.append(sleep_stage_val)
        sleep_timeline_times.append((start_time + end_time) / 2)  # Center of segment
        sleep_timeline_colors.append(sleep_stage_color)
        
        # Also add boundaries
        ax4.axvline(x=start_time, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        ax4.axvline(x=end_time, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    
    # Create bar chart with wider bars for visibility
    if len(sleep_timeline_values) > 0:
        # Calculate bar width based on segment duration
        bar_width = 20  # 20 seconds wide bars for visibility
        ax4.bar(sleep_timeline_times, sleep_timeline_values, width=bar_width, 
                color=sleep_timeline_colors, edgecolor='black', linewidth=1.5, 
                alpha=0.8, zorder=1)
        
        # Add text labels for each segment
        for i, (time_pos, stage_val, stage_name) in enumerate(zip(sleep_timeline_times, sleep_timeline_values, 
                                                                  [SLEEP_STAGE_NAMES.get(int(s), 'Unknown') for s in sleep_timeline_values])):
            ax4.text(time_pos, stage_val + 0.3, stage_name, ha='center', fontsize=9, 
                    fontweight='bold', zorder=3)
    
    ax4.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
    ax4.set_ylim(-0.5, 5.5)
    ax4.set_yticks(range(6))
    ax4.set_yticklabels([SLEEP_STAGE_NAMES[i] for i in range(6)])
    ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_xlim(time_ecg[0], time_ecg[-1])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Overall title with prediction info
    title = (f'Bag {bag_idx} - Highest Attention Segment (Idx: {highest_attn_idx}, Attn: {highest_attn:.4f})\n'
             f'Patient {patient_id}, Follow-up {follow_up} | Pred: {prob:.4f} ({pred}) | True: {true_label}')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function - loops through all patients and files."""
    # Determine which JSON file to use
    json_file_path = project_root / "data_preparation" / "output" / JSON_FILE
    if not json_file_path.exists():
        # Fallback to regular file_mapping.json
        json_file_path = JSON_PATH
        print(f"Warning: {JSON_FILE} not found, using {json_file_path}")
    
    # Load JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group entries by patient_id and follow_up
    patient_file_groups = defaultdict(dict)  # {patient_id: {follow_up: [entries]}}
    
    for entry in data.get('valid_mappings', []):
        patient_id = entry.get('patient_id')
        follow_up = entry.get('follow_up')
        
        if not patient_id:
            continue
        
        # Apply filters if specified
        if PATIENT_ID_FILTER is not None and patient_id != PATIENT_ID_FILTER:
            continue
        if FOLLOW_UP_FILTER is not None and follow_up != FOLLOW_UP_FILTER:
            continue
        
        if patient_id not in patient_file_groups:
            patient_file_groups[patient_id] = {}
        
        if follow_up not in patient_file_groups[patient_id]:
            patient_file_groups[patient_id][follow_up] = []
        
        patient_file_groups[patient_id][follow_up].append(entry)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Processing {len(patient_file_groups)} patients...")
    print("="*80)
    
    total_bags_processed = 0
    
    # Process each patient
    for patient_id, follow_ups_dict in sorted(patient_file_groups.items()):
        print(f"\n{'='*80}")
        print(f"Processing Patient: {patient_id}")
        print(f"{'='*80}")
        
        # Check if model exists for this patient
        checkpoint_path = project_root / f"best_model_loso_patient_{patient_id}.pt"
        if not checkpoint_path.exists():
            print(f"  Skipping {patient_id}: Model checkpoint not found at {checkpoint_path}")
            continue
        
        # IMPORTANT: Load model for THIS patient (each patient has their own LOSO model)
        print(f"  Loading model for patient {patient_id}: {checkpoint_path}")
        model = load_model(str(checkpoint_path), device)
        print(f"  Model loaded successfully for patient {patient_id}")
        
        # Process each follow-up for this patient
        for follow_up, entries in sorted(follow_ups_dict.items()):
            follow_up_str = str(follow_up) if follow_up is not None else "baseline"
            print(f"\n  Processing Follow-up: {follow_up_str} ({len(entries)} file(s))")
            
            # Construct output directory: patient_id/follow_up/
            output_base_dir = project_root / "results" / "bag_analysis" / patient_id / follow_up_str
            
            try:
                # Load patient data for this follow-up
                normalized_bags, original_bags, file_entries, norm_coefficients = load_specific_followup_data(
                    patient_id, follow_up, json_file_path, DATA_DIR
                )
                
                if not normalized_bags:
                    print(f"    No bags found for {patient_id}, follow-up {follow_up_str}")
                    continue
                
                # Analyze all bags
                print(f"    Analyzing {len(normalized_bags)} bags...")
                bag_analyses = analyze_bags(
                    model=model,
                    bags=normalized_bags,
                    original_bags=original_bags,
                    entries=file_entries,
                    data_dir=DATA_DIR,
                    device=device
                )
                
                if not bag_analyses:
                    print(f"    No bag analyses found for {patient_id}, follow-up {follow_up_str}")
                    continue
                
                # Visualize each bag (highest attention segment)
                print(f"    Generating visualizations for {len(bag_analyses)} bags...")
                for bag_analysis in bag_analyses:
                    bag_idx = bag_analysis['bag_idx']
                    
                    # Create output path: patient_id/follow_up/bag_n.png
                    output_path = output_base_dir / f"bag_{bag_idx}.png"
                    
                    visualize_highest_attention_segment(
                        bag_analysis=bag_analysis,
                        patient_id=patient_id,
                        follow_up=follow_up,
                        output_path=output_path
                    )
                    total_bags_processed += 1
                
                # Print summary for this follow-up
                print(f"\n    Summary for {patient_id}, follow-up {follow_up_str}:")
                for bag_analysis in bag_analyses:
                    max_attn_idx = np.argmax(bag_analysis['attention_weights'])
                    max_attn = bag_analysis['attention_weights'][max_attn_idx]
                    # Normalize for display
                    attn_min = bag_analysis['attention_weights'].min()
                    attn_max = bag_analysis['attention_weights'].max()
                    attn_range = attn_max - attn_min
                    norm_max_attn = (max_attn - attn_min) / attn_range if attn_range > 0 else 0.5
                    print(f"      Bag {bag_analysis['bag_idx']}: Max Attn Seg={max_attn_idx}, "
                          f"Norm Attn={norm_max_attn:.4f}, Pred={bag_analysis['prediction_prob']:.4f}, "
                          f"True={bag_analysis['true_label']}")
                print(f"    Figures saved to: {output_base_dir}")
                
            except Exception as e:
                print(f"    Error processing {patient_id}, follow-up {follow_up_str}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clear model from memory before moving to next patient
        # (PyTorch will handle this, but being explicit)
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Completed patient {patient_id}, moving to next patient...\n")
    
    print("\n" + "="*80)
    print(f"COMPLETE: Processed {total_bags_processed} bags across all patients")
    print("="*80)


if __name__ == '__main__':
    main()
