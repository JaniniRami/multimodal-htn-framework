"""
Visualization utilities for ECG-PPG bag data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
import random
from typing import Dict, Optional, List

# Sleep stage mapping (reverse of the one in dataset.py for display)
SLEEP_STAGE_NAMES = {
    0: 'SLEEP-S0',
    1: 'SLEEP-S1',
    2: 'SLEEP-S2',
    3: 'SLEEP-S3',
    4: 'SLEEP-S4',
    5: 'SLEEP-REM'
}


def visualize_random_bag(
    train_loader,
    ecg_fs: int = 200,
    ppg_fs: int = 100,
    segment_duration: int = 30,
    figsize: tuple = (20, 12),
    save_path: Optional[str] = None,
    random_seed: Optional[int] = None
):
    """
    Visualize a random bag from the training loader.
    
    Creates a 4-row figure showing:
    1. Full ECG signal (30 minutes, concatenated segments)
    2. Full PPG signal (30 minutes, concatenated segments)
    3. Apnea labels (binary, one per segment)
    4. Sleep stages (categorical, one per segment)
    
    Args:
        train_loader: PyTorch DataLoader for training data
        ecg_fs: ECG sampling rate in Hz (default: 200)
        ppg_fs: PPG sampling rate in Hz (default: 100)
        segment_duration: Duration of each segment in seconds (default: 30)
        figsize: Figure size tuple (width, height) in inches (default: (20, 12))
        save_path: Optional path to save the figure. If None, displays interactively
        random_seed: Optional random seed for reproducibility
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Get a random batch
    batch = next(iter(train_loader))
    input_dict, cvd_labels = batch
    
    # Select a random bag from the batch
    batch_size = input_dict['ecg'].shape[0]
    bag_idx = random.randint(0, batch_size - 1)
    
    # Extract data for the selected bag
    ecg = input_dict['ecg'][bag_idx].cpu().numpy()  # Shape: (max_instances, L_ecg)
    ppg = input_dict['ppg'][bag_idx].cpu().numpy()  # Shape: (max_instances, L_ppg)
    apnea_labels = input_dict['apnea_label'][bag_idx].cpu().numpy()  # Shape: (max_instances,)
    sleep_stages = input_dict['sleep_stage'][bag_idx].cpu().numpy()  # Shape: (max_instances,)
    mask = input_dict['mask'][bag_idx].cpu().numpy()  # Shape: (max_instances,)
    record_id = input_dict['record_id'][bag_idx]
    cvd_label = cvd_labels[bag_idx].item()
    
    # Get only valid segments (using mask)
    valid_indices = np.where(mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        print("Warning: No valid segments found in selected bag")
        return
    
    ecg_valid = ecg[valid_indices]
    ppg_valid = ppg[valid_indices]
    apnea_valid = apnea_labels[valid_indices]
    sleep_stages_valid = sleep_stages[valid_indices]
    
    # Concatenate segments to form full 30-minute signals
    ecg_full = np.concatenate(ecg_valid, axis=0)
    ppg_full = np.concatenate(ppg_valid, axis=0)
    
    # Create time axes
    ecg_time = np.arange(len(ecg_full)) / ecg_fs  # Time in seconds
    ppg_time = np.arange(len(ppg_full)) / ppg_fs  # Time in seconds
    
    # Convert time to minutes for x-axis
    ecg_time_min = ecg_time / 60.0
    ppg_time_min = ppg_time / 60.0
    
    # Create segment time axis for labels (center of each segment)
    segment_centers = np.arange(n_valid) * segment_duration + segment_duration / 2
    segment_centers_min = segment_centers / 60.0
    
    # Convert sleep stage integers to names
    sleep_stage_names = [SLEEP_STAGE_NAMES.get(int(stage), 'UNKNOWN') for stage in sleep_stages_valid]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)
    fig.suptitle(f'Bag Visualization - Record ID: {record_id}, CVD Label: {int(cvd_label)}, '
                 f'Valid Segments: {n_valid}', fontsize=16, fontweight='bold')
    
    # Plot 1: ECG Signal
    ax1 = axes[0]
    ax1.plot(ecg_time_min, ecg_full, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('ECG Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('ECG Signal (30 minutes)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(ecg_time_min))
    
    # Plot 2: PPG Signal
    ax2 = axes[1]
    ax2.plot(ppg_time_min, ppg_full, 'r-', linewidth=0.5, alpha=0.7)
    ax2.set_ylabel('PPG Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('PPG Signal (30 minutes)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(ppg_time_min))
    
    # Plot 3: Apnea Labels
    ax3 = axes[2]
    # Create a step plot for apnea labels
    apnea_extended = np.repeat(apnea_valid, 2)
    time_extended = np.repeat(segment_centers_min, 2)
    time_extended[1::2] += segment_duration / 60.0 / 2
    time_extended = np.concatenate([[0], time_extended, [max(segment_centers_min) + segment_duration / 60.0]])
    apnea_extended = np.concatenate([[apnea_valid[0]], apnea_extended, [apnea_valid[-1]]])
    
    ax3.fill_between(time_extended, 0, apnea_extended, step='pre', alpha=0.6, color='red', label='Apnea')
    ax3.plot(segment_centers_min, apnea_valid, 'ro', markersize=4)
    ax3.set_ylabel('Apnea Label', fontsize=12, fontweight='bold')
    ax3.set_title('Apnea Labels (1 = Apnea, 0 = No Apnea)', fontsize=14, fontweight='bold')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No Apnea', 'Apnea'])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Sleep Stages
    ax4 = axes[3]
    # Create a step plot for sleep stages
    sleep_stage_numeric = sleep_stages_valid.astype(float)
    sleep_stage_extended = np.repeat(sleep_stage_numeric, 2)
    time_extended_ss = np.repeat(segment_centers_min, 2)
    time_extended_ss[1::2] += segment_duration / 60.0 / 2
    time_extended_ss = np.concatenate([[0], time_extended_ss, [max(segment_centers_min) + segment_duration / 60.0]])
    sleep_stage_extended = np.concatenate([[sleep_stage_numeric[0]], sleep_stage_extended, [sleep_stage_numeric[-1]]])
    
    ax4.fill_between(time_extended_ss, 0, sleep_stage_extended, step='pre', alpha=0.6, color='green')
    ax4.plot(segment_centers_min, sleep_stage_numeric, 'go', markersize=4)
    ax4.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax4.set_title('Sleep Stages', fontsize=14, fontweight='bold')
    ax4.set_ylim(-0.5, 5.5)
    ax4.set_yticks(range(6))
    ax4.set_yticklabels(['S0', 'S1', 'S2', 'S3', 'S4', 'REM'])
    ax4.grid(True, alpha=0.3)
    
    # Add text annotations for sleep stages
    for i, (time_center, stage_name) in enumerate(zip(segment_centers_min, sleep_stage_names)):
        if i % 5 == 0:  # Show every 5th label to avoid clutter
            ax4.text(time_center, sleep_stage_numeric[i] + 0.3, stage_name, 
                    ha='center', fontsize=8, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Bag Summary - Record ID: {record_id}")
    print(f"{'='*80}")
    print(f"CVD Label: {int(cvd_label)} ({'CVD' if cvd_label == 1 else 'No CVD'})")
    print(f"Number of valid segments: {n_valid}")
    print(f"Total duration: {n_valid * segment_duration / 60.0:.1f} minutes")
    print(f"Apnea segments: {int(np.sum(apnea_valid))} ({100 * np.mean(apnea_valid):.1f}%)")
    print(f"\nSleep Stage Distribution:")
    for stage_num, stage_name in SLEEP_STAGE_NAMES.items():
        count = np.sum(sleep_stages_valid == stage_num)
        if count > 0:
            print(f"  {stage_name}: {count} segments ({100 * count / n_valid:.1f}%)")
    print(f"{'='*80}\n")


def visualize_bags_interactive(
    train_dataset,
    ecg_fs: int = 200,
    ppg_fs: int = 100,
    segment_duration: int = 30,
    figsize: tuple = (20, 12),
    save_path: Optional[str] = None,
    start_idx: int = 0
):
    """
    Create an interactive visualization with navigation buttons to scroll through training bags.
    
    Creates a 4-row figure showing:
    1. Full ECG signal (30 minutes, concatenated segments)
    2. Full PPG signal (30 minutes, concatenated segments)
    3. Apnea labels (binary, one per segment)
    4. Sleep stages (categorical, one per segment)
    
    Features:
    - Interactive matplotlib plot (zoomable on each row)
    - Next/Previous buttons to navigate through bags
    - Can save as interactive HTML file
    
    Args:
        train_dataset: PyTorch Dataset for training data
        ecg_fs: ECG sampling rate in Hz (default: 200)
        ppg_fs: PPG sampling rate in Hz (default: 100)
        segment_duration: Duration of each segment in seconds (default: 30)
        figsize: Figure size tuple (width, height) in inches (default: (20, 12))
        save_path: Optional path to save the interactive figure as HTML. If None, displays interactively
        start_idx: Starting bag index (default: 0)
    """
    # Collect all bags from the dataset
    print("Collecting all bags from dataset...")
    all_bags = []
    for idx in range(len(train_dataset)):
        try:
            input_dict, cvd_label = train_dataset[idx]
            all_bags.append({
                'input_dict': input_dict,
                'cvd_label': cvd_label,
                'idx': idx
            })
        except Exception as e:
            print(f"Warning: Could not load bag {idx}: {e}")
            continue
    
    if len(all_bags) == 0:
        print("Error: No bags could be loaded from dataset")
        return
    
    print(f"Loaded {len(all_bags)} bags")
    
    # Current bag index
    current_idx = [start_idx % len(all_bags)]
    
    def extract_bag_data(bag_data):
        """Extract and process bag data for visualization."""
        input_dict = bag_data['input_dict']
        cvd_label = bag_data['cvd_label']
        
        # Extract data for the bag
        ecg = input_dict['ecg'].cpu().numpy()  # Shape: (max_instances, L_ecg)
        ppg = input_dict['ppg'].cpu().numpy()  # Shape: (max_instances, L_ppg)
        apnea_labels = input_dict['apnea_label'].cpu().numpy()  # Shape: (max_instances,)
        sleep_stages = input_dict['sleep_stage'].cpu().numpy()  # Shape: (max_instances,)
        mask = input_dict['mask'].cpu().numpy()  # Shape: (max_instances,)
        record_id = input_dict['record_id']
        cvd_label_val = cvd_label.item()
        
        # Get only valid segments (using mask)
        valid_indices = np.where(mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return None
        
        ecg_valid = ecg[valid_indices]
        ppg_valid = ppg[valid_indices]
        apnea_valid = apnea_labels[valid_indices]
        sleep_stages_valid = sleep_stages[valid_indices]
        
        # Concatenate segments to form full 30-minute signals
        ecg_full = np.concatenate(ecg_valid, axis=0)
        ppg_full = np.concatenate(ppg_valid, axis=0)
        
        # Create time axes
        ecg_time = np.arange(len(ecg_full)) / ecg_fs  # Time in seconds
        ppg_time = np.arange(len(ppg_full)) / ppg_fs  # Time in seconds
        
        # Convert time to minutes for x-axis
        ecg_time_min = ecg_time / 60.0
        ppg_time_min = ppg_time / 60.0
        
        # Create segment time axis for labels (center of each segment)
        segment_centers = np.arange(n_valid) * segment_duration + segment_duration / 2
        segment_centers_min = segment_centers / 60.0
        
        # Convert sleep stage integers to names
        sleep_stage_names = [SLEEP_STAGE_NAMES.get(int(stage), 'UNKNOWN') for stage in sleep_stages_valid]
        
        return {
            'ecg_time_min': ecg_time_min,
            'ecg_full': ecg_full,
            'ppg_time_min': ppg_time_min,
            'ppg_full': ppg_full,
            'segment_centers_min': segment_centers_min,
            'apnea_valid': apnea_valid,
            'sleep_stages_valid': sleep_stages_valid,
            'sleep_stage_names': sleep_stage_names,
            'record_id': record_id,
            'cvd_label_val': cvd_label_val,
            'n_valid': n_valid
        }
    
    def plot_bag(ax_list, bag_data_dict):
        """Plot a bag's data on the given axes."""
        # Clear all axes
        for ax in ax_list:
            ax.clear()
        
        # Extract data
        ecg_time_min = bag_data_dict['ecg_time_min']
        ecg_full = bag_data_dict['ecg_full']
        ppg_time_min = bag_data_dict['ppg_time_min']
        ppg_full = bag_data_dict['ppg_full']
        segment_centers_min = bag_data_dict['segment_centers_min']
        apnea_valid = bag_data_dict['apnea_valid']
        sleep_stages_valid = bag_data_dict['sleep_stages_valid']
        sleep_stage_names = bag_data_dict['sleep_stage_names']
        record_id = bag_data_dict['record_id']
        cvd_label_val = bag_data_dict['cvd_label_val']
        n_valid = bag_data_dict['n_valid']
        
        # Update title
        fig.suptitle(f'Bag {current_idx[0] + 1}/{len(all_bags)} - Record ID: {record_id}, '
                     f'CVD Label: {int(cvd_label_val)}, Valid Segments: {n_valid}',
                     fontsize=16, fontweight='bold')
        
        # Plot 1: ECG Signal
        ax1 = ax_list[0]
        ax1.plot(ecg_time_min, ecg_full, 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('ECG Amplitude', fontsize=12, fontweight='bold')
        ax1.set_title('ECG Signal (30 minutes)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(ecg_time_min))
        
        # Plot 2: PPG Signal
        ax2 = ax_list[1]
        ax2.plot(ppg_time_min, ppg_full, 'r-', linewidth=0.5, alpha=0.7)
        ax2.set_ylabel('PPG Amplitude', fontsize=12, fontweight='bold')
        ax2.set_title('PPG Signal (30 minutes)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(ppg_time_min))
        
        # Plot 3: Apnea Labels
        ax3 = ax_list[2]
        # Create a step plot for apnea labels
        apnea_extended = np.repeat(apnea_valid, 2)
        time_extended = np.repeat(segment_centers_min, 2)
        time_extended[1::2] += segment_duration / 60.0 / 2
        time_extended = np.concatenate([[0], time_extended, [max(segment_centers_min) + segment_duration / 60.0]])
        apnea_extended = np.concatenate([[apnea_valid[0]], apnea_extended, [apnea_valid[-1]]])
        
        ax3.fill_between(time_extended, 0, apnea_extended, step='pre', alpha=0.6, color='red', label='Apnea')
        ax3.plot(segment_centers_min, apnea_valid, 'ro', markersize=4)
        ax3.set_ylabel('Apnea Label', fontsize=12, fontweight='bold')
        ax3.set_title('Apnea Labels (1 = Apnea, 0 = No Apnea)', fontsize=14, fontweight='bold')
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['No Apnea', 'Apnea'])
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Sleep Stages
        ax4 = ax_list[3]
        # Create a step plot for sleep stages
        sleep_stage_numeric = sleep_stages_valid.astype(float)
        sleep_stage_extended = np.repeat(sleep_stage_numeric, 2)
        time_extended_ss = np.repeat(segment_centers_min, 2)
        time_extended_ss[1::2] += segment_duration / 60.0 / 2
        time_extended_ss = np.concatenate([[0], time_extended_ss, [max(segment_centers_min) + segment_duration / 60.0]])
        sleep_stage_extended = np.concatenate([[sleep_stage_numeric[0]], sleep_stage_extended, [sleep_stage_numeric[-1]]])
        
        ax4.fill_between(time_extended_ss, 0, sleep_stage_extended, step='pre', alpha=0.6, color='green')
        ax4.plot(segment_centers_min, sleep_stage_numeric, 'go', markersize=4)
        ax4.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
        ax4.set_title('Sleep Stages', fontsize=14, fontweight='bold')
        ax4.set_ylim(-0.5, 5.5)
        ax4.set_yticks(range(6))
        ax4.set_yticklabels(['S0', 'S1', 'S2', 'S3', 'S4', 'REM'])
        ax4.grid(True, alpha=0.3)
        
        # Add text annotations for sleep stages
        for i, (time_center, stage_name) in enumerate(zip(segment_centers_min, sleep_stage_names)):
            if i % 5 == 0:  # Show every 5th label to avoid clutter
                ax4.text(time_center, sleep_stage_numeric[i] + 0.3, stage_name, 
                        ha='center', fontsize=8, rotation=45)
        
        # Adjust layout but preserve space for buttons
        fig.subplots_adjust(bottom=0.08)
        plt.draw()
    
    def next_bag(event):
        """Navigate to next bag."""
        current_idx[0] = (current_idx[0] + 1) % len(all_bags)
        bag_data = all_bags[current_idx[0]]
        bag_data_dict = extract_bag_data(bag_data)
        if bag_data_dict is not None:
            plot_bag(axes, bag_data_dict)
        else:
            print(f"Warning: Bag {current_idx[0]} has no valid segments, skipping...")
            next_bag(event)  # Skip to next
    
    def prev_bag(event):
        """Navigate to previous bag."""
        current_idx[0] = (current_idx[0] - 1) % len(all_bags)
        bag_data = all_bags[current_idx[0]]
        bag_data_dict = extract_bag_data(bag_data)
        if bag_data_dict is not None:
            plot_bag(axes, bag_data_dict)
        else:
            print(f"Warning: Bag {current_idx[0]} has no valid segments, skipping...")
            prev_bag(event)  # Skip to previous
    
    # Create figure with 4 subplots, leaving space at bottom for buttons
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)
    fig.subplots_adjust(bottom=0.08)  # Leave space for navigation buttons
    
    # Enable interactive mode
    plt.ion()
    
    # Create navigation buttons at the bottom
    ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
    ax_next = plt.axes([0.21, 0.02, 0.1, 0.04])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    
    btn_prev.on_clicked(prev_bag)
    btn_next.on_clicked(next_bag)
    
    # Plot initial bag
    initial_bag = all_bags[current_idx[0]]
    initial_bag_data = extract_bag_data(initial_bag)
    if initial_bag_data is not None:
        plot_bag(axes, initial_bag_data)
    else:
        print(f"Warning: Initial bag {current_idx[0]} has no valid segments, moving to next...")
        current_idx[0] = (current_idx[0] + 1) % len(all_bags)
        initial_bag = all_bags[current_idx[0]]
        initial_bag_data = extract_bag_data(initial_bag)
        if initial_bag_data is not None:
            plot_bag(axes, initial_bag_data)
    
    # Save current state as image if requested (before showing interactive window)
    if save_path:
        if save_path.endswith('.html'):
            # For HTML, save as PNG instead (interactive HTML requires additional libraries)
            save_path_png = save_path.replace('.html', '.png')
            plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
            print(f"Note: Interactive HTML not supported. Saved current view as {save_path_png}")
            print(f"Use the interactive window to navigate and zoom. The window supports full interactivity.")
        else:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved current view to {save_path}")
    
    # Show the plot (interactive - allows zooming and panning on each subplot)
    plt.show(block=True)

