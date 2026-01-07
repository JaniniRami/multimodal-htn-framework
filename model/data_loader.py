"""
Data loading utilities for loading bag files from patient folders.
"""

from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np


def get_folder_name(entry):
    """
    Get the folder name for a given entry.
    
    Args:
        entry: Dictionary entry with 'patient_id' and 'follow_up'
    
    Returns:
        str: Folder name (patient_id_followup or just patient_id for baseline)
    """
    patient_id = entry.get('patient_id')
    follow_up = entry.get('follow_up')
    
    if follow_up is None:
        return patient_id
    else:
        return f"{patient_id}_{follow_up}"


def load_bags_from_folder(folder_path):
    """
    Load all bag files from a folder.
    
    Args:
        folder_path: Path to the patient folder
    
    Returns:
        list: List of dictionaries, each containing bag data and metadata
    """
    bags = []
    
    if not folder_path.exists():
        return bags
    
    # Find all h5 files in the folder
    h5_files = sorted(folder_path.glob('*.h5'))
    
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                # Load sleep_stages (vlen=str dtype, h5py handles decoding)
                sleep_stages_data = f['sleep_stages'][:]
                sleep_stages = [s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                               for s in sleep_stages_data]
                
                bag_data = {
                    'file_path': str(h5_file),
                    'ecg_segments': np.array(f['ecg_segments']),
                    'ppg_segments': np.array(f['ppg_segments']),
                    'apnea_labels': np.array(f['apnea_labels']),
                    'sleep_stages': sleep_stages
                }
                
                # Load HRV features if they exist
                if 'ecg_hrv' in f:
                    bag_data['ecg_hrv'] = {
                        'hrv_hf': float(f['ecg_hrv']['hrv_hf'][()]),
                        'hrv_lf': float(f['ecg_hrv']['hrv_lf'][()])
                    }
                else:
                    bag_data['ecg_hrv'] = None
                
                if 'ppg_hrv' in f:
                    bag_data['ppg_hrv'] = {
                        'hrv_hf': float(f['ppg_hrv']['hrv_hf'][()]),
                        'hrv_lf': float(f['ppg_hrv']['hrv_lf'][()])
                    }
                else:
                    bag_data['ppg_hrv'] = None
                
                bags.append(bag_data)
        except Exception as e:
            print(f"Warning: Could not load {h5_file}: {e}")
    
    return bags


def load_data_from_entries(entries, data_dir):
    """
    Load all bags for a list of entries.
    
    Args:
        entries: List of entry dictionaries
        data_dir: Base directory containing patient folders
    
    Returns:
        dict: Dictionary with 'bags', 'baseline_files', 'followup_files', 'total_bags'
    """
    # Group entries by folder to avoid loading duplicates
    folder_to_entries = defaultdict(list)
    for entry in entries:
        folder_name = get_folder_name(entry)
        folder_to_entries[folder_name].append(entry)
    
    all_bags = []
    baseline_count = 0
    followup_count = 0
    
    for folder_name, folder_entries in folder_to_entries.items():
        folder_path = data_dir / folder_name
        
        # Load bags from this folder
        bags = load_bags_from_folder(folder_path)
        all_bags.extend(bags)
        
        # Count baseline vs followup files
        for entry in folder_entries:
            if entry.get('follow_up') is None:
                baseline_count += 1
            else:
                followup_count += 1
    
    return {
        'bags': all_bags,
        'baseline_files': baseline_count,
        'followup_files': followup_count,
        'total_bags': len(all_bags)
    }

