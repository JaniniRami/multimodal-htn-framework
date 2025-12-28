"""
Filter file_mapping.json to include only patients with apnea (has_apnea == "yes")
and generate statistics.

This script:
1. Reads valid_files_labels.csv to get apnea status
2. Filters file_mapping.json to keep only entries with has_apnea == "yes"
3. Adds label information to the filtered entries
4. Generates statistics about apnea patients
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def load_labels_csv(csv_path):
    """Load the labels CSV file."""
    labels_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edf_path = row['edf_path']
            labels_dict[edf_path] = {
                'has_apnea': row['has_apnea'],
                'has_cvd': row['has_cvd'],
                'AHI': row['AHI'],
                'apnea_checkboxes': row['apnea_checkboxes'],
                'cvd_checkboxes': row['cvd_checkboxes'],
                'patient_id': row['patient_id'],
                'follow_up': row['follow_up'] if row['follow_up'] else None,
                'year': row['year']
            }
    return labels_dict


def filter_file_mapping(file_mapping_path, labels_dict, output_path):
    """
    Filter file_mapping.json to include only entries with has_apnea == "yes".
    
    Args:
        file_mapping_path: Path to original file_mapping.json
        labels_dict: Dictionary of labels keyed by edf_path
        output_path: Path to save filtered file_mapping.json
    
    Returns:
        Filtered mapping dictionary
    """
    # Load original file mapping
    with open(file_mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Filter valid_mappings to only include entries with has_apnea == "yes"
    filtered_valid = []
    filtered_invalid = []
    
    for entry in mapping_data['valid_mappings']:
        edf_path = entry['edf_path']
        if edf_path in labels_dict:
            label_info = labels_dict[edf_path]
            if label_info['has_apnea'] == 'yes':
                # Add label information to entry
                entry['has_apnea'] = label_info['has_apnea']
                entry['has_cvd'] = label_info['has_cvd']
                entry['AHI'] = label_info['AHI']
                entry['apnea_checkboxes'] = label_info['apnea_checkboxes']
                entry['cvd_checkboxes'] = label_info['cvd_checkboxes']
                filtered_valid.append(entry)
    
    # Also check invalid_mappings (though they shouldn't have apnea if they're invalid)
    for entry in mapping_data.get('invalid_mappings', []):
        edf_path = entry.get('edf_path')
        if edf_path and edf_path in labels_dict:
            label_info = labels_dict[edf_path]
            if label_info['has_apnea'] == 'yes':
                entry['has_apnea'] = label_info['has_apnea']
                entry['has_cvd'] = label_info['has_cvd']
                entry['AHI'] = label_info['AHI']
                entry['apnea_checkboxes'] = label_info['apnea_checkboxes']
                entry['cvd_checkboxes'] = label_info['cvd_checkboxes']
                filtered_invalid.append(entry)
    
    # Create filtered mapping
    filtered_mapping = {
        'valid_mappings': filtered_valid,
        'invalid_mappings': filtered_invalid,
        'summary': {
            'total_apnea_patients': len(filtered_valid),
            'total_files': len(filtered_valid) + len(filtered_invalid),
            'original_total': mapping_data['summary']['total_embla_files'],
            'original_valid': mapping_data['summary']['valid_count'],
            'original_invalid': mapping_data['summary']['invalid_count']
        }
    }
    
    # Save filtered mapping
    with open(output_path, 'w') as f:
        json.dump(filtered_mapping, f, indent=2)
    
    return filtered_mapping


def calculate_statistics(labels_dict):
    """
    Calculate statistics about apnea patients.
    
    Args:
        labels_dict: Dictionary of labels keyed by edf_path
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_files': 0,
        'unique_patient_ids': set(),
        'with_cvd': {
            'count': 0,
            'files': [],
            'patient_ids': set()
        },
        'without_cvd': {
            'count': 0,
            'files': [],
            'patient_ids': set()
        },
        'cvd_unknown': {
            'count': 0,
            'files': [],
            'patient_ids': set()
        },
        'ahi_distribution': {
            'low': 0,      # 0-5
            'mild': 0,     # 5-15
            'moderate': 0, # 15-30
            'severe': 0   # >30
        }
    }
    
    for edf_path, label_info in labels_dict.items():
        if label_info['has_apnea'] == 'yes':
            stats['total_files'] += 1
            patient_id = label_info['patient_id']
            if patient_id:
                stats['unique_patient_ids'].add(patient_id)
            
            # Get AHI value
            ahi_str = str(label_info['AHI'])
            try:
                if ahi_str and ahi_str != 'ERROR' and ahi_str != 'nan':
                    ahi = float(ahi_str)
                    if ahi <= 5:
                        stats['ahi_distribution']['low'] += 1
                    elif ahi <= 15:
                        stats['ahi_distribution']['mild'] += 1
                    elif ahi <= 30:
                        stats['ahi_distribution']['moderate'] += 1
                    else:
                        stats['ahi_distribution']['severe'] += 1
            except (ValueError, TypeError):
                pass
            
            # Categorize by CVD status
            patient_id = label_info['patient_id']
            if label_info['has_cvd'] == 'yes':
                stats['with_cvd']['count'] += 1
                stats['with_cvd']['files'].append({
                    'edf_path': edf_path,
                    'patient_id': patient_id,
                    'follow_up': label_info['follow_up'],
                    'AHI': label_info['AHI'],
                    'cvd_checkboxes': label_info['cvd_checkboxes']
                })
                stats['with_cvd']['patient_ids'].add(patient_id)
            elif label_info['has_cvd'] == 'no':
                stats['without_cvd']['count'] += 1
                stats['without_cvd']['files'].append({
                    'edf_path': edf_path,
                    'patient_id': patient_id,
                    'follow_up': label_info['follow_up'],
                    'AHI': label_info['AHI']
                })
                stats['without_cvd']['patient_ids'].add(patient_id)
            else:  # 'idk' or other
                stats['cvd_unknown']['count'] += 1
                stats['cvd_unknown']['files'].append({
                    'edf_path': edf_path,
                    'patient_id': patient_id,
                    'follow_up': label_info['follow_up'],
                    'AHI': label_info['AHI'],
                    'cvd_checkboxes': label_info['cvd_checkboxes']
                })
                stats['cvd_unknown']['patient_ids'].add(patient_id)
    
    # Convert sets to counts
    stats['total_unique_patients'] = len(stats['unique_patient_ids'])
    stats['with_cvd']['unique_patients'] = len(stats['with_cvd']['patient_ids'])
    stats['without_cvd']['unique_patients'] = len(stats['without_cvd']['patient_ids'])
    stats['cvd_unknown']['unique_patients'] = len(stats['cvd_unknown']['patient_ids'])
    
    return stats


def print_statistics(stats, filtered_mapping):
    """Print detailed statistics."""
    print("\n" + "="*80)
    print("APNEA PATIENTS STATISTICS")
    print("="*80)
    
    print(f"\nOVERVIEW:")
    print(f"  Total files with apnea: {stats['total_files']}")
    print(f"  Total unique apnea patients: {stats['total_unique_patients']}")
    print(f"  Original total files: {filtered_mapping['summary']['original_total']}")
    print(f"  Filtered to apnea patients: {filtered_mapping['summary']['total_apnea_patients']}")
    print(f"  Reduction: {filtered_mapping['summary']['original_total'] - filtered_mapping['summary']['total_apnea_patients']} files removed")
    
    print(f"\nCVD STATUS:")
    print(f"  Patients WITH CVD:")
    print(f"    Total files: {stats['with_cvd']['count']}")
    print(f"    Unique patients: {stats['with_cvd']['unique_patients']}")
    
    print(f"  Patients WITHOUT CVD:")
    print(f"    Total files: {stats['without_cvd']['count']}")
    print(f"    Unique patients: {stats['without_cvd']['unique_patients']}")
    
    print(f"  Patients with UNKNOWN CVD status:")
    print(f"    Total files: {stats['cvd_unknown']['count']}")
    print(f"    Unique patients: {stats['cvd_unknown']['unique_patients']}")
    

def main():
    """Main function to filter file mapping and generate statistics."""
    # Define paths
    base_path = Path(__file__).parent
    labels_csv_path = base_path / 'output' / 'valid_files_labels.csv'
    file_mapping_path = base_path / 'output' / 'file_mapping.json'
    output_path = base_path / 'output' / 'file_mapping_apnea_only.json'
    
    # Check if files exist
    if not labels_csv_path.exists():
        print(f"Error: Labels CSV not found at {labels_csv_path}")
        return
    
    if not file_mapping_path.exists():
        print(f"Error: File mapping not found at {file_mapping_path}")
        return
    
    print("Loading labels CSV...")
    labels_dict = load_labels_csv(labels_csv_path)
    print(f"Loaded {len(labels_dict)} entries from labels CSV")
    
    # Filter for apnea patients only
    apnea_labels = {k: v for k, v in labels_dict.items() if v['has_apnea'] == 'yes'}
    print(f"Found {len(apnea_labels)} files with apnea (has_apnea == 'yes')")
    
    print("\nFiltering file_mapping.json...")
    filtered_mapping = filter_file_mapping(file_mapping_path, apnea_labels, output_path)
    print(f"Filtered mapping saved to: {output_path}")
    print(f"  Valid mappings: {len(filtered_mapping['valid_mappings'])}")
    print(f"  Invalid mappings: {len(filtered_mapping['invalid_mappings'])}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(apnea_labels)
    
    # Print statistics
    print_statistics(stats, filtered_mapping)
    
    return filtered_mapping, stats


if __name__ == '__main__':
    filtered_mapping, stats = main()

