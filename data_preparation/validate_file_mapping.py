"""
Script to validate and map Embla EDF files to corresponding Meta data PDF files.

Structure:
- Embla: data/Embla/{year}/{patient_id (number)}/{patient_id (number).edf}
- Meta data: data/Meta data/{patient_id}/{Baseline.pdf or Follow Up {number}.pdf}

Mapping logic:
- patient_id (no number) -> baseline.pdf or Baseline.pdf
- patient_id (1) -> Follow Up 1.pdf
- patient_id (2) -> Follow Up 2.pdf
- etc.
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from signal_validation import validate_edf_signal, validate_ppg_signal

def extract_patient_info(folder_name: str) -> Tuple[str, Optional[int]]:
    """
    Extract patient ID and follow-up number from folder name.
    
    Args:
        folder_name: Folder name like "49010001 (1)" or "49010113"
    
    Returns:
        Tuple of (patient_id, follow_up_number)
        follow_up_number is None for baseline (no number in folder name)
    """
    # Pattern to match: patient_id (number) or just patient_id
    match = re.match(r'^(\d+)(?:\s*\((\d+)\))?$', folder_name.strip())
    if match:
        patient_id = match.group(1)
        follow_up = int(match.group(2)) if match.group(2) else None
        return patient_id, follow_up
    return None, None


def find_pdf_in_meta_data(meta_data_path: Path, patient_id: str, follow_up: Optional[int]) -> Tuple[Optional[Path], Optional[str]]:
    """
    Find the corresponding PDF file in Meta data folder with fallback logic.
    
    Args:
        meta_data_path: Path to Meta data directory
        patient_id: Patient ID
        follow_up: Follow-up number (None for baseline)
    
    Returns:
        Tuple of (Path to PDF file if found, note string)
        If exact match found, note is None. Otherwise note indicates what it was linked to.
    """
    patient_folder = meta_data_path / patient_id
    
    if not patient_folder.exists():
        return None, None
    
    # List all PDF files in the patient folder
    pdf_files = list(patient_folder.glob('*.pdf'))
    
    if follow_up is None:
        # Look for baseline PDF (case-insensitive)
        for pdf in pdf_files:
            pdf_name_lower = pdf.name.lower()
            if 'baseline' in pdf_name_lower and 'follow' not in pdf_name_lower:
                return pdf, None  # Exact match, no note needed
    else:
        # First, try to find exact follow-up match
        patterns = [
            f'Follow Up {follow_up}',
            f'Follow up {follow_up}',
            f'follow up {follow_up}',
            f'FollowUp {follow_up}',
            f'followup {follow_up}',
        ]
        
        for pdf in pdf_files:
            pdf_name_lower = pdf.name.lower()
            for pattern in patterns:
                if pattern.lower() in pdf_name_lower:
                    return pdf, None  # Exact match, no note needed
        
        # If exact match not found, try Follow Up 1 as fallback (only if it exists)
        follow_up_1_patterns = [
            'Follow Up 1',
            'Follow up 1',
            'follow up 1',
            'FollowUp 1',
            'followup 1',
        ]
        
        for pdf in pdf_files:
            pdf_name_lower = pdf.name.lower()
            for pattern in follow_up_1_patterns:
                if pattern.lower() in pdf_name_lower:
                    return pdf, "Linked to follow up 1"
        
        # If Follow Up 1 not found, try Baseline as last resort
        for pdf in pdf_files:
            pdf_name_lower = pdf.name.lower()
            if 'baseline' in pdf_name_lower and 'follow' not in pdf_name_lower:
                return pdf, "Linked to baseline"
    
    return None, None


def scan_embla_files(embla_path: Path) -> List[Dict]:
    """
    Scan all Embla folders and collect file information.
    
    Args:
        embla_path: Path to Embla directory
    
    Returns:
        List of dictionaries with file information
    """
    embla_files = []
    
    # Iterate through year folders
    for year_folder in sorted(embla_path.iterdir()):
        if not year_folder.is_dir():
            continue
        
        year = year_folder.name
        
        # Iterate through patient folders
        for patient_folder in sorted(year_folder.iterdir()):
            if not patient_folder.is_dir():
                continue
            
            folder_name = patient_folder.name
            patient_id, follow_up = extract_patient_info(folder_name)
            
            if patient_id is None:
                print(f"Warning: Could not parse folder name: {folder_name}")
                continue
            
            # Find EDF file in the folder
            edf_files = list(patient_folder.glob('*.edf'))
            
            if not edf_files:
                print(f"Warning: No EDF file found in {patient_folder}")
                continue
            
            # Use the first EDF file (should be only one)
            edf_file = edf_files[0]
            
            # Check for xx.txt file
            txt_file = patient_folder / 'xx.txt'
            has_txt_file = txt_file.exists()
            
            if not has_txt_file:
                print(f"Warning: No xx.txt file found in {patient_folder}")
            
            embla_files.append({
                'year': year,
                'folder_name': folder_name,
                'patient_id': patient_id,
                'follow_up': follow_up,
                'edf_path': str(edf_file),
                'edf_file': edf_file.name,
                'has_edf': True,
                'has_txt': has_txt_file,
                'txt_path': str(txt_file) if has_txt_file else None,
                'is_complete': has_txt_file  # Both EDF and TXT are required
            })
    
    return embla_files


def create_file_mapping(embla_path: Path, meta_data_path: Path) -> Dict:
    """
    Create a mapping dictionary between Embla files and Meta data PDFs.
    
    Args:
        embla_path: Path to Embla directory
        meta_data_path: Path to Meta data directory
    
    Returns:
        Dictionary with mapping information
    """
    embla_files = scan_embla_files(embla_path)
    
    mapping = {
        'valid_mappings': [],
        'invalid_mappings': [],
        'summary': {
            'total_embla_files': len(embla_files),
            'valid_count': 0,
            'invalid_count': 0,
            'complete_files': 0,  # Files with both .edf and xx.txt
            'incomplete_files': 0,  # Files missing xx.txt
            'ecg_valid_count': 0,  # ECG signals that are valid
            'ecg_invalid_count': 0,  # ECG signals that are invalid
            'ppg_valid_count': 0,  # PPG signals that are valid
            'ppg_invalid_count': 0,  # PPG signals that are invalid
            'ecg_channel_found_count': 0,  # Files with ECG/EKG channel
            'ecg_channel_missing_count': 0,  # Files without ECG/EKG channel
            'ppg_channel_found_count': 0,  # Files with PPG channel
            'ppg_channel_missing_count': 0  # Files without PPG channel
        }
    }
    
    total_files = len(embla_files)
    print(f"\nProcessing {total_files} files...")
    
    for idx, file_info in enumerate(embla_files, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"  Processing file {idx}/{total_files}...")
        
        pdf_path, note = find_pdf_in_meta_data(
            meta_data_path,
            file_info['patient_id'],
            file_info['follow_up']
        )
        
        # Validate ECG signal
        ecg_validation = validate_edf_signal(Path(file_info['edf_path']))
        
        # Validate PPG signal (reuse the raw object if available to avoid re-reading)
        # For now, we'll read it separately - could optimize later
        ppg_validation = validate_ppg_signal(Path(file_info['edf_path']))
        
        entry = {
            'year': file_info['year'],
            'embla_folder': file_info['folder_name'],
            'patient_id': file_info['patient_id'],
            'follow_up': file_info['follow_up'],
            'edf_path': file_info['edf_path'],
            'edf_file': file_info['edf_file'],
            'has_edf': file_info['has_edf'],
            'has_txt': file_info['has_txt'],
            'txt_path': file_info['txt_path'],
            'is_complete': file_info['is_complete'],
            'pdf_path': str(pdf_path) if pdf_path else None,
            'pdf_file': pdf_path.name if pdf_path else None,
            # ECG validation results
            'ecg_validation': {
                'has_ecg_channel': ecg_validation['has_ecg_channel'],
                'is_valid': ecg_validation['is_valid'],
                'usable_percentage': ecg_validation['usable_percentage'],
                'error': ecg_validation['error'],
                'statistics': ecg_validation['statistics']
            },
            # PPG validation results
            'ppg_validation': {
                'has_ppg_channel': ppg_validation['has_ppg_channel'],
                'is_valid': ppg_validation['is_valid'],
                'usable_percentage': ppg_validation['usable_percentage'],
                'error': ppg_validation['error'],
                'statistics': ppg_validation['statistics']
            },
            # Overall validity: PDF exists AND both ECG and PPG signals are valid (>50% usable)
            'is_valid': (pdf_path is not None and 
                        ecg_validation['is_valid'] and 
                        ppg_validation['is_valid'])
        }
        
        # Add note if present (when fallback was used)
        if note:
            entry['note'] = note
        
        # Track file completeness
        if file_info['is_complete']:
            mapping['summary']['complete_files'] += 1
        else:
            mapping['summary']['incomplete_files'] += 1
        
        # Track ECG validation
        if ecg_validation['has_ecg_channel']:
            mapping['summary']['ecg_channel_found_count'] += 1
        else:
            mapping['summary']['ecg_channel_missing_count'] += 1
        
        if ecg_validation['is_valid']:
            mapping['summary']['ecg_valid_count'] += 1
        else:
            mapping['summary']['ecg_invalid_count'] += 1
        
        # Track PPG validation
        if ppg_validation['has_ppg_channel']:
            mapping['summary']['ppg_channel_found_count'] += 1
        else:
            mapping['summary']['ppg_channel_missing_count'] += 1
        
        if ppg_validation['is_valid']:
            mapping['summary']['ppg_valid_count'] += 1
        else:
            mapping['summary']['ppg_invalid_count'] += 1
        
        # Overall validity: PDF exists AND both ECG and PPG are valid
        if pdf_path and ecg_validation['is_valid'] and ppg_validation['is_valid']:
            mapping['valid_mappings'].append(entry)
            mapping['summary']['valid_count'] += 1
        else:
            mapping['invalid_mappings'].append(entry)
            mapping['summary']['invalid_count'] += 1
    
    return mapping


def print_summary(mapping: Dict):
    """Print a summary of the mapping results."""
    summary = mapping['summary']
    
    print("\n" + "="*80)
    print("FILE MAPPING SUMMARY")
    print("="*80)
    print(f"Total Embla files found: {summary['total_embla_files']}")
    print(f"Valid mappings (PDF found AND ECG valid AND PPG valid): {summary['valid_count']}")
    print(f"Invalid mappings: {summary['invalid_count']}")
    print(f"Success rate: {summary['valid_count']/summary['total_embla_files']*100:.1f}%")
    print()
    print("FILE VERIFICATION:")
    print(f"  Complete files (.edf + xx.txt): {summary['complete_files']}")
    print(f"  Incomplete files (missing xx.txt): {summary['incomplete_files']}")
    print(f"  Completeness rate: {summary['complete_files']/summary['total_embla_files']*100:.1f}%")
    print()
    print("ECG SIGNAL VALIDATION:")
    print(f"  ECG/EKG channel found: {summary['ecg_channel_found_count']}")
    print(f"  ECG/EKG channel missing: {summary['ecg_channel_missing_count']}")
    print(f"  Valid ECG signals (>50% usable): {summary['ecg_valid_count']}")
    print(f"  Invalid ECG signals: {summary['ecg_invalid_count']}")
    if summary['ecg_channel_found_count'] > 0:
        ecg_valid_rate = summary['ecg_valid_count'] / summary['ecg_channel_found_count'] * 100
        print(f"  ECG validation rate: {ecg_valid_rate:.1f}%")
    print()
    print("PPG SIGNAL VALIDATION:")
    print(f"  PPG channel found: {summary['ppg_channel_found_count']}")
    print(f"  PPG channel missing: {summary['ppg_channel_missing_count']}")
    print(f"  Valid PPG signals (>50% usable): {summary['ppg_valid_count']}")
    print(f"  Invalid PPG signals: {summary['ppg_invalid_count']}")
    if summary['ppg_channel_found_count'] > 0:
        ppg_valid_rate = summary['ppg_valid_count'] / summary['ppg_channel_found_count'] * 100
        print(f"  PPG validation rate: {ppg_valid_rate:.1f}%")
    print("="*80)
    
    if mapping['invalid_mappings']:
        print("\nINVALID MAPPINGS (PDF not found):")
        print("-"*80)
        for entry in mapping['invalid_mappings'][:20]:  # Show first 20
            follow_up_str = f"Follow Up {entry['follow_up']}" if entry['follow_up'] else "Baseline"
            print(f"  {entry['embla_folder']} ({entry['year']}) -> Expected: {entry['patient_id']}/{follow_up_str}.pdf")
        if len(mapping['invalid_mappings']) > 20:
            print(f"  ... and {len(mapping['invalid_mappings']) - 20} more")


def save_mapping_to_json(mapping: Dict, output_path: Path):
    """Save the mapping dictionary to a JSON file."""
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved to: {output_path}")


def get_simple_mapping_dict(mapping: Dict) -> Dict[str, str]:
    """
    Get a simple dictionary mapping EDF file paths to PDF file paths.
    
    Args:
        mapping: The full mapping dictionary from create_file_mapping()
    
    Returns:
        Dictionary with EDF paths as keys and PDF paths as values
    """
    return {
        entry['edf_path']: entry['pdf_path']
        for entry in mapping['valid_mappings']
    }


def main():
    """Main function to run the validation script."""
    # Define paths
    base_path = Path(__file__).parent.parent
    embla_path = base_path / 'data' / 'Embla'
    meta_data_path = base_path / 'data' / 'Meta data'
    output_path = base_path / 'data_analysis' / 'output' / 'file_mapping.json'
    
    # Check if paths exist
    if not embla_path.exists():
        print(f"Error: Embla directory not found at {embla_path}")
        return
    
    if not meta_data_path.exists():
        print(f"Error: Meta data directory not found at {meta_data_path}")
        return
    
    print("Scanning Embla files and matching with Meta data PDFs...")
    print(f"Embla path: {embla_path}")
    print(f"Meta data path: {meta_data_path}")
    
    # Create mapping
    mapping = create_file_mapping(embla_path, meta_data_path)
    
    # Print summary
    print_summary(mapping)
    
    # Save to JSON
    save_mapping_to_json(mapping, output_path)
    
    # Create simple dictionary for easy access
    simple_mapping = get_simple_mapping_dict(mapping)
    
    print(f"\nSimple dictionary created with {len(simple_mapping)} valid mappings.")
    print("Access via: mapping['valid_mappings'] or mapping['invalid_mappings']")
    print("Or use get_simple_mapping_dict(mapping) for a simple {edf_path: pdf_path} dictionary")
    
    return mapping, simple_mapping


if __name__ == '__main__':
    mapping, simple_mapping = main()

