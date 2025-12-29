"""
Extract labels from PDF files for valid EDF files.

This script processes the file_mapping.json to extract labels (AHI, apnea checkboxes,
CVD checkboxes) from PDF files for all valid entries.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pdfplumber
import cv2
import numpy as np
from easyocr import Reader
from ocr import checkbox_analysis

# Constants from the old code
CARDIOVASCULAR_AREA = 403767
SLEEP_DISORDER_AREA = 308448


def find_AHI(reader, pdf_path):
    """
    Extract AHI (Apnea Hypopnea Index) value from PDF page 2.
    
    Args:
        reader: EasyOCR Reader instance
        pdf_path: Path to PDF file
    
    Returns:
        AHI value as string, or "100000" if not found
    """
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) < 3:
            return "100000"
        
        page = pdf.pages[2]
        pil_image = page.to_image(resolution=300).original
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

        binary = cv2.adaptiveThreshold(
            cv2_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        
        text = reader.readtext(binary)  
        text = " ".join([result[1] for result in text])
        match = re.search(r"Apnea Hypopnea Index \(AHI\):.*?(\d+\.\d+)", text)
        ahi_value = match.group(1) if match else "100000"
        
        return ahi_value


def extract_labels_from_pdf(reader, pdf_path):
    """
    Extract all labels from a PDF file.
    
    Args:
        reader: EasyOCR Reader instance
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with extracted labels
    """
    # Extract AHI
    AHI = find_AHI(reader, pdf_path)
    
    # Extract checkboxes
    apnea_checkboxes = checkbox_analysis(pdf_path, SLEEP_DISORDER_AREA, reader)
    cvd_checkboxes = checkbox_analysis(pdf_path, CARDIOVASCULAR_AREA, reader)
    
    # Determine has_apnea
    if AHI == "100000":
        has_apnea = "idk"
    else:
        has_apnea = "yes" if float(AHI) > 5 else "no"
    
    # Determine has_cvd
    if len(cvd_checkboxes) == 0:
        has_cvd = "no"
    elif (has_apnea == 'idk' or has_apnea == 'no') and len(cvd_checkboxes) > 0:
        has_cvd = "idk"
    else:
        has_cvd = "yes"
    
    return {
        'AHI': AHI,
        'apnea_checkboxes': apnea_checkboxes,
        'cvd_checkboxes': cvd_checkboxes,
        'has_apnea': has_apnea,
        'has_cvd': has_cvd
    }


def process_valid_files(file_mapping_path, output_csv_path):
    """
    Process all valid files from file_mapping.json and extract labels.
    
    Args:
        file_mapping_path: Path to file_mapping.json
        output_csv_path: Path to save output CSV
    """
    # Load file mapping
    print(f"Loading file mapping from {file_mapping_path}...")
    with open(file_mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Filter for valid entries
    valid_entries = [entry for entry in mapping_data['valid_mappings'] if entry.get('is_valid', False)]
    print(f"Found {len(valid_entries)} valid files to process")
    
    # Initialize EasyOCR reader
    print("Initializing EasyOCR reader...")
    reader = Reader(['en'], gpu=True)
    
    # Prepare output data
    output_data = {
        'edf_path': [],
        'annotation_file_path': [],
        'meta_data_file_path': [],
        'year': [],
        'patient_id': [],
        'follow_up': [],
        'apnea_checkboxes': [],
        'cvd_checkboxes': [],
        'AHI': [],
        'has_apnea': [],
        'has_cvd': [],
        'ecg_usable_percentage': [],
        'ppg_usable_percentage': []
    }
    
    # Process each valid file
    total_files = len(valid_entries)
    for idx, entry in enumerate(valid_entries, 1):
        print(f"\n[{idx}/{total_files}] Processing: {entry['edf_file']}")
        print(f"  EDF: {entry['edf_path']}")
        print(f"  PDF: {entry['pdf_path']}")
        
        # Extract labels from PDF
        try:
            labels = extract_labels_from_pdf(reader, entry['pdf_path'])
            
            # Add to output data
            output_data['edf_path'].append(entry['edf_path'])
            output_data['annotation_file_path'].append(entry.get('txt_path', ''))
            output_data['meta_data_file_path'].append(entry['pdf_path'])
            output_data['year'].append(entry['year'])
            output_data['patient_id'].append(entry['patient_id'])
            output_data['follow_up'].append(entry['follow_up'])
            output_data['apnea_checkboxes'].append("; ".join(labels['apnea_checkboxes']))
            output_data['cvd_checkboxes'].append("; ".join(labels['cvd_checkboxes']))
            output_data['AHI'].append(labels['AHI'])
            output_data['has_apnea'].append(labels['has_apnea'])
            output_data['has_cvd'].append(labels['has_cvd'])
            output_data['ecg_usable_percentage'].append(
                entry.get('ecg_validation', {}).get('usable_percentage', 0.0)
            )
            output_data['ppg_usable_percentage'].append(
                entry.get('ppg_validation', {}).get('usable_percentage', 0.0)
            )
            
            print(f"  AHI: {labels['AHI']}")
            print(f"  Has apnea: {labels['has_apnea']}")
            print(f"  Has CVD: {labels['has_cvd']}")
            print(f"  Apnea checkboxes: {len(labels['apnea_checkboxes'])}")
            print(f"  CVD checkboxes: {len(labels['cvd_checkboxes'])}")
            
        except Exception as e:
            print(f"  ERROR processing file: {e}")
            # Still add entry with error values
            output_data['edf_path'].append(entry['edf_path'])
            output_data['annotation_file_path'].append(entry.get('txt_path', ''))
            output_data['meta_data_file_path'].append(entry['pdf_path'])
            output_data['year'].append(entry['year'])
            output_data['patient_id'].append(entry['patient_id'])
            output_data['follow_up'].append(entry['follow_up'])
            output_data['apnea_checkboxes'].append("")
            output_data['cvd_checkboxes'].append("")
            output_data['AHI'].append("ERROR")
            output_data['has_apnea'].append("ERROR")
            output_data['has_cvd'].append("ERROR")
            output_data['ecg_usable_percentage'].append(
                entry.get('ecg_validation', {}).get('usable_percentage', 0.0)
            )
            output_data['ppg_usable_percentage'].append(
                entry.get('ppg_validation', {}).get('usable_percentage', 0.0)
            )
    
    # Create DataFrame and save to CSV
    print(f"\nSaving results to {output_csv_path}...")
    df = pd.DataFrame(output_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} entries to {output_csv_path}")
    
    return df


def main():
    """Main function to run label extraction."""
    # Define paths
    base_path = Path(__file__).parent
    file_mapping_path = base_path / 'output' / 'file_mapping.json'
    output_csv_path = base_path / 'output' / 'valid_files_labels.csv'
    
    # Check if file mapping exists
    if not file_mapping_path.exists():
        print(f"Error: File mapping not found at {file_mapping_path}")
        print("Please run validate_file_mapping.py first to generate the mapping.")
        return
    
    # Process valid files
    df = process_valid_files(file_mapping_path, output_csv_path)
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total valid files processed: {len(df)}")
    print(f"\nApnea status:")
    print(f"  Has apnea (yes): {len(df[df['has_apnea'] == 'yes'])}")
    print(f"  No apnea (no): {len(df[df['has_apnea'] == 'no'])}")
    print(f"  Unknown (idk): {len(df[df['has_apnea'] == 'idk'])}")
    print(f"\nCVD status:")
    print(f"  Has CVD (yes): {len(df[df['has_cvd'] == 'yes'])}")
    print(f"  No CVD (no): {len(df[df['has_cvd'] == 'no'])}")
    print(f"  Unknown (idk): {len(df[df['has_cvd'] == 'idk'])}")
    print("="*80)


if __name__ == '__main__':
    main()

