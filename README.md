# Multimodal Hypertension Framework

## Development Steps

### 1. Data Mapping & Signal Validation
**Script:** `data_analysis/validate_file_mapping.py`

* **Function:** Maps patient EDF files to their corresponding metadata PDF files (containing diagnosis and patient info).
* **Validation:** Performs statistical quality checks on ECG and PPG signals to ensure they are usable for modeling.
* **Output:** `data_analysis/output/file_mapping.json`

### 2. Label Extraction & Verification
**Script:** `data_analysis/validate_labels.py`

* **Function:** Uses OCR to extract key clinical metrics from the metadata PDFs, specifically:
    * AHI Index
    * Apnea Status
    * Cardiovascular Disease (CVD) Status
* **Process:** The extracted data undergoes a manual verification step to ensure accuracy.
* **Output:** `data_analysis/output/valid_files_labels.csv`

### 3. Patient Filtering (Apnea Cohort)
**Script:** `data_analysis/filter_apnea_patients.py`

* **Function:** Filters the dataset to retain only patients with confirmed sleep apnea (regardless of CVD status).
* **Output:** `data_analysis/output/file_mapping_apnea_only.json`