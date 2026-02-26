"""
Visualize attention weights from FUSEDNet model for a specific patient.

This script loads a trained model checkpoint and visualizes the attention weights
across time segments for a single patient's test data.
"""
import math
import seaborn as sns
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

# Add parent directory and model directory to path
project_root = Path(__file__).parent
model_dir = project_root / 'model'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(model_dir))

# Import directly from files using importlib to avoid package issues
import importlib.util

# Import config
config_spec = importlib.util.spec_from_file_location("config", model_dir / "config.py")
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
JSON_PATH = config_module.JSON_PATH
DATA_DIR = config_module.DATA_DIR

# Import data_loader
data_loader_spec = importlib.util.spec_from_file_location("data_loader", model_dir / "data_loader.py")
data_loader_module = importlib.util.module_from_spec(data_loader_spec)
data_loader_spec.loader.exec_module(data_loader_module)
load_data_from_entries = data_loader_module.load_data_from_entries

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
import json
from collections import defaultdict


def load_patient_data(patient_id: str, json_path: Path, data_dir: Path):
    """
    Load all data for a specific patient.
    
    Args:
        patient_id: Patient ID (e.g., '49010118')
        json_path: Path to file_mapping JSON
        data_dir: Path to data directory
    
    Returns:
        Tuple of (bags, entries, norm_coefficients)
    """
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find entries for this patient
    patient_entries = []
    for entry in data.get('valid_mappings', []):
        if entry.get('patient_id') == patient_id:
            patient_entries.append(entry)
    
    if not patient_entries:
        raise ValueError(f"Patient {patient_id} not found in the dataset")
    
    print(f"Found {len(patient_entries)} files for patient {patient_id}")
    
    # Load data
    patient_data = load_data_from_entries(patient_entries, data_dir)
    
    # Calculate normalization coefficients (we'll use a dummy training set for this)
    # In practice, you might want to use the same normalization from training
    # For now, we'll calculate from the patient's own data (not ideal but works for visualization)
    norm_coefficients = calculate_normalization_coefficients(patient_data['bags'])
    
    # Apply normalization
    patient_data['bags'] = apply_normalization(patient_data['bags'], norm_coefficients)
    
    return patient_data['bags'], patient_entries, norm_coefficients


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    # Initialize model with same architecture as training
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel if needed
    if 'module.' in list(state_dict.keys())[0]:
        # Checkpoint was saved with DataParallel
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.module  # Unwrap for easier use
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(device)
    
    print(f"Model loaded from {checkpoint_path}")
    return model

def visualize_attention(
    model: nn.Module,
    patient_id: str,
    bags: list,
    entries: list,
    data_dir: Path,
    device: torch.device,
    output_path: Path
):
    # 1. Create dataset and loader
    dataset, _ = create_datasets_from_splits(
        train_bags=bags,
        test_bags=[],
        train_entries=entries,
        test_entries=[],
        data_dir=str(data_dir),
        max_instances=60,
        min_instances=15,
        ecg_fs=200,
        ppg_fs=100,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No valid bags found for patient {patient_id}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Initialize lists (CRITICAL FIX: Define these before the loop)
    all_attention_weights = []
    bag_predictions = []
    
    # 3. Run Inference Loop
    print(f"Processing {len(dataloader)} bags...")
    with torch.no_grad():
        for bag_idx, (input_dict, label) in enumerate(dataloader):
            ecg = input_dict.get('ecg', None)
            ppg = input_dict.get('ppg', None)
            if ecg is not None: ecg = ecg.to(device)
            if ppg is not None: ppg = ppg.to(device)
            
            mask = input_dict['mask'].to(device)
            hrv = input_dict['hrv'].to(device)
            sleep_stage = input_dict['sleep_stage'].to(device)
            apnea = input_dict['apnea_label'].to(device)
            
            # Forward pass
            output = model(ecg=ecg, ppg=ppg, mask=mask, hrv=hrv, sleep_stage=sleep_stage, apnea=apnea)
            
            # Unpack output (Handle both single return and tuple return)
            if isinstance(output, tuple):
                logits, attn_weights = output
            else:
                # Fallback if model wasn't updated to return weights
                print("Warning: Model did not return attention weights. Check architecture.py")
                return 

            # Process Attention Weights
            if attn_weights is not None:
                valid_mask = mask[0].cpu().bool().numpy()
                attn_np = attn_weights[0].cpu().numpy()
                
                # Only keep valid segments
                valid_attn = attn_np[valid_mask]
                all_attention_weights.append(valid_attn)
                
                # Get probability
                prob = torch.sigmoid(logits[0]).cpu().item()
                bag_predictions.append(prob)

    # 4. Check if we collected data
    if not all_attention_weights:
        raise ValueError("No attention weights collected! Check your model returns.")

    # 5. Visualization Logic (The "Heatmap" Code)
    concatenated_attn = np.concatenate(all_attention_weights)
    avg_prediction = np.mean(bag_predictions)
    
    # Setup Grid
    # We create a grid where each row is roughly 10 minutes (20 segments)
    cols = 20 
    rows = math.ceil(len(concatenated_attn) / cols)
    
    pad_length = (rows * cols) - len(concatenated_attn)
    attn_grid = np.pad(concatenated_attn, (0, pad_length), constant_values=np.nan)
    attn_grid = attn_grid.reshape(rows, cols)
    
    # Create Plot
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])
    
    # Subplot 1: Strip
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        concatenated_attn.reshape(1, -1), 
        cmap='Reds', 
        cbar=False, 
        xticklabels=False, 
        yticklabels=False,
        ax=ax1
    )
    ax1.set_title(f"Patient {patient_id} - Global Attention Timeline (Prob: {avg_prediction:.4f})", fontsize=14, fontweight='bold')
    
    # Subplot 2: Grid
    ax2 = fig.add_subplot(gs[1])
    sns.heatmap(
        attn_grid,
        cmap='Reds',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Model Importance'},
        xticklabels=5,
        yticklabels=5,
        ax=ax2
    )
    ax2.set_xlabel("Segments (0-10 mins)", fontsize=10)
    ax2.set_ylabel("Time Blocks", fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()

def main():
    """Main function to run attention visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize attention weights for a patient')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--patient_id', type=str, required=True,
                        help='Patient ID to visualize (e.g., 49010118)')
    parser.add_argument('--output', type=str, default='results/figure_1_attention_timeline.png',
                        help='Output path for the figure')
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    output_path = Path(args.output)
    patient_id = args.patient_id
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(str(checkpoint_path), device)
    
    # Load patient data
    print(f"\nLoading data for patient {patient_id}...")
    bags, entries, norm_coefficients = load_patient_data(patient_id, JSON_PATH, DATA_DIR)
    
    # Generate visualization
    print(f"\nGenerating attention visualization...")
    visualize_attention(
        model=model,
        patient_id=patient_id,
        bags=bags,
        entries=entries,
        data_dir=DATA_DIR,
        device=device,
        output_path=output_path
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()


# python visualize_attention.py \
#     --checkpoint /data/R.Janini_Work/SA-CVD/ECG_PPG/multimodal-htn-framework/best_model_loso_patient_49010001.pt \
#     --patient_id 49010001 \
#     --output results/figure_1_attention_timeline.png



