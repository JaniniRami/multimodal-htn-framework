import os
import sys
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import pickle
import pywt
import gc
import h5py
from pprint import pprint
import math
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

from scipy.signal import decimate
from scipy import signal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
from glob import glob
import h5py
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import Subset
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose
from torchvision.models import resnet18
from collections import Counter
from math import log
from torch.amp import GradScaler, autocast
import random
from transformers import get_cosine_schedule_with_warmup


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # optional: reproducibility
    torch.backends.cudnn.benchmark = False     # optional: slower, more deterministic

# Example usage
set_seed(42)


data_dir = "/data/R.Janini_Work/SA-CVD/ECG_PPG/data/ECG_PPG_30S"
folds_dir = "/data/R.Janini_Work/SA-CVD/ECG_PPG/metadata/folds"
val_path = "/data/R.Janini_Work/SA-CVD/ECG_PPG/metadata/val_split.csv"

ecg_fs = 200
ppg_fs = 100

def crawlDir(dir_path):
    """
    Recursively crawls through a directory and returns a list of all files.
    """
    file_list = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list[file.split('.')[0]] = os.path.join(root, file)
    return file_list

derived_features_tree = crawlDir(os.path.join(data_dir, "derived_features"))

def get_derived_features_paths(csv_path, fold=None, split=None):
    osa_only, osa_with_cvd = [], []

    if fold:
        if split == "train":
            csv_path = os.path.join(folds_dir, f"train_split_{fold}.csv")
        elif split == "test":
            csv_path = os.path.join(folds_dir, f"test_split_{fold}.csv")

    val_guide = pd.read_csv(csv_path, sep=",")
    edf_file_paths = val_guide["edf_file_path"].values
    record_ids = [os.path.basename(path).split(".")[0] for path in edf_file_paths]
    derived_features_paths = [derived_features_tree[record_id] for record_id in record_ids if record_id in derived_features_tree]
    for path in derived_features_paths:
        if "OSA_ONLY" in path:
            osa_only.append(path)
        elif "OSA_WITH_CVD" in path:
            osa_with_cvd.append(path)

    return osa_only, osa_with_cvd

osa_only_val_paths, osa_with_cvd_val_paths = get_derived_features_paths(val_path)
osa_only_train_paths, osa_with_cvd_train_paths = get_derived_features_paths(folds_dir, fold=1, split="train")
osa_only_test_paths, osa_with_cvd_test_paths = get_derived_features_paths(folds_dir, fold=1, split="test")


def loadHRVData(data_dir):
    paths_base_name = ["osa_only_train", "osa_only_val", "osa_only_test",
             "osa_with_cvd_train", "osa_with_cvd_val", "osa_with_cvd_test"]
    paths = [os.path.join("/data/R.Janini_Work/SA-CVD/ECG_PPG/data/ECG_PPG_MIL_V2", f"{name}_hrv.npz") for name in paths_base_name]
    all_data = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            hrv_vector = data[key]
            # all_data[key] = data[key]
            # all_data[key] = [hrv_vector[1], hrv_vector[2], hrv_vector[5], hrv_vector[6], hrv_vector[-2], hrv_vector[-3]]
            # all_data[key] = [hrv_vector[5], hrv_vector[6], hrv_vector[-2], hrv_vector[-3]]
            names = [
                "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_PNN50", "HRV_SDSD",
                "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "sd1", "sd2", "sd1_sd2_ratio"
            ]
            data_dict = {}
            for idx, name in enumerate(names):
                data_dict[name] = hrv_vector[idx]
            
            selected_features = ["HRV_MeanNN", "HRV_SDNN", "HRV_LFHF", "HRV_LFn", "HRV_HFn"]
            all_data[key] = [data_dict[name] for name in selected_features]
           
    return all_data


hrv_data = loadHRVData(data_dir)

def load_data(path_list, hrv_data):
    """
    Load segments and group by record_id and hour_id into a flat dict of bags.
    Each bag_id is 'recordid_hourid' and maps to lists of instances.
    """
    dataset = {}
    for file_path in path_list:
        fname = os.path.basename(file_path)
        with h5py.File(file_path, 'r') as f:
            print(f"{len(f)} segments in {fname}")
            for segment in f:
                seg = f[segment]
                record_id = seg.attrs['record_id']
                hour_id = int(seg.attrs['hour_id'])
                bag_id = f"{record_id}_{hour_id}"

                # Get HRV features and skip if they contain any NaN
                hrv_features = hrv_data.get(bag_id, None)
                if hrv_features is None or np.isnan(hrv_features).any():
                    print(f"Skipping {bag_id} due to NaN in HRV features")
                    continue  # skip this segment

                # initialize bag if first time
                if bag_id not in dataset:
                    dataset[bag_id] = {
                        'ecg': [],
                        'ppg': [],
                        'sleep_stage': [],
                        'hrv_features': [],
                        'label': []
                    }

                ecg_segment = seg['ecg_segment'][:]
                ppg_segment = seg['ppg_segment'][:]
                sleep_stage = seg.attrs['sleep_stage']
                label = seg.attrs['label']

                # append arrays
                dataset[bag_id]['ecg'].append(ecg_segment)
                dataset[bag_id]['ppg'].append(ppg_segment)
                dataset[bag_id]['sleep_stage'].append(sleep_stage)
                dataset[bag_id]['label'].append(label)

                # prevent duplicate HRV features
                if not any(np.array_equal(hrv_features, existing) for existing in dataset[bag_id]['hrv_features']):
                    dataset[bag_id]['hrv_features'].append(hrv_features)

    return dataset



osa_only_train = load_data(osa_only_train_paths, hrv_data)
osa_with_cvd_train = load_data(osa_with_cvd_train_paths, hrv_data)
osa_only_val = load_data(osa_only_val_paths, hrv_data)
osa_with_cvd_val = load_data(osa_with_cvd_val_paths, hrv_data)
osa_only_test = load_data(osa_only_test_paths, hrv_data)
osa_with_cvd_test = load_data(osa_with_cvd_test_paths, hrv_data)


splits = {
    'TRAIN': {
        'OSA-only': osa_only_train,
        'OSA+ CVD':  osa_with_cvd_train
    },
    'VAL': {
        'OSA-only': osa_only_val,
        'OSA+ CVD':  osa_with_cvd_val
    },
    'TEST': {
        'OSA-only': osa_only_test,
        'OSA+ CVD':  osa_with_cvd_test
    }
}

for split_name, groups in splits.items():
    print(f"{split_name} split summary:")
    total_bags = 0
    total_segs = 0
    for group_name, ds in groups.items():
        n_bags = len(ds)
        n_segs = sum(len(bag['ecg']) for bag in ds.values())
        print(f"  {group_name:>10s}: {n_bags:4d} bags, {n_segs:5d} segments")
        total_bags += n_bags
        total_segs += n_segs
    print(f"  {'TOTAL':>10s}: {total_bags:4d} bags, {total_segs:5d} segments\n")


splits = {
'TRAIN': {
    'OSA-only': osa_only_train,
    'OSA+ CVD':  osa_with_cvd_train}
}

for split_name, groups in splits.items():
    print(f"{split_name} split summary:")
    total_bags = 0
    total_segs = 0
    total = []
    for group_name, ds in groups.items():
        n_bags = len(ds)
        n_segs = sum(len(bag['ecg']) for bag in ds.values())
        total.append(n_bags)

pos_weight = torch.tensor([total[0] / total[1]])
pos_weight



def compute_means_and_stds(no_cvd_dataset, cvd_dataset):
    """
    Compute means and standard deviations for each feature
    across all bags in both datasets.
    """
    means = {}
    stds = {}

    for dataset in (no_cvd_dataset, cvd_dataset):
        for bag in dataset.values():
            for feature in ('ecg', 'ppg', 'hrv_features'):
                concatenated = np.concatenate(bag[feature])
                means.setdefault(feature, []).append(concatenated.mean())
                stds.setdefault(feature, []).append(concatenated.std())

    for feature in means:
        means[feature] = np.mean(means[feature])
        stds[feature]  = np.mean(stds[feature])

    return means, stds

means, stds = compute_means_and_stds(osa_only_train, osa_with_cvd_train)



def normalize_dataset(dataset, means, stds):
    """
    Return a new dataset dict where 'ecg', 'ppg' and 'hrv_features' in each bag
    are z‐scored: (x - mean) / std, using the global means/stds.
    """
    normalized = {}
    for bag_id, bag in dataset.items():
        # copy everything except we’ll overwrite the continuous features
        new_bag = {
            'sleep_stage': bag['sleep_stage'],
            'label':       bag['label'],
        }
        # normalize each feature list
        for feature in ('ecg', 'ppg', 'hrv_features'):
            m = means[feature]
            s = stds[feature]
            new_bag[feature] = [ (seg - m) / s for seg in bag[feature] ]
        normalized[bag_id] = new_bag
    return normalized

# compute your means/stds once
means, stds = compute_means_and_stds(osa_only_train, osa_with_cvd_train)

# then normalize each split
osa_only_train_norm    = normalize_dataset(osa_only_train,    means, stds)
osa_with_cvd_train_norm= normalize_dataset(osa_with_cvd_train,means, stds)

osa_only_test_norm     = normalize_dataset(osa_only_test,     means, stds)
osa_with_cvd_test_norm = normalize_dataset(osa_with_cvd_test, means, stds)

osa_only_val_norm      = normalize_dataset(osa_only_val,      means, stds)
osa_with_cvd_val_norm  = normalize_dataset(osa_with_cvd_val,  means, stds)



class RandomShift:
    def __init__(self, max_shift_seconds=2, sampling_rate=200):
        self.max_shift = int(max_shift_seconds * sampling_rate)

    def __call__(self, x):
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
        return torch.roll(x, shifts=shift, dims=-1)
class GaussianNoise:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def __call__(self, x):
        std = x.std(unbiased=False)
        return x + torch.randn_like(x) * (std * self.noise_level)

class AmplitudeScale:
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_s, self.max_s = min_scale, max_scale

    def __call__(self, x):
        factor = torch.empty(1).uniform_(self.min_s, self.max_s)
        return x * factor

class TimeMask:
    def __init__(self, max_masks=10, patch_size=200, p=0.5):
        self.max_masks = max_masks
        self.patch = patch_size
        self.prob = p

    def __call__(self, x):
        # x: (n_samples,)  assume n_samples is divisible by patch_size
        if torch.rand(()) > self.prob:
            return x
        n_patches = x.shape[-1] // self.patch
        k = torch.randint(1, self.max_masks+1, (1,)).item()
        for _ in range(k):
            idx = torch.randint(0, n_patches, (1,)).item()
            start = idx * self.patch
            x[start : start + self.patch] = 0.
        return x



class ECGPPGBagDataset(Dataset):
    def __init__(
            self,
            no_cvd_dataset: dict,
            cvd_dataset: dict,
            augment: bool = False,
            max_instances: int = 120,
            min_instances: int = 15
    ):
        """
        no_cvd_dataset, cvd_dataset: dict mapping bag_id -> {
            'ecg': [np.ndarray,...],
            'ppg': [...],
            'sleep_stage': [...],
            'hrv_features': [...],
            'label': [...]  # apnea flag per segment
        }
        """
        self.max_instances = max_instances
        self.min_instances = min_instances
        self.augment = augment

        # merge and filter by minimum instances
        merged = {**no_cvd_dataset, **cvd_dataset}
        filtered_ids = [bid for bid, bag in merged.items()
                        if len(bag['ecg']) >= self.min_instances]

        self.bag_ids = []
        self.datasets = {}
        self.cvd_labels = {}
        # assign CVD labels
        for bid in filtered_ids:
            self.datasets[bid] = merged[bid]
            self.bag_ids.append(bid)
            # label 0 for no-cvd, 1 for cvd
            self.cvd_labels[bid] = 0 if bid in no_cvd_dataset else 1

        # augmentation pipeline
        self.ecg_augment = Compose([
            GaussianNoise(noise_level=0.01),
            AmplitudeScale(min_scale=0.9, max_scale=1.1),
            RandomShift(max_shift_seconds=2, sampling_rate=ecg_fs),
        ])

        self.ppg_augment = Compose([
            GaussianNoise(noise_level=0.01),
            AmplitudeScale(min_scale=0.9, max_scale=1.1),
            RandomShift(max_shift_seconds=2, sampling_rate=ppg_fs),
        ])

    def __len__(self):
        return len(self.bag_ids)

    def __getitem__(self, idx):
        bag_id = self.bag_ids[idx]
        bag = self.datasets[bag_id]
        n = len(bag['ecg'])

        # down-sample large bags
        if n > self.max_instances:
            print(f"Down-sampling bag {bag_id} from {n} to {self.max_instances} instances")
            indices = np.random.choice(n, self.max_instances, replace=False)
            ecg_list = [bag['ecg'][i] for i in indices]
            ppg_list = [bag['ppg'][i] for i in indices]
            stage_list = [bag['sleep_stage'][i] for i in indices]
            label_list = [bag['label'][i] for i in indices]
            n = self.max_instances
        else:
            ecg_list = bag['ecg']
            ppg_list = bag['ppg']
            stage_list = bag['sleep_stage']
            label_list = bag['label']

        hrv_list = np.array(bag['hrv_features'][0])

        pad_n = self.max_instances - n

        def pad_and_stack(lst):
            if pad_n > 0:
                pad = [np.zeros_like(lst[0])] * pad_n
                lst = lst + pad
            return np.stack(lst, axis=0)

        ecg_arr = pad_and_stack(ecg_list)
        ppg_arr = pad_and_stack(ppg_list)
        stage_arr = pad_and_stack(stage_list)
        lab_arr = pad_and_stack(label_list)

        mask = torch.zeros(self.max_instances, dtype=torch.bool)
        mask[:n] = True

        ecg_t = torch.from_numpy(ecg_arr).float()
        ppg_t = torch.from_numpy(ppg_arr).float()
        stage_t = torch.from_numpy(stage_arr).long()
        hrv_t = torch.from_numpy(hrv_list).float()
        lab_t = torch.from_numpy(lab_arr).float()
        if self.augment:
            ecg_t = torch.stack([self.ecg_augment(x) for x in ecg_t])
            ppg_t = torch.stack([self.ppg_augment(x) for x in ppg_t])

        cvd_t = torch.tensor(self.cvd_labels[bag_id], dtype=torch.float32)
        return ({
            'ecg': ecg_t,
            'ppg': ppg_t,
            'sleep_stage': stage_t,
            'apnea_label': lab_t,
            'mask': mask,
            "record_id": bag_id.split('_')[0]
        }, hrv_t), cvd_t

# usage
train_ds = ECGPPGBagDataset(osa_only_train_norm, osa_with_cvd_train_norm, augment=False)
val_ds   = ECGPPGBagDataset(osa_only_val_norm,   osa_with_cvd_val_norm,   augment=False)
test_ds  = ECGPPGBagDataset(osa_only_test_norm,  osa_with_cvd_test_norm,  augment=False)



train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)



class ECGBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Conv1d(128, embed_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),

                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU()
            )
    
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, 1, L)
        Returns:
            (B, embed_dim)
        """
        x = self.encoder(x)
        x = self.global_pool(x).squeeze(-1)
        return x


class ResidualBiLSTMBlock(nn.Module):
    """Bidirectional LSTM followed by projection + residual + LayerNorm + ReLU."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out, _ = self.bilstm(x)        # (B, T, 2*H)
        out = self.proj(out)           # (B, T, D)
        out = self.norm(out + residual)
        return self.activation(out)



def masked_softmax(logits: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = 1,
                   eps: float = 1e-8) -> torch.Tensor:
        """
        logits : (B, T, 1)
        mask   : (B, T)  bool
        returns softmax over dim with pads set to zero
        """
        mask  = mask.unsqueeze(-1)                     # (B, T, 1)
        big_n = -1e4                                   # safe in fp16
        logits = logits.masked_fill(~mask, big_n)
        attn   = torch.softmax(logits, dim=dim)
        attn   = attn * mask
        attn   = attn / (attn.sum(dim=dim, keepdim=True) + eps)
        return attn                                    # (B, T, 1)

class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # (B, T, A)
        logits = self.w(a)                                    # (B, T, 1)
        A = masked_softmax(logits, mask)                      # (B, T, 1)
        M = torch.sum(A * h, dim=1)                           # (B, D)
        return M, A.squeeze(-1)



class HypertensionModel(nn.Module):
    """
    Pipeline (updated fusion):
      1. ECG/PPG segment → CNN encoder → segment embeddings (B, T, embed_dim)
      2. Concatenate per‑segment sleep stage (embed) + raw apnea flag ➜ pass through a small MLP ("fusion") that maps back to `embed_dim`.
      3. Residual Bi‑LSTM across the fused sequence (context) → (B, T, embed_dim)
      4. Gated MIL pooling over (B, T, embed_dim)
      5. Optional HRV projection & fusion
      6. Bag‑level classifier → logit
    """

    def __init__(
        self,
        embed_dim: int = 256,
        rnn_hidden: int = 128,
        sleep_embed_dim: int = 8,
        hrv_dim: int = 5,
        use_hrv: bool = True,
        attn_dim: int = 128,
        backbone_cls= None,           # allow injection of ECGBackbone, PPGBackbone, etc.
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sleep_embed_dim = sleep_embed_dim
        self.hrv_dim = hrv_dim
        self.use_hrv = use_hrv

        # 1. Segment‑level encoder (default CNNBackbone if not provided)
        self.segment_encoder = ECGBackbone(1, embed_dim)

        # 2. Per‑segment context embeddings (sleep stage + raw apnea flag)
        self.stage_embed = nn.Embedding(num_embeddings=6, embedding_dim=sleep_embed_dim, padding_idx=0)

        # Fusion MLP: (embed_dim + sleep_embed_dim + 1) → embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + sleep_embed_dim + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # 3. Temporal Bi‑LSTM on fused embeddings
        self.temporal_rnn = ResidualBiLSTMBlock(input_dim=embed_dim, hidden_dim=rnn_hidden)

        # 4. Gated MIL pooling
        self.instance_attention = GatedAttentionMIL(in_dim=embed_dim, attn_dim=attn_dim)

        # 5. HRV projection (optional)
        if self.use_hrv:
            self.hrv_proj = nn.Sequential(
                nn.LayerNorm(hrv_dim),
                nn.Linear(hrv_dim, 16),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            hrv_out_dim = 16
        else:
            hrv_out_dim = 0

        # 6. Bag‑level classifier
        self.bag_classifier = nn.Linear(embed_dim + hrv_out_dim, 1)

    def forward(
        self,
        signal: torch.Tensor,       # (B, T, L)
        mask: torch.Tensor,         # (B, T) bool
        hrv: torch.Tensor,          # (B, hrv_dim)
        sleep_stage: torch.Tensor,  # (B, T)
        apnea: torch.Tensor,        # (B, T) 0/1
    ) -> torch.Tensor:
        B, T, L = signal.shape

        # 1. Encode each segment
        seg_flat = signal.view(B * T, 1, L)
        seg_emb = self.segment_encoder(seg_flat).view(B, T, self.embed_dim)  # (B, T, embed_dim)

        # 2. Bi‑LSTM over segment embeddings
        ctx_emb = self.temporal_rnn(seg_emb)  # (B, T, embed_dim)

        # 3. Fuse context with per‑segment sleep & apnea info
        stage_emb = self.stage_embed(sleep_stage)          # (B, T, sleep_embed_dim)
        apnea_raw = apnea.unsqueeze(-1).float()            # (B, T, 1)
        combined = torch.cat([ctx_emb, stage_emb, apnea_raw], dim=-1)  # (B, T, embed+sleep+1)
        fused = self.fusion(combined.view(-1, combined.shape[-1]))     # (B*T, embed_dim)
        fused = fused.view(B, T, self.embed_dim)                       # (B, T, embed_dim)

        # 4. MIL pooling
        bag_emb, _ = self.instance_attention(fused, mask.bool())

        # 5. Optional HRV fusion
        if self.use_hrv:
            hrv_emb = self.hrv_proj(hrv)  # (B, 16)
            bag_emb = torch.cat([bag_emb, hrv_emb], dim=-1)

        # 6. Logit
        return self.bag_classifier(bag_emb)



from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss
)

import torch
train_losses = []
def train_model(
        model,
        scaler: GradScaler,
        train_loader,
        val_loader,
        num_epochs,
        criterion,
        optimizer,
        scheduler,
        device,
        patience=15,
        save_path="best_model.pt"
):
    model.to(device)
    best_auprc = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # === TRAIN ===
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for input_data, y in train_loader:
            X, hrv = input_data
            ecg  = X['ecg'].to(device)   # [B, N, L]
            mask = X['mask'].to(device)  # [B, N]
            sleep_stage = X['sleep_stage'].to(device) # [B, N]
            ap_label = X['apnea_label'].to(device) # [B, N]
            hrv = hrv.to(device)  # [B, F]
            y    = y.view(-1).to(device) # [B]

            optimizer.zero_grad()


            with autocast(device_type='cuda'):
                logits = model(ecg, mask, hrv, sleep_stage, ap_label).squeeze(-1)  # [B]
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            # print(preds, y.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = accuracy_score(all_labels, all_preds)
        train_bal  = balanced_accuracy_score(all_labels, all_preds)
        train_mcc  = matthews_corrcoef(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds)
        train_auc  = roc_auc_score(all_labels, all_preds)
        train_auprc= average_precision_score(all_labels, all_preds)
        train_losses.append(train_loss)
    

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_labels, val_preds = [], []
        with torch.no_grad():
            for input_data, y in val_loader:
                X, hrv = input_data
                ecg  = X['ecg'].to(device)
                mask = X['mask'].to(device)
                hrv = hrv.to(device)
                sleep_stage = X['sleep_stage'].to(device)
                ap_label = X['apnea_label'].to(device)
                y    = y.view(-1).to(device)

                logits = model(ecg, mask, hrv, sleep_stage, ap_label).squeeze(-1)
                loss   = criterion(logits, y)
                val_loss += loss.item() * y.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_labels.extend(y.cpu().numpy())
                val_preds.extend(preds)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc  = accuracy_score(val_labels, val_preds)
        val_bal  = balanced_accuracy_score(val_labels, val_preds)
        val_mcc  = matthews_corrcoef(val_labels, val_preds)
        val_auc  = roc_auc_score(val_labels, val_preds) if len(set(val_labels))>1 else float('nan')
        val_auprc= average_precision_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)

        # # === LOG ===
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Train Loss:{train_loss:.4f} | Val Loss:{val_loss:.4f}")
        print(f"Train Acc :{train_acc:.4f} | Val Acc :{val_acc:.4f}")
        print(f"Train Prec:{train_precision:.4f} | Val Prec:{val_precision:.4f}")
        print(f"Train F1  :{train_f1:.4f} | Val F1  :{val_f1:.4f}")
        print(f"Train Bal :{train_bal:.4f} | Val Bal :{val_bal:.4f}")
        print(f"Train MCC :{train_mcc:.4f} | Val MCC :{val_mcc:.4f}")
        print(f"Train AUROC:{train_auc:.4f} | Val AUROC:{val_auc:.4f}")
        print(f"Train AUPRC:{train_auprc:.4f} | Val AUPRC:{val_auprc:.4f}")
        print("-"*30)

        # early stopping
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "best_model.pt")
            print(f"✔️ Saved model at epoch {epoch+1} (Val AUPRC: {val_auprc:.4f})")
        

   
    return best_epoch, best_auprc



        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HypertensionModel(
    embed_dim=256,
    rnn_hidden=128,
    attn_dim=128,
    hrv_dim=5,
    sleep_embed_dim=8,
    use_hrv=True,
)

model = nn.DataParallel(model, device_ids=[0,1,2,3])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                 factor=0.5, patience=3,
                                                 min_lr=1e-6)


scaler = GradScaler(enabled=torch.cuda.is_available())

#print total model paramaters mnumber
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")


train_model(
    model,
    scaler,
    train_loader,
    test_loader,
    num_epochs=75,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    patience=15,
    save_path="best_model.pt"
)
