"""
Fusion architecture using pretrained ECG and PPG encoders (full standalone models).

Pipeline:
  INPUT LAYER
  ├─ ECG signal (30s segments)
  ├─ PPG signal (30s segments)
  ├─ Mask (valid segments)
  ├─ HRV features (bag-level)
  ├─ Sleep stage (per segment)
  ├─ Apnea labels (per segment)
  └─ PPG features (bag-level)

  PRETRAINED ENCODERS (frozen or partially frozen)
  ├─ ECG Encoder (ECGNet) → bag embedding (B, ecg_emb_dim)  e.g. 80 or 144
  └─ PPG Encoder (PPGNet) → bag embedding (B, ppg_emb_dim)  e.g. 96 or 160

  FUSION LAYER (trained) — Gated Adaptive Fusion (Option 1)
  ├─ ECG_emb (ecg_emb_dim) → Gate Network → importance score for ECG
  ├─ PPG_emb (ppg_emb_dim) → Gate Network → importance score for PPG
  ├─ Softmax over [score_ecg, score_ppg] → per-sample weights (e.g. 80% PPG, 20% ECG)
  ├─ Weighted ECG = ECG_emb × ECG_importance, Weighted PPG = PPG_emb × PPG_importance
  ├─ Combined = Concat[Weighted ECG, Weighted PPG] → (B, ecg_emb_dim + ppg_emb_dim)
  └─ MLP: (176 or dims) → 128 → 64 → 1 → Prediction

  OUTPUT: Hypertension prediction (B, 1)

LOSO: For each fold, load the corresponding ECG and PPG checkpoints for that
left-out patient (same fold index), e.g.:
  ecg_ckpt = ecg_run_dir / f"best_model_ecg_loso_patient_{test_patient_id}.pt"
  ppg_ckpt = ppg_run_dir / f"best_model_ppg_loso_patient_{test_patient_id}.pt"
"""

from pathlib import Path
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

try:
    from .ecg_architecture import ECGNet
    from .ppg_architecture import PPGNet
except ImportError:
    from ecg_architecture import ECGNet
    from ppg_architecture import PPGNet


def _strip_data_parallel(state_dict: dict) -> dict:
    """Remove 'module.' prefix from keys if present (saved with DataParallel)."""
    stripped = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        stripped[key] = v
    return stripped


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load checkpoint from path; return state_dict (with module. stripped)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    return _strip_data_parallel(state_dict)


class FusionPretrainedNet(nn.Module):
    """
    Gated Adaptive Fusion of pretrained ECG and PPG encoders.

    - Runs full ECGNet and PPGNet with return_embeddings=True to get bag embeddings.
    - Per-modality gate: ECG_emb → Gate → importance score, PPG_emb → Gate → importance score.
    - Softmax over [score_ecg, score_ppg] gives per-sample weights (e.g. 80% PPG, 20% ECG).
    - Weighted ECG and Weighted PPG are concatenated, then MLP 176→128→64→1.
    - Encoders can be frozen; only gate networks and fusion MLP are trained.
    """

    def __init__(
        self,
        ecg_net: nn.Module,
        ppg_net: nn.Module,
        fusion_dropout: float = 0.3,
        freeze_encoders: bool = True,
        fusion_hidden_1: int = 128,
        fusion_hidden_2: int = 64,
    ):
        super().__init__()
        self.ecg_net = ecg_net
        self.ppg_net = ppg_net

        # Infer embedding dims from each model's classifier input
        ecg_emb_dim = ecg_net.bag_classifier.in_features
        ppg_emb_dim = ppg_net.bag_classifier.in_features
        self.ecg_emb_dim = ecg_emb_dim
        self.ppg_emb_dim = ppg_emb_dim
        fusion_in_dim = ecg_emb_dim + ppg_emb_dim

        if freeze_encoders:
            for p in self.ecg_net.parameters():
                p.requires_grad = False
            for p in self.ppg_net.parameters():
                p.requires_grad = False

        # Gated Adaptive Fusion: per-modality gate → importance score
        # ECG embedding → Gate Network → logit for ECG importance
        self.gate_ecg = nn.Sequential(
            nn.Linear(ecg_emb_dim, 32),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(32, 1),
        )
        # PPG embedding → Gate Network → logit for PPG importance
        self.gate_ppg = nn.Sequential(
            nn.Linear(ppg_emb_dim, 32),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(32, 1),
        )

        # MLP: Combined (176 or fusion_in_dim) → 128 → 64 → 1
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_1),
            nn.LayerNorm(fusion_hidden_1),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_1, fusion_hidden_2),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_2, 1),
        )

    def forward(
        self,
        ecg: Optional[torch.Tensor] = None,
        ppg: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        hrv: Optional[torch.Tensor] = None,
        sleep_stage: Optional[torch.Tensor] = None,
        apnea: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass: get bag embeddings from ECG and PPG nets, then fusion MLP.

        Returns:
            logits: (B, 1)
            ecg_attn: (B, T) or None
            ppg_attn: (B, T) or None
            If return_embeddings: (ecg_emb, ppg_emb) as well
        """
        # Slice HRV: ECG gets dims 0:2 (ecg_hf, ecg_lf), PPG gets dims 2:4 (ppg_hf, ppg_lf = PRV)
        hrv_ecg = hrv[:, :2] if hrv is not None and hrv.dim() >= 2 and hrv.size(-1) >= 2 else hrv
        hrv_ppg = hrv[:, 2:4] if hrv is not None and hrv.dim() >= 2 and hrv.size(-1) >= 4 else hrv
        # ECG encoder → bag embedding (before classifier)
        _, ecg_attn, ecg_emb = self.ecg_net(
            ecg=ecg,
            mask=mask,
            hrv=hrv_ecg,
            sleep_stage=sleep_stage,
            apnea=apnea,
            return_embeddings=True,
        )
        # PPG encoder → bag embedding (uses PRV: ppg_hf, ppg_lf)
        _, ppg_attn, ppg_emb = self.ppg_net(
            ppg=ppg,
            mask=mask,
            hrv=hrv_ppg,
            sleep_stage=sleep_stage,
            apnea=apnea,
            return_embeddings=True,
        )

        # Gated Adaptive Fusion: per-sample importance (e.g. 80% PPG, 20% ECG)
        logit_ecg = self.gate_ecg(ecg_emb).squeeze(-1)   # (B,)
        logit_ppg = self.gate_ppg(ppg_emb).squeeze(-1)   # (B,)
        importance = torch.softmax(torch.stack([logit_ecg, logit_ppg], dim=-1), dim=-1)  # (B, 2)
        weighted_ecg = ecg_emb * importance[:, 0:1]
        weighted_ppg = ppg_emb * importance[:, 1:2]

        combined = torch.cat([weighted_ecg, weighted_ppg], dim=-1)
        logits = self.fusion_mlp(combined)

        ecg_attn_out = ecg_attn.detach() if ecg_attn is not None else None
        ppg_attn_out = ppg_attn.detach() if ppg_attn is not None else None

        if return_embeddings:
            return logits, ecg_attn_out, ppg_attn_out, weighted_ecg, weighted_ppg
        return logits, ecg_attn_out, ppg_attn_out


def create_fusion_pretrained_for_fold(
    ecg_run_dir: Path,
    ppg_run_dir: Path,
    test_patient_id: Any,
    device: torch.device,
    *,
    freeze_encoders: bool = True,
    embed_dim: int = 128,
    sleep_embed_dim: int = 8,
    hrv_dim: int = 4,
    fusion_dropout: float = 0.3,
    fusion_hidden_1: int = 128,
    fusion_hidden_2: int = 64,
    ecg_hrv_dim: Optional[int] = None,
    ecg_use_hrv: bool = True,
    ppg_use_prv: bool = True,
    ppg_prv_dim: Optional[int] = None,
    ecg_use_sleep_stage: bool = False,
    ecg_use_apnea: bool = False,
    ppg_use_sleep_stage: bool = False,
    ppg_use_apnea: bool = False,
    **kwargs: Any,
) -> FusionPretrainedNet:
    """
    Build FusionPretrainedNet for one LOSO fold: create ECGNet and PPGNet,
    load their checkpoints for this fold's test patient, then build the fusion model.

    Uses Gated Adaptive Fusion: per-modality gates → importance weights → MLP 176→128→64→1.

    Checkpoint naming (must match training):
      ecg_run_dir / f"best_model_ecg_loso_patient_{test_patient_id}.pt"
      ppg_run_dir / f"best_model_ppg_loso_patient_{test_patient_id}.pt"

    Args:
        ecg_run_dir: Directory containing ECG LOSO checkpoints.
        ppg_run_dir: Directory containing PPG LOSO checkpoints.
        test_patient_id: Patient ID left out in this fold (used in checkpoint filename).
        device: Device to load models onto.
        freeze_encoders: If True, freeze ECG and PPG parameters.
        embed_dim: Embed dimension for both encoders (must match how they were trained).
        sleep_embed_dim: Sleep embedding dimension (only used if use_sleep_stage=True).
        hrv_dim: Deprecated; use ecg_hrv_dim and ppg_prv_dim instead.
        fusion_dropout: Dropout in gate networks and fusion MLP.
        fusion_hidden_1: First hidden size in fusion MLP (default 128).
        fusion_hidden_2: Second hidden size in fusion MLP (default 64).
        ecg_hrv_dim: HRV dim for ECG (default 2 for ECG-only; must match checkpoint).
        ecg_use_hrv: If False, ECGNet has no HRV branch (must match checkpoint).
        ppg_use_prv: If False, PPGNet has no PRV branch (must match checkpoint).
        ppg_prv_dim: PRV dim for PPG (default 2 for ppg_hf, ppg_lf; must match checkpoint).
        ecg_use_sleep_stage: If False, ECGNet has no sleep stage (matches checkpoints trained without context).
        ecg_use_apnea: If False, ECGNet has no apnea in fusion (matches checkpoints trained without context).
        ppg_use_sleep_stage: If False, PPGNet has no sleep stage (matches checkpoints trained without context).
        ppg_use_apnea: If False, PPGNet has no apnea in fusion (matches checkpoints trained without context).
    """
    ecg_hrv_dim = ecg_hrv_dim if ecg_hrv_dim is not None else 2
    ppg_prv_dim = ppg_prv_dim if ppg_prv_dim is not None else 2

    ecg_ckpt_path = Path(ecg_run_dir) / f"best_model_ecg_loso_patient_{test_patient_id}.pt"
    ppg_ckpt_path = Path(ppg_run_dir) / f"best_model_ppg_loso_patient_{test_patient_id}.pt"

    if not ecg_ckpt_path.exists():
        raise FileNotFoundError(f"ECG checkpoint not found: {ecg_ckpt_path}")
    if not ppg_ckpt_path.exists():
        raise FileNotFoundError(f"PPG checkpoint not found: {ppg_ckpt_path}")

    ecg_net = ECGNet(
        embed_dim=embed_dim,
        sleep_embed_dim=sleep_embed_dim,
        hrv_dim=ecg_hrv_dim,
        use_hrv=ecg_use_hrv,
        use_sleep_stage=ecg_use_sleep_stage,
        use_apnea=ecg_use_apnea,
        attn_dim=kwargs.get("attn_dim", 128),
        mil_pooling=kwargs.get("mil_pooling", "attention"),
        topk_k=kwargs.get("topk_k", 5),
    )
    ppg_net = PPGNet(
        embed_dim=embed_dim,
        sleep_embed_dim=sleep_embed_dim,
        prv_dim=ppg_prv_dim,
        use_prv=ppg_use_prv,
        use_sleep_stage=ppg_use_sleep_stage,
        use_apnea=ppg_use_apnea,
        attn_dim=kwargs.get("attn_dim", 128),
        mil_pooling=kwargs.get("mil_pooling", "attention"),
        topk_k=kwargs.get("topk_k", 5),
    )

    ecg_sd = load_checkpoint(ecg_ckpt_path, device)
    ppg_sd = load_checkpoint(ppg_ckpt_path, device)

    ecg_net.load_state_dict(ecg_sd, strict=True)
    ppg_net.load_state_dict(ppg_sd, strict=True)

    ecg_net = ecg_net.to(device)
    ppg_net = ppg_net.to(device)

    model = FusionPretrainedNet(
        ecg_net=ecg_net,
        ppg_net=ppg_net,
        fusion_dropout=fusion_dropout,
        freeze_encoders=freeze_encoders,
        fusion_hidden_1=fusion_hidden_1,
        fusion_hidden_2=fusion_hidden_2,
    )
    return model.to(device)
