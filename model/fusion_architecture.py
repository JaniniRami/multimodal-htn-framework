"""
DEPRECATED: This file is no longer used.
Neural network architecture for ECG-PPG multimodal fusion hypertension prediction.

Fusion ECG and PPG backbones: attention pooling over time only (no statistics
[mean, std, max] pooling). MIL uses gated attention over instances.

Module I (Instance Processing, 30s):
  - Multi-scale CNN (2 branches: fine + coarse)
  - Attention pooling over time (learned weights) — no stats pool
  - Instance embedding h (64 dim)

ECG–PPG fusion (paired instances, no cross-attention):
  - Per-instance: ecg_features (B,T,64), ppg_features (B,T,64) from backbones.
  - Instance-level context: sleep stage + apnea concatenated with [ECG_t, PPG_t] per t → paired (B, T, 128+context_dim).
  - Instance fusion MLP (2*embed_dim+context_dim → embed_dim) on paired → (B, T, 64); positional encoding; cross-modal + context patterns (e.g. PTT) emerge here.
  - MIL (gated attention) pools fused instances → bag (B, 64).

Bag-level only:
  - HRV/PRV: bag-level projections concatenated to bag embedding before classifier (sleep/apnea are instance-level only).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ---------------------------------------------------------------------------
# Positional encoding (zero learnable parameters)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Add position information to instance (segment) embeddings.
    Zero learnable parameters.
    """
    def __init__(self, embed_dim: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        num_freqs = embed_dim // 2
        div_term = torch.exp(
            torch.arange(num_freqs, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2][:, :num_freqs] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[: x.size(1), :]


# ---------------------------------------------------------------------------
# Attention pooling over time (used in fusion ECG/PPG backbones; no stats pool)
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Pool over time with learned attention. Input (B, C, T) -> output (B, C).
    Used in fusion ECGBackbone and PPGBackbone (no statistics pooling).
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        scores = self.attn(x).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)  # (B, 1, T)
        out = torch.bmm(weights, x).squeeze(1)  # (B, C)
        return out


# ---------------------------------------------------------------------------
# Module I: Multi-scale CNN + Attention Pooling → Instance embedding h
# ---------------------------------------------------------------------------
class ECGBackbone(nn.Module):
    """
    Module I (ECG): 30s at 200Hz → multi-scale CNN → attention pool → instance embedding h.
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        branch_channels = 32
        out_channels = 32

        self.fine = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=10, stride=2, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, branch_channels, kernel_size=6, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=branch_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.coarse = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=30, stride=3, padding=7),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, branch_channels, kernel_size=14, stride=2, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=branch_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(2 * branch_channels, out_channels, kernel_size=6, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_pool = AttentionPooling(out_channels, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(out_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, L) at 200Hz.
        Returns: (B, embed_dim) — feature vector h from Module I.
        """
        fine_out = self.fine(x)
        coarse_out = self.coarse(x)
        t = min(fine_out.size(-1), coarse_out.size(-1))
        fine_out = fine_out[..., :t]
        coarse_out = coarse_out[..., :t]
        x = torch.cat([fine_out, coarse_out], dim=1)
        x = self.mix(x)
        x = self.attention_pool(x)  # (B, out_channels)
        return self.proj(x)


class PPGBackbone(nn.Module):
    """
    Module I (PPG): 30s at 100Hz → multi-scale CNN → attention pool → instance embedding h.
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        branch_channels = 32
        out_channels = 32

        self.fine = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, branch_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=branch_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.coarse = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=3, padding=7),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, branch_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=branch_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(2 * branch_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_pool = AttentionPooling(out_channels, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(out_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, L) at 100Hz.
        Returns: (B, embed_dim) — feature vector h from Module I.
        """
        fine_out = self.fine(x)
        coarse_out = self.coarse(x)
        t = min(fine_out.size(-1), coarse_out.size(-1))
        fine_out = fine_out[..., :t]
        coarse_out = coarse_out[..., :t]
        x = torch.cat([fine_out, coarse_out], dim=1)
        x = self.mix(x)
        x = self.attention_pool(x)  # (B, out_channels)
        return self.proj(x)



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
    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # (B, T, A)
        a = self.dropout(a)  # Apply dropout to attention features
        logits = self.w(a)                                    # (B, T, 1)
        A = masked_softmax(logits, mask)                      # (B, T, 1)
        M = torch.sum(A * h, dim=1)                           # (B, D)
        return M, A.squeeze(-1)


class FusionNet(nn.Module):
    """
    Fusion network: paired ECG–PPG instances + instance-level context (sleep, apnea) → instance fusion MLP → MIL; only HRV/PRV at bag level.
    """
    def __init__(
        self,
        embed_dim=64,
        sleep_embed_dim=8,
        hrv_dim=5,
        use_ecg_ppg=True,
        use_hrv=True,
        use_ppg_ft=True,
        use_sleep_stage=True,
        use_apnea=True,
        attn_dim=128,
        **kwargs,  # ignore mil_pooling, topk_k from config
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_ecg_ppg = use_ecg_ppg
        self.use_hrv = use_hrv
        self.use_ppg_ft = use_ppg_ft
        self.use_sleep_stage = use_sleep_stage
        self.use_apnea = use_apnea

        sleep_dim = sleep_embed_dim if self.use_sleep_stage else 0
        apnea_dim = 1 if self.use_apnea else 0
        context_dim = sleep_dim + apnea_dim

        # Only create signal encoders and instance fusion if we're using ECG/PPG
        if self.use_ecg_ppg:
            self.ecg_encoder = ECGBackbone(1, embed_dim, dropout=0.5)
            self.ppg_encoder = PPGBackbone(1, embed_dim, dropout=0.5)
            # Instance fusion MLP: [ecg, ppg, context] (B, T, 2*embed_dim+context_dim) → (B, T, embed_dim)
            self.instance_fusion_mlp = nn.Sequential(
                nn.Linear(2 * embed_dim + context_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.3),
            )
            # Positional encoding over instance (segment) dimension
            self.pos_encoding = PositionalEncoding(embed_dim, max_len=100)

        if self.use_sleep_stage:
            self.stage_embed = nn.Embedding(num_embeddings=6, embedding_dim=sleep_embed_dim, padding_idx=0)

        # Scalar-only embedding layer (when signals are not provided)
        scalar_input_dim = sleep_dim + apnea_dim
        if scalar_input_dim > 0:
            self.scalar_embed = nn.Sequential(
                nn.Linear(scalar_input_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.4),
            )
        else:
            # If no scalars, create a learnable embedding
            self.scalar_embed = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Dropout(0.4),
            )

        self.fusion_ln = nn.LayerNorm(embed_dim)

        hrv_out_dim = 0
        if self.use_hrv:
            self.hrv_proj = nn.Sequential(
                nn.LayerNorm(hrv_dim),
                nn.Linear(hrv_dim, 16),
                nn.GELU(),
                nn.Dropout(0.3),
            )
            hrv_out_dim = 16

        ppg_out_dim = 0
        if self.use_ppg_ft:
            self.ppg_proj = nn.Sequential(
                nn.LayerNorm(5),
                nn.Linear(5, 16),
                nn.GELU(),
                nn.Dropout(0.3),
            )
            ppg_out_dim = 16

        # MIL: gated attention over instances (B, T, embed_dim)
        self.instance_attention = GatedAttentionMIL(embed_dim, attn_dim, dropout=0.0)

        # Final classifier: bag = [MIL(fused), hrv, ppg_ft]; sleep/apnea are at instance level only
        classifier_in_dim = embed_dim + hrv_out_dim + ppg_out_dim
        self.pre_classifier_drop = nn.Dropout(0.4)
        self.bag_classifier = nn.Linear(classifier_in_dim, 1)

    def forward(self, ecg=None, ppg=None, mask=None, hrv=None, sleep_stage=None, apnea=None, ppg_ft=None, return_embeddings=False):
        """
        Returns:
            tuple: (logits, instance_attn_weights, cross_attn_weights) or, if return_embeddings=True, (logits, attn_weights, cross_attn_weights, bag_embeddings)
            - logits: (B, 1)
            - instance_attn_weights: (B, T) or None - weights over instances when pooling to bag
            - cross_attn_weights: (B, T, embed_dim) or None - fusion gate (ECG vs PPG per channel), only in full ECG+PPG mode
            - bag_embeddings (optional): (B, D) - embeddings before final classifier, for t-SNE diagnostics
        """
        scalar_only = not self.use_ecg_ppg or (ecg is None) or (ppg is None)
        attn_weights = None
        cross_attn_weights = None  # Fusion gate: ECG vs PPG per channel (B, T, embed_dim)
        
        if scalar_only:
            # Scalar-only mode: use only HRV, sleep stage, and apnea
            # Infer batch size from available inputs
            if hrv is not None:
                B = hrv.shape[0]
                device = hrv.device
            elif mask is not None:
                B = mask.shape[0]
                device = mask.device
            elif sleep_stage is not None:
                B = sleep_stage.shape[0]
                device = sleep_stage.device
            elif apnea is not None:
                B = apnea.shape[0]
                device = apnea.device
            else:
                raise ValueError("At least one of hrv, mask, sleep_stage, or apnea must be provided")
            
            # Create instance-level embeddings from scalars (if available)
            scalar_features = []
            has_instance_features = False
            
            if self.use_sleep_stage and sleep_stage is not None:
                T = sleep_stage.shape[1] if len(sleep_stage.shape) > 1 else 1
                stage_emb = self.stage_embed(sleep_stage)  # (B, T, sleep_embed_dim)
                scalar_features.append(stage_emb)
                has_instance_features = True
            
            if self.use_apnea and apnea is not None:
                if not has_instance_features:
                    T = apnea.shape[1] if len(apnea.shape) > 1 else 1
                # Ensure apnea is (B, T) shape, then add dimension
                if len(apnea.shape) == 1:
                    apnea = apnea.unsqueeze(1)  # (B, 1)
                apnea_raw = apnea.unsqueeze(-1).float()  # (B, T, 1)
                scalar_features.append(apnea_raw)
                has_instance_features = True
            
            if has_instance_features and len(scalar_features) > 0:
                # We have instance-level features: create embeddings and use attention
                combined_scalars = torch.cat(scalar_features, dim=-1)  # (B, T, scalar_dim)
                fused = self.scalar_embed(combined_scalars.view(-1, combined_scalars.shape[-1])).view(B, T, self.embed_dim)
                fused = self.fusion_ln(fused)
                
                # Use attention to aggregate instance-level embeddings
                if mask is not None:
                    bag_emb, attn_weights = self.instance_attention(fused, mask.bool())
                else:
                    # If no mask, use mean pooling
                    bag_emb = fused.mean(dim=1)  # (B, embed_dim)
                    attn_weights = None
            else:
                # No instance-level features: only HRV (bag-level)
                # Create zero embedding to match expected dimension
                bag_emb = torch.zeros(B, self.embed_dim, device=device)
                attn_weights = None
            # Sleep/apnea are at instance level only; bag_emb is (B, embed_dim)

        else:
            # Full mode: paired instances + instance-level context → instance fusion MLP → MIL; only HRV/ppg_ft at bag level
            B, T, L_ecg = ecg.shape
            _, _, L_ppg = ppg.shape

            ecg_flat = ecg.view(B * T, 1, L_ecg)
            ppg_flat = ppg.view(B * T, 1, L_ppg)

            # Step 1: Backbones → per-instance features (B, T, 64)
            ecg_features = self.ecg_encoder(ecg_flat).view(B, T, self.embed_dim)
            ppg_features = self.ppg_encoder(ppg_flat).view(B, T, self.embed_dim)

            # Step 2: Add instance-level context (sleep stage, apnea) like standalone models
            add_ons = []
            if self.use_sleep_stage:
                stage_emb = self.stage_embed(sleep_stage) if sleep_stage is not None else torch.zeros(B, T, self.stage_embed.embedding_dim, device=ecg_features.device, dtype=ecg_features.dtype)
                add_ons.append(stage_emb)
            if self.use_apnea:
                apnea_raw = apnea.unsqueeze(-1).float() if apnea is not None else torch.zeros(B, T, 1, device=ecg_features.device, dtype=ecg_features.dtype)
                add_ons.append(apnea_raw)

            # Step 3: Pair with context (always match 2*embed_dim + context_dim for MLP)
            if add_ons:
                context = torch.cat(add_ons, dim=-1)
                paired = torch.cat([ecg_features, ppg_features, context], dim=-1)
            else:
                paired = torch.cat([ecg_features, ppg_features], dim=-1)

            # Step 4: Instance fusion MLP (learns cross-modal + context)
            fused = self.instance_fusion_mlp(paired)  # (B, T, embed_dim)
            fused = fused + self.pos_encoding(fused)

            # Step 5: MIL pools fused instances → bag embedding
            bag_emb, attn_weights = self.instance_attention(fused, mask.bool())  # (B, embed_dim)
            cross_attn_weights = None  # no fusion gate in this design

            # HRV at bag level (concat after MIL)
            if self.use_hrv:
                hrv_emb = self.hrv_proj(hrv) if hrv is not None else torch.zeros(B, 16, device=bag_emb.device, dtype=bag_emb.dtype)
                bag_emb = torch.cat([bag_emb, hrv_emb], dim=-1)

        # In scalar-only mode, HRV is still concatenated at bag level (no instance stream to broadcast to)
        if scalar_only and self.use_hrv and hrv is not None:
            hrv_emb = self.hrv_proj(hrv)
            bag_emb = torch.cat([bag_emb, hrv_emb], dim=-1)

        if self.use_ppg_ft:
            if ppg_ft is None:
                # Create zero tensor if ppg_ft is not provided
                device = bag_emb.device
                ppg_ft = torch.zeros(B, 5, device=device)
            ppg_ft_emb = self.ppg_proj(ppg_ft)
            bag_emb = torch.cat([bag_emb, ppg_ft_emb], dim=-1)

        # Embeddings before final classifier (for diagnostics: cluster by patient vs by CVD)
        embeddings_before_classifier = bag_emb  # (B, embed_dim + hrv_out_dim + ppg_out_dim)

        # Apply dropout before final classifier
        bag_emb = self.pre_classifier_drop(bag_emb)
        out = self.bag_classifier(bag_emb)
        
        # Return: logits, instance attention weights (B, T), cross-attention weights (B, T, embed_dim) or None
        # cross_attn_weights: fusion gate (ECG vs PPG per channel), only in full ECG+PPG mode
        attn_out = attn_weights.detach() if attn_weights is not None else None
        if return_embeddings:
            return out, attn_out, cross_attn_weights, embeddings_before_classifier
        return out, attn_out, cross_attn_weights
