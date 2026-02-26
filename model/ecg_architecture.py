"""
Neural network architecture for ECG-only hypertension prediction.

Module I (Instance Processing, 30s):
  - Multi-scale CNN (2 branches: fine + coarse)
  - Attention pooling over time (learned weights) — no stats pool
  - Instance embedding h (64 dim)

Context (Sleep Stage ⊕ Apnea):
  - Concatenated with ECG instance embeddings per instance, then fused via MLP.

HRV (global context):
  - Project bag-level HRV → global_emb; broadcast to all instances; concat [h, global_emb] per instance.
  - MIL pooling runs on h_with_global; bag embedding = pooled over instances.

Expects ECG signal + HRV + sleep stage + apnea.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Positional encoding (zero learnable parameters)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Add position information to segment embeddings.
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
# Attention pooling over time (no statistics pooling)
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Pool over time with learned attention. Input (B, C, T) -> output (B, C).
    Used in ECGBackbone (no statistics pooling).
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


# ---------------------------------------------------------------------------
# Masked softmax for attention pooling
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MIL pooling variants
# ---------------------------------------------------------------------------
class MeanMaxPooling(nn.Module):
    """
    Mean-Max pooling over instances (segments) per bag.
    For each bag: compute mean and max over valid instances (respecting mask), then concatenate and project to in_dim.
    """
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        mask_exp = mask.unsqueeze(-1).float()  # (B, T, 1)
        h_masked = h * mask_exp
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
        mean_h = h_masked.sum(dim=1) / n_valid.squeeze(-1)                 # (B, D)
        max_h = h_for_max.max(dim=1).values                                # (B, D)

        combined = torch.cat([mean_h, max_h], dim=-1)  # (B, 2*D)
        M = self.proj(combined)                         # (B, D)

        weights = mask_exp.squeeze(-1)
        return M, weights


class MaxPoolMIL(nn.Module):
    """
    Max-pooling MIL: bag embedding = max over instances (per dimension), respecting mask.
    """
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        M = h_for_max.max(dim=1).values  # (B, D)
        M = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
        weights = mask.float()
        return M, weights


class TopKPooling(nn.Module):
    """
    Top-K MIL: score each instance, select top-k by score,
    bag embedding = mean of those top-k instance embeddings.
    """
    def __init__(self, in_dim: int, k: int = 5, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.k = k
        self.instance_scorer = nn.Linear(in_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        B, T, D = h.shape
        instance_logits = self.instance_scorer(self.dropout(h)).squeeze(-1)  # (B, T)
        instance_logits = instance_logits.masked_fill(~mask, float('-inf'))
        k_actual = min(self.k, T)
        topk_vals, topk_idx = torch.topk(instance_logits, k=k_actual, dim=1)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        h_topk = torch.gather(h, 1, topk_idx_exp)
        valid_topk = torch.gather(mask.unsqueeze(-1).float(), 1, topk_idx.unsqueeze(-1)).expand(-1, -1, D)
        h_topk_masked = h_topk * valid_topk
        count = valid_topk[:, :, 0].sum(dim=1, keepdim=True).clamp(min=1)
        M = h_topk_masked.sum(dim=1) / count
        weights = torch.zeros(B, T, device=h.device, dtype=h.dtype)
        weights.scatter_(1, topk_idx, valid_topk[:, :, 0] / count)
        return M, weights


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # (B, T, A)
        a = self.dropout(a)
        logits = self.w(a)                                    # (B, T, 1)
        A = masked_softmax(logits, mask)                      # (B, T, 1)
        M = torch.sum(A * h, dim=1)                           # (B, D)
        return M, A.squeeze(-1)


# ---------------------------------------------------------------------------
# ECGNet: ECG-only network mirroring FusionNet style
# ---------------------------------------------------------------------------
class ECGNet(nn.Module):
    """
    ECG-only network for hypertension prediction.
    Uses ECG signal + HRV  + sleep stage + apnea.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        sleep_embed_dim: int = 8,
        hrv_dim: int = 4,
        use_hrv: bool = True,
        use_sleep_stage: bool = True,
        use_apnea: bool = True,
        attn_dim: int = 128,
        mil_pooling: str = 'attention',  # 'topk', 'maxpool', 'attention', or 'meanmax'
        topk_k: int = 5,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mil_pooling = mil_pooling
        self.use_hrv = use_hrv
        self.use_sleep_stage = use_sleep_stage
        self.use_apnea = use_apnea

        # ECG encoder: multi-scale CNN + attention pooling (same dropout as PPG)
        self.ecg_encoder = ECGBackbone(1, embed_dim, dropout=0.3)
        # Positional encoding so segments know their order (zero learnable params)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=60)
        self.embed_dropout = nn.Dropout(0.3)  # after pos_encoding

        # Context features concatenated with ECG instance embeddings
        sleep_dim = sleep_embed_dim if self.use_sleep_stage else 0
        apnea_dim = 1 if self.use_apnea else 0

        if self.use_sleep_stage:
            self.stage_embed = nn.Embedding(num_embeddings=6, embedding_dim=sleep_embed_dim, padding_idx=0)

        # Fusion layer: ECG embed + context → embed_dim
        context_concat_dim = embed_dim + sleep_dim + apnea_dim
        self.fusion = nn.Sequential(
            nn.Linear(context_concat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.fusion_ln = nn.LayerNorm(embed_dim)

        # HRV global projection (broadcast to instances before MIL pooling)
        hrv_out_dim = 0
        if self.use_hrv:
            self.hrv_proj = nn.Sequential(
                nn.LayerNorm(hrv_dim),
                nn.Linear(hrv_dim, 16),
                nn.GELU(),
                nn.Dropout(0.3),
            )
            hrv_out_dim = 16

        # MIL pooling (input includes broadcast HRV if used; same dropout as PPG)
        instance_in_dim = embed_dim + hrv_out_dim if self.use_hrv else embed_dim
        if mil_pooling == 'topk':
            self.instance_attention = TopKPooling(instance_in_dim, k=topk_k, dropout=0.2)
        elif mil_pooling == 'maxpool':
            self.instance_attention = MaxPoolMIL(instance_in_dim, dropout=0.2)
        elif mil_pooling == 'meanmax':
            self.instance_attention = MeanMaxPooling(instance_in_dim, dropout=0.2)
        else:
            self.instance_attention = GatedAttentionMIL(instance_in_dim, attn_dim, dropout=0.2)

        # Final classifier with dropout (same as PPG)
        self.pre_classifier_drop = nn.Dropout(0.3)
        self.bag_classifier = nn.Linear(embed_dim + hrv_out_dim, 1)

    def forward(self, ecg=None, ppg=None, mask=None, hrv=None,
                sleep_stage=None, apnea=None,
                return_embeddings: bool = False):
        """
        Args:
            ecg:          (B, T, L_ecg) — ECG signal segments
            mask:         (B, T) — boolean mask for valid segments
            hrv:          (B, hrv_dim) — bag-level HRV features
            sleep_stage:  (B, T) — integer sleep stage per segment
            apnea:        (B, T) — apnea label per segment
            return_embeddings: if True, also return bag embeddings before classifier

        Returns:
            tuple: (logits, attn_weights) or (logits, attn_weights, bag_embeddings)
            - logits: (B, 1)
            - attn_weights: (B, T) or None
            - bag_embeddings (optional): (B, D) for t-SNE diagnostics
        """
        B, T, L_ecg = ecg.shape

        # Module I: instance embeddings from ECG backbone
        ecg_flat = ecg.view(B * T, 1, L_ecg)
        ecg_emb = self.ecg_encoder(ecg_flat).view(B, T, self.embed_dim)  # (B, T, embed_dim)
        ecg_emb = self.pos_encoding(ecg_emb)  # segments now know their position
        ecg_emb = self.embed_dropout(ecg_emb)  # regularize after positional encoding

        # Concatenate context features (sleep stage, apnea) per instance
        add_ons = [ecg_emb]

        if self.use_sleep_stage and sleep_stage is not None:
            stage_emb = self.stage_embed(sleep_stage)  # (B, T, sleep_embed_dim)
            add_ons.append(stage_emb)

        if self.use_apnea and apnea is not None:
            apnea_raw = apnea.unsqueeze(-1).float()  # (B, T, 1)
            add_ons.append(apnea_raw)

        combined = torch.cat(add_ons, dim=-1)  # (B, T, embed_dim + sleep_dim + apnea_dim)

        # Fuse ECG + context → embed_dim
        fused = self.fusion(combined.view(-1, combined.shape[-1])).view(B, T, self.embed_dim)
        fused = self.fusion_ln(fused)

        # Global HRV context: project and broadcast to all instances, then concatenate
        if self.use_hrv:
            hrv_emb = self.hrv_proj(hrv) if hrv is not None else torch.zeros(B, 16, device=fused.device, dtype=fused.dtype)
            hrv_broadcast = hrv_emb.unsqueeze(1).expand(B, T, -1)       # (B, T, 16)
            h_with_global = torch.cat([fused, hrv_broadcast], dim=-1)    # (B, T, embed_dim + 16)
            bag_emb, attn_weights = self.instance_attention(h_with_global, mask.bool())
        else:
            bag_emb, attn_weights = self.instance_attention(fused, mask.bool())

        # Embeddings before classifier (for diagnostics)
        embeddings_before_classifier = bag_emb  # (B, embed_dim + hrv_out_dim)

        # Final classifier
        bag_emb = self.pre_classifier_drop(bag_emb)
        out = self.bag_classifier(bag_emb)

        attn_out = attn_weights.detach() if attn_weights is not None else None
        if return_embeddings:
            return out, attn_out, embeddings_before_classifier
        return out, attn_out
