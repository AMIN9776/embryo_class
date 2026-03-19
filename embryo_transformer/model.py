"""
Embryo Transformer: BERT-style bidirectional encoder for stage prediction.

Architecture
────────────
Input per timestep t:
    vis_feat(t)  [D_v-dim]  – frozen Phase-A features
    time(t)      [scalar]   – absolute hours → sinusoidal embedding

Projection → d_model
Transformer Encoder (N layers, bidirectional self-attention)
    • attention mask: valid positions only attend to valid positions
    • MSP masked positions participate as keys (via mask_token) so
      neighbouring frames can attend to them for context
Linear classifier → 16 stage logits

Key biological priors encoded
──────────────────────────────
1. Bidirectional context  → short stages (t5, t6) are inferred from
   what comes before (t4) AND after (t7, t8) simultaneously.
2. Monotonic Viterbi      → optional post-processing at inference;
   enforces non-decreasing stage index (biology guarantee).
3. Sinusoidal time embed  → absolute hour encodes population-level
   prior (t2 ≈ 24 h, t4 ≈ 48 h …).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time embedding ────────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """
    Maps absolute time in hours to d_model-dim vectors.

    Uses log-spaced sinusoidal frequencies spanning the biological range
    (0.006 – 2.5 cycles/hour) then projects through a small MLP.
    """

    def __init__(self, d_model: int, max_hours: float = 160.0):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        # Frequencies: log-spaced from 1/max_hours to 1/0.4 h (Nyquist for 0.2 h grid)
        freq_lo = 2 * math.pi / max_hours
        freq_hi = 2 * math.pi / 0.4
        freqs = torch.exp(torch.linspace(math.log(freq_lo), math.log(freq_hi), half))
        self.register_buffer("freqs", freqs)  # (half,)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, time_hours: torch.Tensor) -> torch.Tensor:
        """time_hours: (B, 1, T) → returns (B, T, d_model)"""
        if time_hours.dim() == 3:
            t = time_hours.squeeze(1)       # (B, T)
        else:
            t = time_hours
        t = t.unsqueeze(-1)                 # (B, T, 1)
        args = t * self.freqs               # (B, T, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, T, d_model)
        return self.proj(emb)


# ── Main model ────────────────────────────────────────────────────────────────

class EmbryoTransformer(nn.Module):
    """
    Parameters
    ----------
    visual_input_dim : int     – dim of precomputed visual features (128)
    d_model          : int     – transformer hidden dim
    n_heads          : int     – number of attention heads
    n_layers         : int     – number of transformer encoder layers
    d_ff             : int     – feedforward dim inside each layer
    num_classes      : int     – number of embryo stages (16)
    dropout          : float
    max_time_hours   : float   – normalisation constant for time embedding
    """

    def __init__(
        self,
        visual_input_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        num_classes: int = 16,
        dropout: float = 0.1,
        max_time_hours: float = 160.0,
    ):
        super().__init__()
        self.d_model     = d_model
        self.num_classes = num_classes

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(visual_input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.time_emb = TimeEmbedding(d_model, max_time_hours)

        # Learnable token that replaces masked positions during MSP training
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-norm: more stable for deeper models
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        # ── Output classifier ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        vis_feats: torch.Tensor,
        time_series: torch.Tensor,
        valid_mask: torch.Tensor,
        msp_replace: torch.Tensor | None = None,
        msp_random_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        vis_feats        : (B, D_v, T)
        time_series      : (B, 1,   T)  absolute hours
        valid_mask       : (B, 1,   T)  1 = valid, 0 = padding
        msp_replace      : (B, T)  bool  positions to replace with mask_token
        msp_random_feats : (B, D_v, T)  shuffled features for 10% random replacement

        Returns
        -------
        logits : (B, num_classes, T)
        """
        B, D, T = vis_feats.shape
        vm = valid_mask.squeeze(1)          # (B, T)  float

        # ── Build input ───────────────────────────────────────────────────────
        x = vis_feats.permute(0, 2, 1)      # (B, T, D_v)

        # Zero out invalid (padding) positions before projection
        x = x * vm.unsqueeze(-1)

        # Apply MSP: replace masked positions' features
        if msp_random_feats is not None:
            x = x + msp_random_feats.permute(0, 2, 1) * 0.0   # no-op placeholder

        x = self.input_proj(x)              # (B, T, d_model)

        # Inject time embedding
        t_emb = self.time_emb(time_series)  # (B, T, d_model)
        x = x + t_emb

        # Replace MSP positions with learnable mask token (after time emb)
        if msp_replace is not None:
            mask_3d = msp_replace.unsqueeze(-1).float()                # (B, T, 1)
            x = x * (1.0 - mask_3d) + self.mask_token * mask_3d

        # ── Attention mask ────────────────────────────────────────────────────
        # Prevent any query from attending to padding positions as keys.
        # MSP-masked positions are NOT in this mask (they still participate as keys
        # so that surrounding valid frames can use their mask-token signal).
        key_padding_mask = (vm == 0)        # (B, T)  True = ignore as key

        # ── Transformer ───────────────────────────────────────────────────────
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # ── Classify ──────────────────────────────────────────────────────────
        logits = self.classifier(x)         # (B, T, num_classes)
        return logits.permute(0, 2, 1)      # (B, num_classes, T)

    # ── Inference helper ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        vis_feats: torch.Tensor,
        time_series: torch.Tensor,
        valid_mask: torch.Tensor,
        use_monotonic_decoding: bool = False,
    ) -> torch.Tensor:
        """
        Returns argmax predictions (B, T).
        Optionally applies monotonic Viterbi post-processing per sample.
        """
        logits = self.forward(vis_feats, time_series, valid_mask)   # (B, C, T)
        pred   = torch.argmax(logits, dim=1)                        # (B, T)

        if use_monotonic_decoding:
            log_probs = F.log_softmax(logits, dim=1).permute(0, 2, 1)  # (B, T, C)
            out = pred.clone()
            for b in range(log_probs.shape[0]):
                vm_b = valid_mask[b, 0].bool().cpu().numpy()
                lp_b = log_probs[b].cpu().float().numpy()           # (T, C)
                # Run Viterbi only on valid positions
                valid_idx = np.where(vm_b)[0]
                if len(valid_idx) > 0:
                    lp_valid = lp_b[valid_idx]                      # (T_v, C)
                    mono_pred = _monotonic_viterbi(lp_valid)        # (T_v,)
                    out[b, valid_idx] = torch.tensor(
                        mono_pred, dtype=torch.long, device=pred.device)
            pred = out

        return pred


# ── Monotonic Viterbi (pure numpy, O(T_v × C)) ───────────────────────────────

def _monotonic_viterbi(log_probs: np.ndarray) -> np.ndarray:
    """
    Viterbi decoding that enforces non-decreasing stage indices.

    Parameters
    ----------
    log_probs : (T, C) float  – log-softmax probabilities

    Returns
    -------
    pred : (T,) int  – argmax sequence with monotonic constraint
    """
    T, C = log_probs.shape
    NEG_INF = -1e9
    dp = np.full((T, C), NEG_INF, dtype=np.float64)
    bp = np.zeros((T, C), dtype=np.int32)

    dp[0] = log_probs[0]

    for t in range(1, T):
        # Cumulative max from left so transition j→c (j≤c) can be looked up in O(1)
        cum_max_val = np.full(C, NEG_INF, dtype=np.float64)
        cum_max_idx = np.zeros(C, dtype=np.int32)
        cum_max_val[0] = dp[t - 1, 0]
        cum_max_idx[0] = 0
        for j in range(1, C):
            if dp[t - 1, j] > cum_max_val[j - 1]:
                cum_max_val[j] = dp[t - 1, j]
                cum_max_idx[j] = j
            else:
                cum_max_val[j] = cum_max_val[j - 1]
                cum_max_idx[j] = cum_max_idx[j - 1]
        dp[t] = cum_max_val + log_probs[t]
        bp[t] = cum_max_idx

    # Backtrack
    pred = np.empty(T, dtype=np.int32)
    pred[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        pred[t] = bp[t + 1, pred[t + 1]]

    return pred
