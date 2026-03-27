"""
TimesNet — Temporal 2D-Variation Modeling for General Time Series Analysis.

Core idea: use FFT to discover multi-periodicity, reshape 1D series into 2D
tensors (rows=num_periods, cols=period_length), apply 2D Inception blocks to
capture intra-period and inter-period variations simultaneously.

Architecture:
1. Embedding: Linear → d_model
2. TimesBlock × N:
   a. FFT → find top-k dominant periods
   b. For each period p: reshape (B, L, D) → (B, p, L/p, D) → 2D Conv
   c. Reshape back to 1D
   d. Adaptive aggregation weighted by FFT amplitudes
3. Linear projection → prediction

Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis" (ICLR 2023).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from myforecaster.models import register_model
from myforecaster.models.deep_learning.base import AbstractDLModel
from myforecaster.models.deep_learning.patchtst import RevIN


# ---------------------------------------------------------------------------
# Inception Block (parameter-efficient 2D conv)
# ---------------------------------------------------------------------------

class InceptionBlock(nn.Module):
    """Parameter-efficient Inception block for 2D temporal tensors.

    Per paper: uses parallel 2D convolutions with different kernel sizes,
    then concatenates and projects back. Shared across different period
    tensors for parameter efficiency.
    """

    def __init__(self, d_model: int, d_ff: int, num_kernels: int = 6):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_ff, kernel_size=(1, 2 * i + 1), padding=(0, i)),
                nn.GELU(),
                nn.BatchNorm2d(d_ff),
            )
            for i in range(num_kernels)
        ])
        self.proj = nn.Conv2d(d_ff * num_kernels, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, num_periods, period_len) → (B, D, num_periods, period_len)"""
        outs = [conv(x) for conv in self.convs]
        # Truncate to match smallest spatial size
        min_h = min(o.size(2) for o in outs)
        min_w = min(o.size(3) for o in outs)
        outs = [o[:, :, :min_h, :min_w] for o in outs]
        out = torch.cat(outs, dim=1)
        return self.proj(out)


# ---------------------------------------------------------------------------
# TimesBlock
# ---------------------------------------------------------------------------

class TimesBlock(nn.Module):
    """Single TimesBlock: FFT period discovery → 2D reshape → Inception → aggregate.

    Parameters
    ----------
    seq_len : int
    d_model : int
    d_ff : int
    top_k : int — number of dominant periods to extract
    num_kernels : int — inception kernel variants
    """

    def __init__(self, seq_len: int, d_model: int, d_ff: int = 64,
                 top_k: int = 5, num_kernels: int = 6):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.d_model = d_model

        # Shared Inception block (parameter-efficient: same params for all periods)
        self.inception = InceptionBlock(d_model, d_ff, num_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape

        # --- FFT to find top-k periods ---
        # Average across channels for period detection
        x_freq = torch.fft.rfft(x.mean(dim=-1), dim=1)  # (B, L//2+1)
        amplitudes = x_freq.abs().mean(dim=0)  # (L//2+1,) — average across batch
        amplitudes[0] = 0  # remove DC component

        # Top-k frequencies → corresponding periods
        _, top_indices = torch.topk(amplitudes, min(self.top_k, len(amplitudes) - 1))
        # Period = L / frequency_index
        periods = (L / (top_indices.float() + 1e-6)).clamp(min=2).long()
        periods = periods.unique()[:self.top_k]  # deduplicate

        # Get per-batch amplitudes for weighting
        batch_amps = x_freq.abs()[:, top_indices[:len(periods)]]  # (B, k)
        weights = F.softmax(batch_amps, dim=-1)  # (B, k)

        # --- For each period, reshape to 2D, apply Inception, reshape back ---
        aggregated = torch.zeros_like(x)

        for i, p in enumerate(periods):
            p = p.item()
            if p < 2 or p > L:
                continue

            # Pad to make L divisible by period
            num_periods = math.ceil(L / p)
            pad_len = num_periods * p - L
            x_padded = F.pad(x, (0, 0, 0, pad_len))  # (B, num_periods*p, D)

            # Reshape: (B, num_periods, p, D) → (B, D, num_periods, p)
            x_2d = x_padded.reshape(B, num_periods, p, D).permute(0, 3, 1, 2)

            # 2D Inception
            out_2d = self.inception(x_2d)  # (B, D, num_periods, p)

            # Reshape back: (B, D, num_periods, p) → (B, num_periods*p, D)
            out_1d = out_2d.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :L, :]

            # Weight by amplitude
            w = weights[:, i].unsqueeze(1).unsqueeze(2) if i < weights.size(1) else 1.0
            aggregated = aggregated + w * out_1d

        return aggregated


# ---------------------------------------------------------------------------
# TimesNet Network
# ---------------------------------------------------------------------------

class TimesNetNetwork(nn.Module):
    """Stacked TimesBlocks with embedding and linear head.

    Parameters
    ----------
    context_length, prediction_length : int
    d_model : int (default 64)
    d_ff : int (default 64)
    n_layers : int (default 2)
    top_k : int (default 5)
    num_kernels : int (default 6)
    dropout : float
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_model: int = 64,
        d_ff: int = 64,
        n_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Input embedding
        self.input_proj = nn.Linear(1, d_model)

        # Stacked TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(context_length, d_model, d_ff, top_k, num_kernels)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        # Prediction head
        self.head = nn.Linear(context_length * d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, H)"""
        x = x.unsqueeze(-1)  # (B, L, 1)

        if self.revin is not None:
            x = self.revin(x)

        x = self.input_proj(x)  # (B, L, D)

        # TimesBlocks with residual
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + self.dropout(block(x)))

        # Flatten and project
        pred = self.head(x.flatten(1))  # (B, H)

        if self.revin is not None:
            pred = pred.unsqueeze(-1)
            pred = self.revin.inverse(pred)
            pred = pred.squeeze(-1)

        return pred


# ---------------------------------------------------------------------------
# TimesNet Model
# ---------------------------------------------------------------------------

@register_model("TimesNet")
class TimesNetModel(AbstractDLModel):
    """TimesNet: Temporal 2D-Variation Modeling (ICLR 2023).

    Other Parameters
    ----------------
    d_model : int (default 64)
    d_ff : int (default 64)
    n_layers : int (default 2)
    top_k : int (default 5)
    num_kernels : int (default 6)
    dropout : float (default 0.1)
    revin : bool (default True)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 64, "d_ff": 64, "n_layers": 2,
        "top_k": 5, "num_kernels": 6,
        "dropout": 0.1, "revin": True,
        "max_epochs": 50, "learning_rate": 1e-3, "stride": 2,
        "loss_type": "mse",            # "mse" or "quantile"
        "quantile_levels": (0.1, 0.5, 0.9),
    }

    def _build_network(self, context_length, prediction_length):
        self._quantile_head = None
        if self.get_hyperparameter("loss_type") == "quantile":
            from myforecaster.models.deep_learning.layers.distributions import QuantileOutput
            self._quantile_head = QuantileOutput(
                input_dim=1,
                quantile_levels=tuple(self.get_hyperparameter("quantile_levels")),
            )

        return TimesNetNetwork(
            context_length=context_length, prediction_length=prediction_length,
            d_model=self.get_hyperparameter("d_model"),
            d_ff=self.get_hyperparameter("d_ff"),
            n_layers=self.get_hyperparameter("n_layers"),
            top_k=self.get_hyperparameter("top_k"),
            num_kernels=self.get_hyperparameter("num_kernels"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        pred = self._network(self._enrich_target(batch))  # (B, H)
        future = batch["future_target"]

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        return F.mse_loss(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))  # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        return {"mean": pred, "quantiles": {q: pred for q in quantile_levels}}
