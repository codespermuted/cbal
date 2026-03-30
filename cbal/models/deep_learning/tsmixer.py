"""
TSMixer — An All-MLP Architecture for Time Series Forecasting.

Alternates between two types of mixing:
- **Time-mixing MLP**: shared across channels, captures temporal patterns
- **Feature-mixing MLP**: shared across time, captures cross-variate dependencies

Simple but effective — often matches or beats Transformer models.

Reference: Chen et al., "TSMixer: An All-MLP Architecture for Time Series
Forecasting" (arXiv 2023, Google).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel, _get_loss_fn
from cbal.models.deep_learning.patchtst import RevIN


class _TSBatchNorm(nn.Module):
    """BatchNorm for 3D time series tensor (B, T, C).

    Per the TSMixer paper, BatchNorm is applied along the feature (C)
    dimension. We reshape to (B*T, C) for nn.BatchNorm1d, then back.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return self.bn(x.reshape(B * T, C)).reshape(B, T, C)


class MixerBlock(nn.Module):
    """Single TSMixer block: time-mix → feature-mix with residual + BN.

    Per paper: uses BatchNorm (not LayerNorm).
    """

    def __init__(self, seq_len: int, n_features: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Time-mixing: MLP across the time axis (shared across features)
        self.time_norm = _TSBatchNorm(n_features)
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len),
            nn.Dropout(dropout),
        )

        # Feature-mixing: MLP across the feature axis (shared across time)
        self.feat_norm = _TSBatchNorm(n_features)
        self.feat_mix = nn.Sequential(
            nn.Linear(n_features, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T, C)"""
        # Time-mixing: transpose → MLP along time → transpose back
        residual = x
        x_norm = self.time_norm(x)
        x_t = x_norm.permute(0, 2, 1)  # (B, C, T)
        x_t = self.time_mix(x_t)        # (B, C, T)
        x = residual + x_t.permute(0, 2, 1)  # (B, T, C)

        # Feature-mixing: MLP along features
        residual = x
        x = residual + self.feat_mix(self.feat_norm(x))  # (B, T, C)

        return x


class TSMixerNetwork(nn.Module):
    """TSMixer network.

    Architecture::

        Input (B, L, C) → [MixerBlock × n_layers] → Linear → (B, H, C)

    For univariate-per-item: C=1, so feature-mixing is trivial but
    time-mixing still captures temporal patterns.

    Parameters
    ----------
    context_length, prediction_length : int
    n_channels : int
    d_ff : int
        MLP hidden dimension (default 64).
    n_layers : int
        Number of mixer blocks (default 4).
    dropout : float
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_channels: int = 1,
        d_ff: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_channels = n_channels

        self.revin = RevIN(num_features=1, affine=False) if revin else None

        self.blocks = nn.ModuleList([
            MixerBlock(context_length, n_channels, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Temporal projection: L → H
        self.head = nn.Linear(context_length, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) or (B, L, C) → (B, H) or (B, H, C)"""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(-1)  # (B, L, 1)

        if self.revin is not None:
            x = self.revin(x)

        for block in self.blocks:
            x = block(x)  # (B, L, C)

        # Project time: (B, C, L) → (B, C, H) → (B, H, C)
        x = self.head(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.revin is not None:
            x = self.revin.inverse(x)

        if squeeze:
            x = x.squeeze(-1)
        return x


@register_model("TSMixer")
class TSMixerModel(AbstractDLModel):
    """TSMixer: All-MLP time+channel mixing (Google, 2023).

    Other Parameters
    ----------------
    d_ff : int (default 64)
    n_layers : int (default 4)
    dropout : float (default 0.1)
    revin : bool (default True)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_ff": 64, "n_layers": 4, "dropout": 0.1, "revin": True,
        "max_epochs": 50, "learning_rate": 1e-3,
        "quantile_levels": (0.1, 0.5, 0.9),
    }

    def _build_network(self, context_length, prediction_length):
        self._quantile_head = None
        if self.get_hyperparameter("loss_type") == "quantile":
            from cbal.models.deep_learning.layers.distributions import QuantileOutput
            self._quantile_head = QuantileOutput(
                input_dim=1,
                quantile_levels=tuple(self.get_hyperparameter("quantile_levels")),
            )

        return TSMixerNetwork(
            context_length=context_length, prediction_length=prediction_length,
            n_channels=1, d_ff=self.get_hyperparameter("d_ff"),
            n_layers=self.get_hyperparameter("n_layers"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        pred = self._network(self._enrich_target(batch))  # (B, H)
        future = batch["future_target"]

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        loss_fn = _get_loss_fn(self.get_hyperparameter("loss_type"))
        return loss_fn(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))  # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        return {"mean": pred, "quantiles": {q: pred for q in quantile_levels}}
