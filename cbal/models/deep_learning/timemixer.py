"""
TimeMixer — Decomposable Multiscale Mixing for Time Series Forecasting.

Fully MLP-based. Core idea: different time scales reveal different patterns.

Architecture:
1. **Multi-scale downsampling**: AvgPool at rates [1, 2, 4, ...] to create M scales
2. **Past-Decomposable-Mixing (PDM)**: at each scale, decompose into seasonal + trend
   - Seasonal mixing: fine → coarse (bottom-up MLP aggregation)
   - Trend mixing: coarse → fine (top-down MLP with prior from coarser scales)
3. **Future-Multipredictor-Mixing (FMM)**: M separate predictors (one per scale),
   results summed to produce final forecast

Reference: Wang et al., "TimeMixer: Decomposable Multiscale Mixing for
Time Series Forecasting" (ICLR 2024).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel
from cbal.models.deep_learning.patchtst import RevIN


# ---------------------------------------------------------------------------
# Series Decomposition (moving average)
# ---------------------------------------------------------------------------

class SeriesDecomp(nn.Module):
    """Decompose into trend (moving avg) and seasonal (residual)."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x: torch.Tensor):
        """x: (B, L, C) → seasonal (B, L, C), trend (B, L, C)"""
        # AvgPool1d expects (B, C, L)
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)[:, :x.size(1), :]
        seasonal = x - trend
        return seasonal, trend


# ---------------------------------------------------------------------------
# PDM: Past-Decomposable-Mixing
# ---------------------------------------------------------------------------

class PDMBlock(nn.Module):
    """Past-Decomposable-Mixing block.

    Applies decomposition at each scale, then:
    - Seasonal mixing: bottom-up (fine → coarse), aggregates microscopic info
    - Trend mixing: top-down (coarse → fine), injects macroscopic prior

    Parameters
    ----------
    n_channels : int
    d_model : int
    n_scales : int
    scale_lengths : list of int — sequence length at each scale
    kernel_size : int — decomposition kernel
    dropout : float
    """

    def __init__(self, n_channels: int, d_model: int, n_scales: int,
                 scale_lengths: list[int], kernel_size: int = 25,
                 dropout: float = 0.1):
        super().__init__()
        self.n_scales = n_scales
        self.decomps = nn.ModuleList([SeriesDecomp(kernel_size) for _ in range(n_scales)])

        # Bottom-up seasonal mixing: MLP from finer scale to coarser
        # At scale s, mix seasonal[s] with seasonal[s-1] (upsampled)
        self.seasonal_mixers = nn.ModuleList()
        for s in range(1, n_scales):
            self.seasonal_mixers.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # Top-down trend mixing: MLP from coarser scale to finer
        self.trend_mixers = nn.ModuleList()
        for s in range(n_scales - 1):
            self.trend_mixers.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

    def forward(self, x_scales: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        x_scales : list of (B, L_s, D) for each scale s
        Returns : list of mixed representations at each scale
        """
        # Decompose each scale
        seasonals = []
        trends = []
        for s, x_s in enumerate(x_scales):
            sea, tre = self.decomps[s](x_s)
            seasonals.append(sea)
            trends.append(tre)

        # Bottom-up seasonal mixing (fine → coarse)
        for s in range(1, self.n_scales):
            # Downsample finer seasonal to match coarser length
            finer = F.adaptive_avg_pool1d(
                seasonals[s - 1].permute(0, 2, 1),
                seasonals[s].size(1)
            ).permute(0, 2, 1)
            seasonals[s] = seasonals[s] + self.seasonal_mixers[s - 1](finer)

        # Top-down trend mixing (coarse → fine)
        for s in range(self.n_scales - 2, -1, -1):
            # Upsample coarser trend to match finer length
            coarser = F.interpolate(
                trends[s + 1].permute(0, 2, 1),
                size=trends[s].size(1), mode='linear', align_corners=False
            ).permute(0, 2, 1)
            trends[s] = trends[s] + self.trend_mixers[s](coarser)

        # Combine seasonal + trend at each scale
        outputs = [seasonals[s] + trends[s] for s in range(self.n_scales)]
        return outputs


# ---------------------------------------------------------------------------
# FMM: Future-Multipredictor-Mixing
# ---------------------------------------------------------------------------

class FMMBlock(nn.Module):
    """Future-Multipredictor-Mixing: ensemble of scale-specific predictors.

    Each scale has its own linear predictor mapping L_s → H.
    Final prediction = sum of all scale predictions.
    """

    def __init__(self, prediction_length: int, d_model: int,
                 scale_lengths: list[int], dropout: float = 0.1):
        super().__init__()
        self.predictors = nn.ModuleList()
        for L_s in scale_lengths:
            self.predictors.append(nn.Sequential(
                nn.Linear(L_s, prediction_length),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(prediction_length, prediction_length),
            ))

    def forward(self, x_scales: list[torch.Tensor]) -> torch.Tensor:
        """
        x_scales : list of (B, L_s, D)
        Returns : (B, H, D)
        """
        preds = []
        for s, x_s in enumerate(x_scales):
            # (B, D, L_s) → (B, D, H) → (B, H, D)
            p = self.predictors[s](x_s.permute(0, 2, 1)).permute(0, 2, 1)
            preds.append(p)
        return sum(preds)


# ---------------------------------------------------------------------------
# TimeMixer Network
# ---------------------------------------------------------------------------

class TimeMixerNetwork(nn.Module):
    """TimeMixer: multiscale PDM blocks + FMM ensemble.

    Parameters
    ----------
    context_length, prediction_length : int
    d_model : int (default 64)
    n_scales : int (default 4)
    n_layers : int — number of PDM blocks (default 2)
    kernel_size : int — decomposition kernel (default 25)
    dropout : float
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_model: int = 64,
        n_scales: int = 4,
        n_layers: int = 2,
        kernel_size: int = 25,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_scales = n_scales

        # RevIN
        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Compute scale lengths: L, L//2, L//4, ...
        self.scale_lengths = []
        for s in range(n_scales):
            self.scale_lengths.append(max(context_length // (2 ** s), 1))

        # Input embedding per scale
        self.input_proj = nn.Linear(1, d_model)

        # Stacked PDM blocks
        self.pdm_blocks = nn.ModuleList([
            PDMBlock(1, d_model, n_scales, self.scale_lengths, kernel_size, dropout)
            for _ in range(n_layers)
        ])

        # Norm per scale
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])

        # FMM block
        self.fmm = FMMBlock(prediction_length, d_model, self.scale_lengths, dropout)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, H)"""
        x = x.unsqueeze(-1)  # (B, L, 1)

        if self.revin is not None:
            x = self.revin(x)

        # Multi-scale downsampling via AvgPool
        x_scales = []
        for s, L_s in enumerate(self.scale_lengths):
            if s == 0:
                x_scales.append(x)
            else:
                # AvgPool to downsample
                pooled = F.adaptive_avg_pool1d(
                    x.permute(0, 2, 1), L_s
                ).permute(0, 2, 1)  # (B, L_s, 1)
                x_scales.append(pooled)

        # Project to d_model
        x_scales = [self.input_proj(x_s) for x_s in x_scales]  # list of (B, L_s, D)

        # PDM blocks
        for pdm in self.pdm_blocks:
            x_scales = pdm(x_scales)

        # Normalize
        x_scales = [self.norms[s](x_scales[s]) for s in range(self.n_scales)]

        # FMM: ensemble prediction
        pred = self.fmm(x_scales)  # (B, H, D)
        pred = self.output_proj(pred).squeeze(-1)  # (B, H)

        if self.revin is not None:
            pred = pred.unsqueeze(-1)
            pred = self.revin.inverse(pred)
            pred = pred.squeeze(-1)

        return pred


# ---------------------------------------------------------------------------
# TimeMixer Model
# ---------------------------------------------------------------------------

@register_model("TimeMixer")
class TimeMixerModel(AbstractDLModel):
    """TimeMixer: Decomposable Multiscale Mixing (ICLR 2024).

    Other Parameters
    ----------------
    d_model : int (default 64)
    n_scales : int (default 4)
    n_layers : int (default 2)
    kernel_size : int (default 25)
    dropout : float (default 0.1)
    revin : bool (default True)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 64, "n_scales": 4, "n_layers": 2,
        "kernel_size": 25, "dropout": 0.1, "revin": True,
        "max_epochs": 50, "learning_rate": 5e-4, "stride": 2,
        "loss_type": "mse",            # "mse" or "quantile"
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

        return TimeMixerNetwork(
            context_length=context_length, prediction_length=prediction_length,
            d_model=self.get_hyperparameter("d_model"),
            n_scales=self.get_hyperparameter("n_scales"),
            n_layers=self.get_hyperparameter("n_layers"),
            kernel_size=self.get_hyperparameter("kernel_size"),
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
