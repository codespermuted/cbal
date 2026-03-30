"""
DLinear — Decomposition Linear model.

Decomposes time series into trend (moving average) and seasonal (residual)
components, then applies separate linear layers to each.

Reference: Zeng et al., "Are Transformers Effective for Time Series
Forecasting?" (AAAI 2023).

Despite its extreme simplicity, DLinear often outperforms complex
Transformer models on standard benchmarks.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel, _get_loss_fn


class _MovingAvg(nn.Module):
    """Moving average block for trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, channels) -> (batch, seq_len, channels)."""
        # AvgPool1d expects (B, C, L)
        out = self.avg(x.permute(0, 2, 1))
        return out.permute(0, 2, 1)[:, : x.size(1), :]


class _SeriesDecomp(nn.Module):
    """Decompose into trend + seasonal using moving average."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearNetwork(nn.Module):
    """DLinear network: separate linear layers for trend and seasonal.

    Operates on univariate input (B, L) — covariates handled externally
    via the base class ``_enrich_target`` additive injection.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.decomp = _SeriesDecomp(kernel_size)

        # Single-channel linear layers (univariate)
        self.linear_seasonal = nn.Linear(context_length, prediction_length)
        self.linear_trend = nn.Linear(context_length, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, context_length) — univariate enriched target
        Returns: (batch, prediction_length)
        """
        # Add channel dim for decomposition: (B, L) → (B, L, 1)
        x = x.unsqueeze(-1)
        seasonal, trend = self.decomp(x)

        # (B, L, 1) → squeeze → (B, L) → linear → (B, H)
        seasonal_out = self.linear_seasonal(seasonal.squeeze(-1))
        trend_out = self.linear_trend(trend.squeeze(-1))

        return seasonal_out + trend_out


@register_model("DLinear")
class DLinearModel(AbstractDLModel):
    """DLinear: Decomposition + Linear forecasting model.

    Covariates are handled via the base class ``_enrich_target`` additive
    injection: ``enriched = target + gate * MLP(covariates)``. This avoids
    the network-rebuild bug of the previous multi-channel approach and is
    consistent with all other DL models.

    Other Parameters
    ----------------
    kernel_size : int
        Moving average kernel for decomposition (default 25).
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "kernel_size": 25,
        "max_epochs": 100,
        "learning_rate": 1e-3,         # AG-aligned LR
        "loss_type": "mse",            # MSE provides natural gradient damping for linear models
        "stride": 4,                   # larger stride for efficiency (DLinear is tiny)
        "use_amp": False,              # DLinear is too simple for AMP
        "target_scaling": "none",      # DLinear's decomposition handles normalization
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

        return DLinearNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            kernel_size=self.get_hyperparameter("kernel_size"),
        )

    def _train_step(self, batch):
        past = self._enrich_target(batch)  # (B, L) — covariates via additive gate
        future = batch["future_target"]    # (B, H)
        pred = self._network(past)         # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))
            return self._quantile_head.loss(q_preds, future)

        loss_fn = _get_loss_fn(self.get_hyperparameter("loss_type"))
        return loss_fn(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        past = self._enrich_target(batch)
        pred = self._network(past)  # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        result = {"mean": pred, "quantiles": {}}
        for q in quantile_levels:
            result["quantiles"][q] = pred
        return result
