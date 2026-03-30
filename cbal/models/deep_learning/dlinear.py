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

    AG-aligned: includes hidden_dimension (default 20) for richer output
    projection. Operates on univariate input (B, L).
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        kernel_size: int = 25,
        hidden_dimension: int = 20,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.hidden_dimension = hidden_dimension

        self.decomp = _SeriesDecomp(kernel_size)

        # AG-style: project to prediction_length * hidden_dimension, then reduce
        if hidden_dimension > 1:
            self.linear_seasonal = nn.Linear(context_length, prediction_length * hidden_dimension)
            self.linear_trend = nn.Linear(context_length, prediction_length * hidden_dimension)
            self.output_proj = nn.Linear(hidden_dimension, 1)
        else:
            self.linear_seasonal = nn.Linear(context_length, prediction_length)
            self.linear_trend = nn.Linear(context_length, prediction_length)
            self.output_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, context_length) — univariate enriched target
        Returns: (batch, prediction_length)
        """
        # Per-window MeanScaler (AG-style): scale by mean absolute value
        scale = x.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)  # (B, 1)
        x = x / scale

        # Add channel dim for decomposition: (B, L) → (B, L, 1)
        x = x.unsqueeze(-1)
        seasonal, trend = self.decomp(x)

        # (B, L, 1) → squeeze → (B, L) → linear
        seasonal_out = self.linear_seasonal(seasonal.squeeze(-1))
        trend_out = self.linear_trend(trend.squeeze(-1))

        combined = seasonal_out + trend_out  # (B, H*D) or (B, H)

        if self.output_proj is not None:
            # (B, H*D) → (B, H, D) → linear → (B, H, 1) → (B, H)
            B = combined.size(0)
            combined = combined.view(B, self.prediction_length, self.hidden_dimension)
            combined = self.output_proj(combined).squeeze(-1)

        # Inverse scale
        return combined * scale


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
        "hidden_dimension": 20,        # AG default: 20 (richer output projection)
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "loss_type": "mse",            # MSE provides natural gradient damping for linear models
        "stride": 4,                   # larger stride for efficiency (DLinear is tiny)
        "use_amp": False,              # DLinear is too simple for AMP
        "target_scaling": "none",      # Per-window scaling is inside the network now (AG-style)
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
            hidden_dimension=self.get_hyperparameter("hidden_dimension"),
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
