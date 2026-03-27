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

from myforecaster.models import register_model
from myforecaster.models.deep_learning.base import AbstractDLModel


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
    """DLinear network: separate linear layers for trend and seasonal."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_channels: int = 1,
        kernel_size: int = 25,
        individual: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_channels = n_channels
        self.individual = individual

        self.decomp = _SeriesDecomp(kernel_size)

        if individual:
            # Per-channel linear layers
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(context_length, prediction_length) for _ in range(n_channels)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(context_length, prediction_length) for _ in range(n_channels)
            ])
        else:
            self.linear_seasonal = nn.Linear(context_length, prediction_length)
            self.linear_trend = nn.Linear(context_length, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, context_length, n_channels)
        Returns: (batch, prediction_length, n_channels)
        """
        seasonal, trend = self.decomp(x)

        if self.individual:
            seasonal_out = torch.zeros(
                x.size(0), self.prediction_length, self.n_channels, device=x.device
            )
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.n_channels):
                seasonal_out[:, :, i] = self.linear_seasonal[i](seasonal[:, :, i])
                trend_out[:, :, i] = self.linear_trend[i](trend[:, :, i])
        else:
            # (B, C, L) -> (B, C, H) -> (B, H, C)
            seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        return seasonal_out + trend_out


@register_model("DLinear")
class DLinearModel(AbstractDLModel):
    """DLinear: Decomposition + Linear forecasting model.

    Supports covariates: past_time_features (past covariates) and
    future_time_features (known covariates) are concatenated as
    additional input channels, making DLinear covariate-aware.

    Other Parameters
    ----------------
    kernel_size : int
        Moving average kernel for decomposition (default 25).
    individual : bool
        Per-channel linear layers (default True). Only matters for
        multivariate — for univariate, always individual.
    use_covariates : bool
        If True (default), include covariates as extra channels.
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "kernel_size": 25,
        "individual": True,
        "max_epochs": 50,
        "learning_rate": 1e-4,         # lower LR for stability (linear model)
        "use_covariates": True,
        "stride": 4,  # larger stride for efficiency (DLinear is tiny)
        "use_amp": False,              # DLinear is too simple for AMP, causes grad scaler issues
        "target_scaling": "standard",  # DLinear needs good scaling (no RevIN)
        "loss_type": "mse",            # "mse" or "quantile"
        "quantile_levels": (0.1, 0.5, 0.9),
    }

    def _build_network(self, context_length, prediction_length):
        # n_channels determined at first train_step (depends on data)
        # Start with 1 (target only), will be rebuilt if covariates exist
        self._n_channels = 1
        self._quantile_head = None
        if self.get_hyperparameter("loss_type") == "quantile":
            from myforecaster.models.deep_learning.layers.distributions import QuantileOutput
            self._quantile_head = QuantileOutput(
                input_dim=1,  # per-step: (B, H, 1) → (B, H, Q)
                quantile_levels=tuple(self.get_hyperparameter("quantile_levels")),
            )

        return DLinearNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            n_channels=1,
            kernel_size=self.get_hyperparameter("kernel_size"),
            individual=self.get_hyperparameter("individual"),
        )

    def _build_input(self, batch):
        """Build multi-channel input: [target, past_covariates, future_covariates].

        Returns (past_input, n_channels) where past_input is (B, C, n_channels).
        """
        past = batch["past_target"].unsqueeze(-1)  # (B, C, 1)

        if not self.get_hyperparameter("use_covariates"):
            return past

        channels = [past]

        # Add past time features (past covariates)
        if "past_time_features" in batch and batch["past_time_features"].dim() >= 2:
            ptf = batch["past_time_features"]
            if ptf.dim() == 2:
                ptf = ptf.unsqueeze(-1)
            # ptf: (B, C, n_past_features)
            if ptf.size(1) == past.size(1):
                channels.append(ptf)

        return torch.cat(channels, dim=-1)  # (B, C, 1 + n_features)

    def _maybe_rebuild_network(self, n_channels):
        """Rebuild network if channel count changed (first batch with covariates)."""
        if n_channels != self._n_channels:
            self._n_channels = n_channels
            self._network = DLinearNetwork(
                context_length=self._context_length,
                prediction_length=self.prediction_length,
                n_channels=n_channels,
                kernel_size=self.get_hyperparameter("kernel_size"),
                individual=self.get_hyperparameter("individual"),
            ).to(self._device)
            # Rebuild quantile head if needed (moves to correct device)
            if self._quantile_head is not None:
                self._quantile_head = self._quantile_head.to(self._device)

    def _train_step(self, batch):
        x = self._build_input(batch)  # (B, C, n_channels)
        self._maybe_rebuild_network(x.size(-1))

        future = batch["future_target"]  # (B, H)
        pred = self._network(x)[:, :, 0]  # (B, H) — first channel = target

        if self._quantile_head is not None:
            # Quantile loss: pred → quantile head → pinball loss
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        return nn.functional.mse_loss(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        x = self._build_input(batch)  # (B, C, n_channels)
        pred = self._network(x)[:, :, 0]  # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        result = {"mean": pred, "quantiles": {}}
        for q in quantile_levels:
            result["quantiles"][q] = pred
        return result
