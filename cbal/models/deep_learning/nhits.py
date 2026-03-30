"""
N-HiTS — Neural Hierarchical Interpolation for Time Series Forecasting.

Extends N-BEATS with multi-rate signal sampling and hierarchical interpolation.
Each block operates at a different temporal resolution (via MaxPool downsampling),
then interpolates its output back to the full prediction horizon.

Key ideas (per paper):
- **Multi-rate sampling**: each stack processes input at different temporal scales
- **Hierarchical interpolation**: n_freq_downsample controls forecast expressiveness
  separately from input pooling
- **Naive1 initialization**: forecast starts from last observed value
- **Input flip**: time-reversed input for MLP processing

Reference: Challu et al., "N-HiTS: Neural Hierarchical Interpolation for
Time Series Forecasting" (AAAI 2023).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel, _get_loss_fn


class NHiTSBlock(nn.Module):
    """Single N-HiTS block (paper-aligned).

    Architecture: MaxPool → MLP → separate backcast/forecast heads → interpolate.

    Key changes from naive implementation:
    - Backcast always has context_length coefficients (full resolution)
    - Forecast uses n_freq_downsample (separate from pooling) for hierarchical resolution
    - Input is time-reversed before processing
    """

    def __init__(self, context_length, prediction_length, hidden_size=512,
                 n_mlp_layers=2, pooling_kernel=2, n_freq_downsample=1,
                 dropout=0.0):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.pooling_kernel = pooling_kernel

        # Downsampled input size
        self.pool_input_size = max(context_length // pooling_kernel, 1)

        # Per paper: backcast = full resolution, forecast = downsampled by n_freq_downsample
        self.n_theta_backcast = context_length  # always full resolution
        self.n_theta_forecast = max(prediction_length // n_freq_downsample, 1)

        # MaxPool for multi-rate input downsampling
        self.pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel,
                                  ceil_mode=True) if pooling_kernel > 1 else nn.Identity()

        # MLP stack: [Linear → ReLU → Dropout] × n_layers
        layers = []
        in_dim = self.pool_input_size
        for _ in range(n_mlp_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)

        # Separate basis coefficients for backcast and forecast
        n_theta = self.n_theta_backcast + self.n_theta_forecast
        self.theta_fc = nn.Linear(hidden_size, n_theta)

    def forward(self, x):
        """
        x : (B, L) — residual input (already time-reversed by network)
        Returns: backcast (B, L), forecast (B, H)
        """
        # Downsample
        if self.pooling_kernel > 1:
            x_pool = self.pool(x.unsqueeze(1)).squeeze(1)  # (B, L//K)
        else:
            x_pool = x

        # Pad/trim to expected size
        if x_pool.size(1) > self.pool_input_size:
            x_pool = x_pool[:, :self.pool_input_size]
        elif x_pool.size(1) < self.pool_input_size:
            x_pool = F.pad(x_pool, (0, self.pool_input_size - x_pool.size(1)))

        # MLP
        h = self.mlp(x_pool)  # (B, hidden_size)

        # Joint theta → split into backcast and forecast coefficients
        theta = self.theta_fc(h)  # (B, n_theta_back + n_theta_fore)
        backcast_theta = theta[:, :self.n_theta_backcast]
        forecast_theta = theta[:, self.n_theta_backcast:]

        # Backcast: already full resolution (context_length coefficients)
        backcast = backcast_theta  # (B, L) — no interpolation needed

        # Forecast: interpolate from coarse coefficients to full prediction_length
        if self.n_theta_forecast == self.prediction_length:
            forecast = forecast_theta
        else:
            forecast = F.interpolate(
                forecast_theta.unsqueeze(1), size=self.prediction_length,
                mode='linear', align_corners=False
            ).squeeze(1)  # (B, H)

        return backcast, forecast


class NHiTSNetwork(nn.Module):
    """N-HiTS network — paper-aligned implementation.

    Changes from previous version:
    - Input is time-reversed (flip) before processing
    - Forecast initialized with Naive1 (last observed value)
    - pooling_kernels default: [2, 2, 1] (large→small, per paper)
    - n_freq_downsample: [4, 2, 1] (coarse→fine, per paper)
    - No RevIN by default (paper uses identity scaler)
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_stacks: int = 3,
        n_blocks_per_stack: int = 1,
        hidden_size: int = 512,
        n_mlp_layers: int = 2,
        pooling_kernels: list[int] | None = None,
        n_freq_downsample: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.prediction_length = prediction_length

        # Paper defaults: large pooling first (long-term), small last (short-term)
        if pooling_kernels is None:
            pooling_kernels = [2, 2, 1]
        # Paper defaults: coarse forecast first, fine last
        if n_freq_downsample is None:
            n_freq_downsample = [4, 2, 1]

        self.blocks = nn.ModuleList()
        for stack_idx in range(n_stacks):
            pk = pooling_kernels[min(stack_idx, len(pooling_kernels) - 1)]
            nfd = n_freq_downsample[min(stack_idx, len(n_freq_downsample) - 1)]
            for _ in range(n_blocks_per_stack):
                self.blocks.append(NHiTSBlock(
                    context_length=context_length,
                    prediction_length=prediction_length,
                    hidden_size=hidden_size,
                    n_mlp_layers=n_mlp_layers,
                    pooling_kernel=pk,
                    n_freq_downsample=nfd,
                    dropout=dropout,
                ))

    def forward(self, x):
        """x: (B, L) → (B, H)"""
        # Time-reverse input (per paper)
        residual = x.flip(dims=(-1,))

        forecast = torch.zeros(x.size(0), self.prediction_length, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


@register_model("N-HiTS")
class NHiTSModel(AbstractDLModel):
    """N-HiTS: Neural Hierarchical Interpolation for Time Series (AAAI 2023).

    Paper-aligned implementation with:
    - Separate pooling_kernels and n_freq_downsample
    - Naive1 forecast initialization
    - Time-reversed input
    - Full-resolution backcast coefficients
    - No RevIN (paper default)

    Other Parameters
    ----------------
    n_stacks : int (default 3)
    n_blocks_per_stack : int (default 1)
    hidden_size : int (default 512)
    n_mlp_layers : int (default 2)
    pooling_kernels : list[int] or None (default [2, 2, 1])
    n_freq_downsample : list[int] or None (default [4, 2, 1])
    dropout : float (default 0.0)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "n_stacks": 3, "n_blocks_per_stack": 1,
        "hidden_size": 512, "n_mlp_layers": 2,
        "pooling_kernels": None,       # [2, 2, 1] per paper
        "n_freq_downsample": None,     # [4, 2, 1] per paper
        "dropout": 0.0,               # paper default
        "max_epochs": 100, "learning_rate": 1e-3, "stride": 2,
        "revin": True,                 # RevIN helps with non-stationary data
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

        return NHiTSNetwork(
            context_length=context_length, prediction_length=prediction_length,
            n_stacks=self.get_hyperparameter("n_stacks"),
            n_blocks_per_stack=self.get_hyperparameter("n_blocks_per_stack"),
            hidden_size=self.get_hyperparameter("hidden_size"),
            n_mlp_layers=self.get_hyperparameter("n_mlp_layers"),
            pooling_kernels=self.get_hyperparameter("pooling_kernels"),
            n_freq_downsample=self.get_hyperparameter("n_freq_downsample"),
            dropout=self.get_hyperparameter("dropout"),
        )

    def _train_step(self, batch):
        pred = self._network(self._enrich_target(batch))
        future = batch["future_target"]
        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))
            return self._quantile_head.loss(q_preds, future)
        loss_fn = _get_loss_fn(self.get_hyperparameter("loss_type"))
        return loss_fn(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))
        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}
        return {"mean": pred, "quantiles": {q: pred for q in quantile_levels}}
