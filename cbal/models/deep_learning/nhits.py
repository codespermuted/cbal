"""
N-HiTS — Neural Hierarchical Interpolation for Time Series Forecasting.

Extends N-BEATS with multi-rate signal sampling and hierarchical interpolation.
Each block operates at a different temporal resolution (via MaxPool downsampling),
then interpolates its output back to the full prediction horizon.

Key ideas:
- **Multi-rate sampling**: each stack processes input at different temporal scales
- **Hierarchical interpolation**: coarse predictions are upsampled and refined
- **Expressiveness**: captures both long-term trends and short-term patterns

Reference: Challu et al., "N-HiTS: Neural Hierarchical Interpolation for
Time Series Forecasting" (AAAI 2023).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel
from cbal.models.deep_learning.patchtst import RevIN


class NHiTSBlock(nn.Module):
    """Single N-HiTS block: downsample → MLP → interpolate.

    Parameters
    ----------
    input_size : int
        Lookback length after pooling.
    output_size : int
        Number of interpolation coefficients (before upsample).
    n_theta : int
        Dimension of basis expansion coefficients.
    hidden_size : int
    n_layers : int
        MLP depth.
    pooling_kernel : int
        MaxPool kernel size for downsampling.
    dropout : float
    """

    def __init__(self, context_length, prediction_length, hidden_size=256,
                 n_layers=2, pooling_kernel=1, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.pooling_kernel = pooling_kernel

        # Downsampled input size
        self.pool_input_size = context_length // pooling_kernel

        # Per-paper: separate interpolation coefficients for backcast/forecast
        # n_theta_backcast = context_length / pooling_kernel (coarser at higher pools)
        # n_theta_forecast = prediction_length / pooling_kernel
        self.n_theta_backcast = max(self.pool_input_size, 1)
        self.n_theta_forecast = max(prediction_length // pooling_kernel, 1)

        # MaxPool for multi-rate sampling
        self.pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel,
                                  ceil_mode=True) if pooling_kernel > 1 else nn.Identity()

        # MLP stack
        layers = [nn.Linear(self.pool_input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*layers)

        # Separate basis coefficients for backcast and forecast
        self.backcast_fc = nn.Linear(hidden_size, self.n_theta_backcast)
        self.forecast_fc = nn.Linear(hidden_size, self.n_theta_forecast)

    def forward(self, x):
        """
        x : (B, L) — residual input
        Returns: backcast (B, L), forecast (B, H)
        """
        B = x.size(0)

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
        h = self.mlp(x_pool)  # (B, H_hidden)

        # Basis coefficients
        backcast_theta = self.backcast_fc(h)  # (B, n_theta)
        forecast_theta = self.forecast_fc(h)  # (B, n_theta)

        # Interpolate to full resolution
        backcast = F.interpolate(
            backcast_theta.unsqueeze(1), size=self.context_length, mode='linear', align_corners=False
        ).squeeze(1)  # (B, L)
        forecast = F.interpolate(
            forecast_theta.unsqueeze(1), size=self.prediction_length, mode='linear', align_corners=False
        ).squeeze(1)  # (B, H)

        return backcast, forecast


class NHiTSNetwork(nn.Module):
    """N-HiTS network: stacked blocks with hierarchical interpolation.

    Parameters
    ----------
    context_length, prediction_length : int
    n_stacks : int
        Number of stacks (each with different pooling rate).
    n_blocks_per_stack : int
        Blocks per stack.
    hidden_size : int
    n_mlp_layers : int
    pooling_kernels : list of int or None
        Pooling kernel per stack. Default: exponentially increasing.
    dropout : float
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_stacks: int = 3,
        n_blocks_per_stack: int = 1,
        hidden_size: int = 256,
        n_mlp_layers: int = 2,
        pooling_kernels: list[int] | None = None,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Default pooling: [1, 2, 4, ...] or user-specified
        if pooling_kernels is None:
            pooling_kernels = [min(2**i, context_length) for i in range(n_stacks)]

        self.blocks = nn.ModuleList()
        for stack_idx in range(n_stacks):
            pk = pooling_kernels[min(stack_idx, len(pooling_kernels) - 1)]
            for _ in range(n_blocks_per_stack):
                self.blocks.append(NHiTSBlock(
                    context_length=context_length,
                    prediction_length=prediction_length,
                    hidden_size=hidden_size,
                    n_layers=n_mlp_layers,
                    pooling_kernel=pk,
                    dropout=dropout,
                ))

    def forward(self, x):
        """x: (B, L) → (B, H)"""
        if self.revin is not None:
            x = x.unsqueeze(-1)
            x = self.revin(x)
            x = x.squeeze(-1)

        residual = x
        forecast = torch.zeros(x.size(0), self.blocks[0].prediction_length, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        if self.revin is not None:
            forecast = forecast.unsqueeze(-1)
            forecast = self.revin.inverse(forecast)
            forecast = forecast.squeeze(-1)

        return forecast


@register_model("N-HiTS")
class NHiTSModel(AbstractDLModel):
    """N-HiTS: Neural Hierarchical Interpolation for Time Series (AAAI 2023).

    Other Parameters
    ----------------
    n_stacks : int (default 3)
    n_blocks_per_stack : int (default 1)
    hidden_size : int (default 256)
    n_mlp_layers : int (default 2)
    pooling_kernels : list of int or None
    dropout : float (default 0.1)
    revin : bool (default True)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "n_stacks": 3, "n_blocks_per_stack": 1,
        "hidden_size": 256, "n_mlp_layers": 2,
        "pooling_kernels": None, "dropout": 0.1, "revin": True,
        "max_epochs": 50, "learning_rate": 5e-4, "stride": 2,
        "loss_type": "mse",            # "mse" or "quantile"
        "quantile_levels": (0.1, 0.5, 0.9),
    }

    def _build_network(self, context_length, prediction_length):
        self._quantile_head = None
        if self.get_hyperparameter("loss_type") == "quantile":
            from cbal.models.deep_learning.layers.distributions import QuantileOutput
            self._quantile_head = QuantileOutput(
                input_dim=1,  # per-step: (B, H, 1) → (B, H, Q)
                quantile_levels=tuple(self.get_hyperparameter("quantile_levels")),
            )

        return NHiTSNetwork(
            context_length=context_length, prediction_length=prediction_length,
            n_stacks=self.get_hyperparameter("n_stacks"),
            n_blocks_per_stack=self.get_hyperparameter("n_blocks_per_stack"),
            hidden_size=self.get_hyperparameter("hidden_size"),
            n_mlp_layers=self.get_hyperparameter("n_mlp_layers"),
            pooling_kernels=self.get_hyperparameter("pooling_kernels"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        pred = self._network(self._enrich_target(batch))
        future = batch["future_target"]
        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))
            return self._quantile_head.loss(q_preds, future)
        return F.mse_loss(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))
        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}
        return {"mean": pred, "quantiles": {q: pred for q in quantile_levels}}
