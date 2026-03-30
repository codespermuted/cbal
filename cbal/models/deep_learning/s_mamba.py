"""
S-Mamba — Simple Mamba for Time Series Forecasting.

Tokenizes each variate's time points via a linear layer, then uses a
bidirectional Mamba block to capture inter-variate correlations and a
Feed-Forward Network (FFN) for temporal dependencies.

Architecture::

    Input (B, L, N) → Linear tokenize → Bidirectional Mamba (variate axis)
    → FFN (temporal axis) → Linear projection → (B, H, N)

For univariate-per-item: N=1, so the Mamba block captures temporal
patterns directly (since there's only one variate "token").

Reference: Wang et al., "Is Mamba Effective for Time Series Forecasting?"
(Neurocomputing, 2024).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel, _get_loss_fn
from cbal.models.deep_learning.patchtst import RevIN
from cbal.models.deep_learning.layers.mamba import BidirectionalMambaBlock


class SMambaNetwork(nn.Module):
    """S-Mamba network.

    Parameters
    ----------
    context_length : int
    prediction_length : int
    n_variates : int
        Number of variates (default 1).
    d_model : int
        Hidden dimension (default 128).
    d_state : int
        SSM state dimension (default 16).
    n_layers : int
        Number of stacked Mamba + FFN blocks (default 2).
    d_ff : int
        FFN hidden dimension (default 256).
    dropout : float
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_variates: int = 1,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_variates = n_variates

        # RevIN
        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Variate tokenizer: project each variate's lookback to d_model
        self.tokenizer = nn.Linear(context_length, d_model)

        # Stacked blocks: Bidirectional Mamba + FFN
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "mamba": BidirectionalMambaBlock(
                    d_model=d_model, d_state=d_state,
                    expand=2, d_conv=4, dropout=dropout,
                ),
                "ffn_norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        self.norm = nn.LayerNorm(d_model)

        # Output projection: d_model → prediction_length per variate
        self.head = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L) for univariate, or (B, L, N) for multivariate.
        Returns: (B, H) or (B, H, N).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)
        B, L, N = x.shape

        # RevIN
        if self.revin is not None:
            x = self.revin(x)

        # Tokenize: (B, L, N) → (B, N, L) → embed → (B, N, D)
        tokens = self.tokenizer(x.permute(0, 2, 1))  # (B, N, D)

        # Stacked Mamba + FFN blocks
        for block in self.blocks:
            # Bidirectional Mamba across variates
            tokens = block["mamba"](tokens)  # (B, N, D)
            # FFN
            residual = tokens
            tokens = block["ffn"](block["ffn_norm"](tokens)) + residual

        tokens = self.norm(tokens)  # (B, N, D)

        # Project to prediction horizon
        pred = self.head(tokens)  # (B, N, H)
        pred = pred.permute(0, 2, 1)  # (B, H, N)

        # RevIN denormalize
        if self.revin is not None:
            pred = self.revin.inverse(pred)

        if N == 1:
            pred = pred.squeeze(-1)  # (B, H)

        return pred


@register_model("S-Mamba")
class SMambaModel(AbstractDLModel):
    """S-Mamba: Simple Mamba for time series forecasting.

    Other Parameters
    ----------------
    d_model : int
        Hidden dimension (default 128).
    d_state : int
        SSM state dimension (default 16).
    n_layers : int
        Number of Mamba + FFN blocks (default 2).
    d_ff : int
        FFN hidden dim (default 256).
    dropout : float
    revin : bool
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 128,
        "d_state": 16,
        "n_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "revin": True,
        "max_epochs": 50,
        "learning_rate": 1e-3,
    }

    def _build_network(self, context_length, prediction_length):
        return SMambaNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            n_variates=1,
            d_model=self.get_hyperparameter("d_model"),
            d_state=self.get_hyperparameter("d_state"),
            n_layers=self.get_hyperparameter("n_layers"),
            d_ff=self.get_hyperparameter("d_ff"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        past = self._enrich_target(batch)
        future = batch["future_target"]
        pred = self._network(past)
        loss_fn = _get_loss_fn(self.get_hyperparameter("loss_type"))
        return loss_fn(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))
        result = {"mean": pred, "quantiles": {}}
        for q in quantile_levels:
            result["quantiles"][q] = pred
        return result
