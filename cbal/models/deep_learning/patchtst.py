"""
PatchTST — A Time Series is Worth 64 Words.

Segments each univariate time series into patches, feeds the resulting
sequence of patch tokens into a Transformer encoder, and outputs the
full prediction horizon via a linear head.

Key design choices (from the paper):
- **Patching**: groups of P consecutive time steps → 1 token (reduces
  sequence length quadratically and retains local semantics).
- **Channel-independence (CI)**: each variate is processed independently
  through the same shared Transformer backbone. No cross-variate attention.
- **RevIN** (Reversible Instance Normalization): normalizes input per-instance
  to handle distribution shift, then denormalizes predictions.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers" (ICLR 2023).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel


# ---------------------------------------------------------------------------
# RevIN — Reversible Instance Normalization
# ---------------------------------------------------------------------------
class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).

    Normalizes each sample independently (zero mean, unit variance),
    then denormalizes predictions to restore the original scale.
    Learnable affine parameters (gamma, beta) are optional.
    """

    def __init__(self, num_features: int = 1, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize. x: (B, L, C) or (B, L)."""
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize. x: (B, H, C) or (B, H)."""
        if self.affine:
            x = (x - self.beta) / self.gamma
        return x * self._std + self._mean


# ---------------------------------------------------------------------------
# PatchTST Network
# ---------------------------------------------------------------------------
class PatchTSTNetwork(nn.Module):
    """PatchTST core network.

    Architecture::

        Input (B, L) → RevIN → Patch (B, N, P) → Linear embed (B, N, D)
        → + PositionalEncoding → TransformerEncoder → Flatten → Linear → (B, H)

    Parameters
    ----------
    context_length : int
        Look-back window size L.
    prediction_length : int
        Forecast horizon H.
    patch_len : int
        Length of each patch P. Default 16.
    stride : int
        Stride between consecutive patches. Default 8.
    d_model : int
        Transformer hidden dimension. Default 128.
    n_heads : int
        Number of attention heads. Default 4.
    n_layers : int
        Number of Transformer encoder layers. Default 2.
    d_ff : int
        Feed-forward dimension. Default 256.
    dropout : float
        Dropout rate. Default 0.2.
    revin : bool
        Whether to use RevIN. Default True.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches (with potential padding)
        self.n_patches = (max(context_length, patch_len) - patch_len) // stride + 1
        # Pad length so last patch is complete
        self.pad_len = max(0, stride * (self.n_patches - 1) + patch_len - context_length)

        # RevIN
        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Patch embedding: Linear(P → D)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.dropout_embed = nn.Dropout(dropout)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head: flatten all patch representations → linear → predictions
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),  # (B, N*D)
            nn.Linear(self.n_patches * d_model, prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, L)
            Past target values (channel-independent, single variate).

        Returns
        -------
        Tensor (B, H) — point forecast for the prediction horizon.
        """
        B = x.size(0)

        # RevIN normalize
        if self.revin is not None:
            x = x.unsqueeze(-1)   # (B, L, 1)
            x = self.revin(x)
            x = x.squeeze(-1)     # (B, L)

        # Pad left if needed
        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0), mode="replicate")

        # Extract patches: unfold along time dimension
        # x: (B, L+pad) → (B, N, P)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches shape: (B, N, P)

        # Patch embedding + positional encoding
        z = self.patch_embed(patches)          # (B, N, D)
        z = self.dropout_embed(z)
        z = z + self.pos_embed[:, :z.size(1), :]

        # Transformer encoder
        z = self.encoder(z)                    # (B, N, D)

        # Prediction head
        pred = self.head(z)                    # (B, H)

        # RevIN denormalize
        if self.revin is not None:
            pred = pred.unsqueeze(-1)  # (B, H, 1)
            pred = self.revin.inverse(pred)
            pred = pred.squeeze(-1)    # (B, H)

        return pred


# ---------------------------------------------------------------------------
# PatchTST Model (integrates with AbstractDLModel)
# ---------------------------------------------------------------------------
@register_model("PatchTST")
class PatchTSTModel(AbstractDLModel):
    """PatchTST: Patch-based Transformer for time series forecasting.

    Channel-independent: each time series item is processed independently
    through the same shared backbone.

    Other Parameters
    ----------------
    patch_len : int
        Patch length P (default 16).
    stride : int
        Stride between patches (default 8).
    d_model : int
        Transformer hidden dimension (default 128).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of Transformer encoder layers (default 2).
    d_ff : int
        Feed-forward dimension (default 256).
    dropout : float
        Dropout rate (default 0.2).
    revin : bool
        Use Reversible Instance Normalization (default True).
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "patch_len": 16,
        "stride": 8,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "dropout": 0.2,
        "revin": True,
        "max_epochs": 50,
        "learning_rate": 1e-4,
        "stride": 2,
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

        return PatchTSTNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            patch_len=self.get_hyperparameter("patch_len"),
            stride=self.get_hyperparameter("stride"),
            d_model=self.get_hyperparameter("d_model"),
            n_heads=self.get_hyperparameter("n_heads"),
            n_layers=self.get_hyperparameter("n_layers"),
            d_ff=self.get_hyperparameter("d_ff"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        past = self._enrich_target(batch)    # (B, L)
        future = batch["future_target"]  # (B, H)
        pred = self._network(past)       # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        return F.mse_loss(pred, future)

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
