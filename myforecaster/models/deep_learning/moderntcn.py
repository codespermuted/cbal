"""
ModernTCN — A Modern Pure Convolution Structure for Time Series Analysis.

Modernizes traditional TCN with insights from ConvNeXt/modern CV:
- **Large-kernel DWConv**: depthwise separable convolution with large kernels
  for much larger effective receptive fields (ERFs)
- **ConvFFN**: two pointwise (1×1) convolutions with GELU, analogous to
  Transformer's FFN
- **Patching**: groups time steps (like PatchTST) before conv layers

Architecture per block:
    DWConv(large_kernel) → BN → ConvFFN(1×1 → GELU → 1×1) → Residual

Reference: Luo & Wang, "ModernTCN: A Modern Pure Convolution Structure for
General Time Series Analysis" (ICLR 2024, Spotlight).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from myforecaster.models import register_model
from myforecaster.models.deep_learning.base import AbstractDLModel
from myforecaster.models.deep_learning.patchtst import RevIN


# ---------------------------------------------------------------------------
# ModernTCN Block
# ---------------------------------------------------------------------------

class ModernTCNBlock(nn.Module):
    """Single ModernTCN block: DWConv + ConvFFN with residual.

    Parameters
    ----------
    d_model : int
    kernel_size : int — large kernel for DWConv (default 51)
    d_ff : int — FFN expansion dimension
    dropout : float
    """

    def __init__(self, d_model: int, kernel_size: int = 51,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        # Large-kernel depthwise conv (1D, along time axis)
        # Padding = (kernel_size - 1) // 2 for "same" output
        padding = (kernel_size - 1) // 2
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model,  # depthwise
        )
        self.bn1 = nn.BatchNorm1d(d_model)

        # ConvFFN: two pointwise (1×1) convolutions
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, L) → (B, D, L)"""
        # DWConv + residual
        residual = x
        x = self.bn1(self.dwconv(x))
        x = x + residual

        # ConvFFN + residual
        residual = x
        x = self.bn2(self.ffn(x) + residual)

        return x


# ---------------------------------------------------------------------------
# ModernTCN Network
# ---------------------------------------------------------------------------

class ModernTCNNetwork(nn.Module):
    """ModernTCN with patching + stacked conv blocks.

    Parameters
    ----------
    context_length, prediction_length : int
    d_model : int (default 128)
    d_ff : int (default 256)
    n_layers : int (default 4)
    kernel_size : int — large DWConv kernel (default 51)
    patch_len : int (default 16)
    stride : int (default 8)
    dropout : float
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_model: int = 128,
        d_ff: int = 256,
        n_layers: int = 4,
        kernel_size: int = 51,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches
        self.n_patches = (max(context_length, patch_len) - patch_len) // stride + 1
        self.pad_len = max(0, stride * (self.n_patches - 1) + patch_len - context_length)

        # RevIN
        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Stacked ModernTCN blocks
        # Clamp kernel to not exceed sequence length
        effective_kernel = min(kernel_size, self.n_patches)
        if effective_kernel % 2 == 0:
            effective_kernel -= 1
        effective_kernel = max(effective_kernel, 3)

        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, effective_kernel, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Prediction head
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_patches * d_model, prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, H)"""
        x = x.unsqueeze(-1)  # (B, L, 1)

        if self.revin is not None:
            x = self.revin(x)

        x = x.squeeze(-1)  # (B, L)

        # Pad if needed
        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0), mode="replicate")

        # Patching
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, P, patch_len)

        # Embed
        z = self.patch_embed(patches)  # (B, P, D)
        z = z.permute(0, 2, 1)  # (B, D, P) — conv format

        # Stacked ModernTCN blocks
        for block in self.blocks:
            z = block(z)

        z = z.permute(0, 2, 1)  # (B, P, D)

        # Prediction head
        pred = self.head(z)  # (B, H)

        if self.revin is not None:
            pred = pred.unsqueeze(-1)
            pred = self.revin.inverse(pred)
            pred = pred.squeeze(-1)

        return pred


# ---------------------------------------------------------------------------
# ModernTCN Model
# ---------------------------------------------------------------------------

@register_model("ModernTCN")
class ModernTCNModel(AbstractDLModel):
    """ModernTCN: Modern Pure Convolution (ICLR 2024 Spotlight).

    Other Parameters
    ----------------
    d_model : int (default 128)
    d_ff : int (default 256)
    n_layers : int (default 4)
    kernel_size : int (default 51)
    patch_len : int (default 16)
    stride : int (default 8)
    dropout : float (default 0.1)
    revin : bool (default True)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 128, "d_ff": 256, "n_layers": 4,
        "kernel_size": 51, "patch_len": 16, "stride": 8,
        "dropout": 0.1, "revin": True,
        "max_epochs": 50, "learning_rate": 5e-4, "stride": 2,
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

        return ModernTCNNetwork(
            context_length=context_length, prediction_length=prediction_length,
            d_model=self.get_hyperparameter("d_model"),
            d_ff=self.get_hyperparameter("d_ff"),
            n_layers=self.get_hyperparameter("n_layers"),
            kernel_size=self.get_hyperparameter("kernel_size"),
            patch_len=self.get_hyperparameter("patch_len"),
            stride=self.get_hyperparameter("stride"),
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
