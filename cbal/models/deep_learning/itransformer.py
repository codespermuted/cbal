"""
iTransformer — Inverted Transformers Are Effective for Time Series Forecasting.

Key insight: instead of applying attention across the **time** axis (like
standard Transformers), iTransformer applies attention across the **variate**
axis. Each variate's entire lookback window is embedded as a single token.

This means:
- Attention captures **inter-variate correlations** (like cross-correlation)
- The MLP inside each Transformer block learns **temporal patterns**
- Works in a channel-dependent manner (opposite of PatchTST's CI design)

For our univariate-per-item framework, iTransformer treats each item's
lookback window as a single-variate input. The temporal MLP still captures
time patterns effectively. For true multivariate use, multiple variates
would each become a token.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting" (ICLR 2024).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel
from cbal.models.deep_learning.patchtst import RevIN


class InvertedTransformerBlock(nn.Module):
    """Single iTransformer encoder block.

    The "inverted" part: input shape is (B, N_variates, D) where D encodes
    the full time series of each variate. Attention is across variates,
    and the FFN processes the temporal embedding.

    Parameters
    ----------
    d_model : int
        Embedding dimension (encodes the full lookback per variate).
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Attention across variates
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN on temporal embedding
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        # Pre-norm attention
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + self.dropout1(attn_out)

        # Pre-norm FFN
        x2 = self.norm2(x)
        x = x + self.ffn(x2)

        return x


class iTransformerNetwork(nn.Module):
    """iTransformer network.

    Architecture::

        Input (B, N, L) → Linear embed (B, N, D) → Transformer blocks
        → Linear project (B, N, H) → output

    Where N = number of variates, L = lookback, H = prediction horizon.

    For univariate per-item: N=1, so attention is trivial but the FFN
    still learns temporal patterns from the (L → D) embedding.

    Parameters
    ----------
    context_length : int
        Lookback window L.
    prediction_length : int
        Forecast horizon H.
    n_variates : int
        Number of variates N (default 1 for channel-independent).
    d_model : int
        Embedding dimension (default 256).
    n_heads : int
        Number of attention heads (default 8).
    n_layers : int
        Number of Transformer blocks (default 2).
    d_ff : int
        Feed-forward dimension (default 512).
    dropout : float
        Dropout rate (default 0.1).
    revin : bool
        Use RevIN normalization (default True).
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_variates: int = 1,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_variates = n_variates

        # RevIN
        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Variate embedding: project each variate's lookback to d_model
        self.embed = nn.Linear(context_length, d_model)

        # Transformer encoder blocks (attention across variates)
        self.blocks = nn.ModuleList([
            InvertedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Projection head: d_model → prediction_length per variate
        self.head = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, L) for univariate, or (B, L, N) for multivariate.

        Returns
        -------
        Tensor (B, H) for univariate, or (B, H, N) for multivariate.
        """
        # Handle univariate input
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)
        B, L, N = x.shape

        # RevIN normalize
        if self.revin is not None:
            x = self.revin(x)  # (B, L, N)

        # Invert: (B, L, N) → (B, N, L) — each variate is a token
        x = x.permute(0, 2, 1)  # (B, N, L)

        # Embed each variate's full lookback into d_model
        x = self.embed(x)  # (B, N, D)

        # Transformer blocks — attention across variates
        for block in self.blocks:
            x = block(x)  # (B, N, D)

        x = self.norm(x)

        # Project to prediction horizon
        pred = self.head(x)  # (B, N, H)

        # Back to (B, H, N)
        pred = pred.permute(0, 2, 1)  # (B, H, N)

        # RevIN denormalize
        if self.revin is not None:
            pred = self.revin.inverse(pred)

        # Squeeze for univariate
        if N == 1:
            pred = pred.squeeze(-1)  # (B, H)

        return pred


@register_model("iTransformer")
class iTransformerModel(AbstractDLModel):
    """iTransformer: Inverted Transformer for time series forecasting.

    Applies attention across variates (not time steps). The MLP within
    each Transformer block learns temporal patterns from the embedded
    lookback window.

    Other Parameters
    ----------------
    d_model : int
        Embedding dimension (default 256).
    n_heads : int
        Attention heads (default 8).
    n_layers : int
        Transformer blocks (default 2).
    d_ff : int
        Feed-forward dimension (default 512).
    dropout : float
        Dropout rate (default 0.1).
    revin : bool
        Use RevIN (default True).
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "revin": True,
        "max_epochs": 50,
        "learning_rate": 1e-4,
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

        return iTransformerNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            n_variates=1,  # channel-independent per item
            d_model=self.get_hyperparameter("d_model"),
            n_heads=self.get_hyperparameter("n_heads"),
            n_layers=self.get_hyperparameter("n_layers"),
            d_ff=self.get_hyperparameter("d_ff"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        past = self._enrich_target(batch)      # (B, L)
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
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        result = {"mean": pred, "quantiles": {}}
        for q in quantile_levels:
            result["quantiles"][q] = pred
        return result
