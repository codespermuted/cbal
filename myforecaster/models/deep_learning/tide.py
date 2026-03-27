"""
TiDE — Time-series Dense Encoder.

An MLP-based encoder-decoder model for long-term time series forecasting
that avoids the quadratic cost of attention while achieving competitive
accuracy.

Architecture:
    1. Feature projection: project each time step's features to hidden dim
    2. Dense encoder: MLP that maps flattened projected past → dense vector
    3. Dense decoder: MLP that maps dense vector → flattened future predictions
    4. Temporal decoder: per-step linear projection + residual from lookback
    5. Optional: static covariate integration via concatenation

Reference:
    Das, Abhimanyu, et al. "Long-term Forecasting with TiDE:
    Time-series Dense Encoder." Transactions of Machine Learning
    Research. 2023.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from myforecaster.models.deep_learning.base import AbstractDLModel


class _ResidualBlock(nn.Module):
    """MLP block with residual connection + layer norm + dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.fc1(x))
        out = self.dropout(self.fc2(out))
        return self.ln(out + residual)


class TiDENetwork(nn.Module):
    """TiDE: encoder-decoder architecture with residual blocks.

    Parameters
    ----------
    context_length : int
        Lookback window length (L).
    prediction_length : int
        Forecast horizon (H).
    hidden_dim : int
        Hidden dimension for all MLP blocks.
    num_encoder_layers : int
        Number of residual blocks in the encoder.
    num_decoder_layers : int
        Number of residual blocks in the decoder.
    decoder_output_dim : int
        Per-step output dimension of the dense decoder (p).
    temporal_decoder_hidden : int
        Hidden dim of the per-step temporal decoder.
    dropout : float
        Dropout rate.
    feature_projection_dim : int or None
        If set, project each time step's features to this dim.
        Reduces input dimension when many covariates are present.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        decoder_output_dim: int = 32,
        temporal_decoder_hidden: int = 64,
        dropout: float = 0.1,
        feature_projection_dim: int | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.decoder_output_dim = decoder_output_dim

        # Feature projection (optional)
        self.use_feature_proj = feature_projection_dim is not None
        if self.use_feature_proj:
            self.feature_proj = nn.Linear(1, feature_projection_dim)
            encoder_input_dim = context_length * feature_projection_dim
        else:
            encoder_input_dim = context_length

        # Dense encoder: stack of residual blocks
        encoder_layers = []
        in_dim = encoder_input_dim
        for _ in range(num_encoder_layers):
            encoder_layers.append(
                _ResidualBlock(in_dim, hidden_dim, hidden_dim, dropout)
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Dense decoder: map encoded vector → H * p
        decoder_layers = []
        in_dim = hidden_dim
        for i in range(num_decoder_layers):
            out_dim = hidden_dim if i < num_decoder_layers - 1 else prediction_length * decoder_output_dim
            decoder_layers.append(
                _ResidualBlock(in_dim, hidden_dim, out_dim, dropout)
            )
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Temporal decoder: per-step projection p → 1
        self.temporal_decoder = nn.Sequential(
            nn.Linear(decoder_output_dim, temporal_decoder_hidden),
            nn.ReLU(),
            nn.Linear(temporal_decoder_hidden, 1),
        )

        # Lookback skip connection: linear from past → future
        self.lookback_skip = nn.Linear(context_length, prediction_length)

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        """
        x_past: (B, L) — past target values

        Returns: (B, H) — point predictions
        """
        B, L = x_past.shape

        # Feature projection
        if self.use_feature_proj:
            # (B, L) → (B, L, 1) → (B, L, d_proj)
            projected = self.feature_proj(x_past.unsqueeze(-1))
            encoder_input = projected.reshape(B, -1)  # (B, L * d_proj)
        else:
            encoder_input = x_past  # (B, L)

        # Encode
        encoded = self.encoder(encoder_input)  # (B, hidden_dim)

        # Decode
        decoded = self.decoder(encoded)  # (B, H * p)
        decoded = decoded.view(B, self.prediction_length, self.decoder_output_dim)

        # Temporal decoder: per-step (B, H, p) → (B, H, 1) → (B, H)
        temporal_out = self.temporal_decoder(decoded).squeeze(-1)

        # Lookback skip: direct linear from past → future
        skip = self.lookback_skip(x_past)  # (B, H)

        return temporal_out + skip


class TiDEModel(AbstractDLModel):
    """TiDE: Time-series Dense Encoder (Das et al. TMLR 2023).

    A purely MLP-based model that avoids attention overhead while
    matching or exceeding Transformer performance on long-horizon
    forecasting benchmarks.

    Other Parameters
    ----------------
    hidden_dim : int
        Hidden size for encoder/decoder blocks (default ``256``).
    num_encoder_layers : int
        Number of residual blocks in encoder (default ``2``).
    num_decoder_layers : int
        Number of residual blocks in decoder (default ``2``).
    decoder_output_dim : int
        Per-step decoder output dim before temporal decoder (default ``32``).
    temporal_decoder_hidden : int
        Hidden size in the per-step temporal decoder (default ``64``).
    dropout : float
        Dropout rate (default ``0.1``).
    feature_projection_dim : int or None
        If set, project each past time step to this dim (default ``None``).
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "hidden_dim": 256,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "decoder_output_dim": 32,
        "temporal_decoder_hidden": 64,
        "dropout": 0.1,
        "feature_projection_dim": None,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 32,
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

        return TiDENetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            hidden_dim=self.get_hyperparameter("hidden_dim"),
            num_encoder_layers=self.get_hyperparameter("num_encoder_layers"),
            num_decoder_layers=self.get_hyperparameter("num_decoder_layers"),
            decoder_output_dim=self.get_hyperparameter("decoder_output_dim"),
            temporal_decoder_hidden=self.get_hyperparameter("temporal_decoder_hidden"),
            dropout=self.get_hyperparameter("dropout"),
            feature_projection_dim=self.get_hyperparameter("feature_projection_dim"),
        )

    def _train_step(self, batch):
        past = self._enrich_target(batch)       # (B, C)
        future = batch["future_target"]   # (B, H)

        pred = self._network(past)        # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        return nn.functional.mse_loss(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        past = self._enrich_target(batch)       # (B, C)
        pred = self._network(past)        # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        result = {"mean": pred, "quantiles": {}}
        for q in quantile_levels:
            result["quantiles"][q] = pred  # deterministic — same as mean
        return result
