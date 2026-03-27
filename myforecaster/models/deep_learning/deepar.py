"""
DeepAR — Autoregressive recurrent network for probabilistic forecasting.

Key features (from the original paper):
- **Auto-scaling**: divide inputs by item mean, multiply outputs back.
  Handles widely-varying scales across time series.
- **Lagged features**: automatically feed y_{t-1}, y_{t-7}, y_{t-14}... as
  additional inputs to capture seasonal patterns.
- **Teacher forcing** during training, **trajectory sampling** at inference.
  Each sample path is generated independently with actual sampling from
  the learned distribution (not using the mean).
- **Probabilistic output** via Gaussian / StudentT / NegBinomial likelihood.

Reference: Salinas et al., "DeepAR: Probabilistic Forecasting with
Autoregressive Recurrent Networks" (Int. Journal of Forecasting, 2020).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from myforecaster.models import register_model
from myforecaster.models.deep_learning.base import AbstractDLModel
from myforecaster.models.deep_learning.layers.distributions import (
    get_distribution_output,
)


# ---------------------------------------------------------------------------
# Lagged feature helpers
# ---------------------------------------------------------------------------

_FREQ_LAGS = {
    # freq_tier -> list of lag indices (in time steps)
    "second":     [1, 2, 3, 60, 120, 180],            # 1-3s, 1-3min ago
    "minute":     [1, 2, 3, 60, 120, 1440],            # 1-3min, 1h, 2h, 1d
    "hourly":     [1, 2, 3, 24, 48, 168],              # 1-3h, 1d, 2d, 1w
    "daily_plus": [1, 2, 3, 7, 14, 21, 28],            # 1-3d, 1w, 2w, 3w, 4w
}


def _get_lags_for_freq(freq: str | None) -> list[int]:
    """Return appropriate lag indices based on frequency."""
    if freq is None:
        return _FREQ_LAGS["daily_plus"]
    f = freq.upper().rstrip("0123456789")
    if f in ("S",):
        return _FREQ_LAGS["second"]
    if f in ("T", "MIN"):
        return _FREQ_LAGS["minute"]
    if f in ("H", "BH"):
        return _FREQ_LAGS["hourly"]
    return _FREQ_LAGS["daily_plus"]


def _extract_lags(series: torch.Tensor, lags: list[int]) -> torch.Tensor:
    """Extract lagged values from a series.

    Parameters
    ----------
    series : (B, T) — full history available up to current step
    lags : list of int — lag offsets

    Returns
    -------
    (B, T, n_lags) — lagged values, 0.0 where unavailable
    """
    B, T = series.shape
    n_lags = len(lags)
    lagged = torch.zeros(B, T, n_lags, device=series.device, dtype=series.dtype)
    for i, lag in enumerate(lags):
        if lag < T:
            lagged[:, lag:, i] = series[:, :T - lag]
    return lagged


# ---------------------------------------------------------------------------
# DeepAR Network
# ---------------------------------------------------------------------------

class DeepARNetwork(nn.Module):
    """DeepAR core network with auto-scaling, lags, item embedding, age, and trajectory sampling.

    Architecture per time step::

        input = [scaled_value, lag_1..k, time_features, item_embedding, age]
        h_t = LSTM(input, h_{t-1})
        (mu, sigma, ...) = DistributionHead(h_t)

    Parameters
    ----------
    hidden_size : int
    num_layers : int
    dropout : float
    n_time_features : int
    lags : list of int
    distribution : str
    n_items : int
        Number of unique items for embedding table.
    embedding_dim : int
        Dimension of item embedding vector.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_time_features: int = 5,
        lags: list[int] | None = None,
        distribution: str = "gaussian",
        n_items: int = 1,
        embedding_dim: int = 10,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lags = lags or [1, 2, 3, 7, 14]
        self.n_lags = len(self.lags)
        self.embedding_dim = embedding_dim

        # Item embedding (categorical item → learned vector, per paper section 3.4)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Input: scaled_value(1) + lags(n_lags) + time_features + item_embedding + age(1)
        lstm_input_dim = 1 + self.n_lags + n_time_features + embedding_dim + 1

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dist_head = get_distribution_output(distribution, hidden_size)

    def forward(
        self,
        past_target: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        future_target: torch.Tensor | None = None,
        item_id_index: torch.Tensor | None = None,
        past_age: torch.Tensor | None = None,
        future_age: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Training forward pass with teacher forcing.

        Returns dict with "loss" (if future_target given).
        """
        B, C = past_target.shape
        H = future_time_features.size(1)
        device = past_target.device

        # --- Auto-scaling ---
        scale = past_target.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)  # (B, 1)
        past_scaled = past_target / scale

        # --- Item embedding (static, same for all time steps) ---
        if item_id_index is not None:
            item_emb = self.item_embedding(item_id_index)  # (B, E)
        else:
            item_emb = torch.zeros(B, self.embedding_dim, device=device)
        item_emb_expanded = item_emb.unsqueeze(1)  # (B, 1, E) — broadcast over time

        # --- Age covariate (log-scaled for numerical stability) ---
        if past_age is not None:
            past_age_feat = torch.log1p(past_age).unsqueeze(-1)  # (B, C, 1)
        else:
            past_age_feat = torch.zeros(B, C, 1, device=device)

        if future_age is not None:
            future_age_feat = torch.log1p(future_age)  # (B, H)
        else:
            future_age_feat = torch.zeros(B, H, device=device)

        # --- Lagged features for encoder ---
        past_lags = _extract_lags(past_scaled, self.lags)  # (B, C, n_lags)

        # --- Encode past ---
        past_input = torch.cat([
            past_scaled.unsqueeze(-1),                          # (B, C, 1)
            past_lags,                                           # (B, C, n_lags)
            past_time_features,                                  # (B, C, F)
            item_emb_expanded.expand(-1, C, -1),                # (B, C, E)
            past_age_feat,                                       # (B, C, 1)
        ], dim=-1)

        _, (h, c_state) = self.lstm(past_input)

        # --- Decode future (teacher forcing) ---
        if future_target is not None:
            future_scaled = future_target / scale
        else:
            future_scaled = None

        full_series = past_scaled

        all_params = []
        prev_value = past_scaled[:, -1:]

        for t in range(H):
            step_lags = self._get_step_lags(full_series, self.lags)  # (B, 1, n_lags)
            time_feat = future_time_features[:, t:t+1, :]             # (B, 1, F)
            age_feat = future_age_feat[:, t:t+1].unsqueeze(-1)        # (B, 1, 1)

            step_input = torch.cat([
                prev_value.unsqueeze(-1),     # (B, 1, 1)
                step_lags,                     # (B, 1, n_lags)
                time_feat,                     # (B, 1, F)
                item_emb_expanded,             # (B, 1, E)
                age_feat,                      # (B, 1, 1)
            ], dim=-1)

            out, (h, c_state) = self.lstm(step_input, (h, c_state))
            params = self.dist_head(out)
            all_params.append(params)

            if future_scaled is not None:
                prev_value = future_scaled[:, t:t+1]
                full_series = torch.cat([full_series, prev_value], dim=1)
            else:
                prev_value = self.dist_head.mean(params)
                full_series = torch.cat([full_series, prev_value], dim=1)

        result = {"all_params": all_params, "scale": scale}

        if future_target is not None:
            total_loss = torch.tensor(0.0, device=device)
            for t, params in enumerate(all_params):
                target_t = future_scaled[:, t]
                squeezed = tuple(p.squeeze(1) for p in params)
                total_loss = total_loss + self.dist_head.loss(squeezed, target_t)
            result["loss"] = total_loss / H

        return result

    def _get_step_lags(self, full_series: torch.Tensor, lags: list[int]) -> torch.Tensor:
        """Get lag values at the current last position of full_series."""
        B, T = full_series.shape
        lag_vals = []
        for lag in lags:
            if lag <= T:
                lag_vals.append(full_series[:, T - lag])
            else:
                lag_vals.append(torch.zeros(B, device=full_series.device))
        return torch.stack(lag_vals, dim=-1).unsqueeze(1)  # (B, 1, n_lags)

    def sample_trajectories(
        self,
        past_target: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        n_samples: int = 100,
        item_id_index: torch.Tensor | None = None,
        past_age: torch.Tensor | None = None,
        future_age: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate multiple forecast trajectories via sampling.

        Each trajectory is an independent sample path where at each step
        we SAMPLE from the distribution (not use the mean).

        Returns
        -------
        samples : (n_samples, B, H) — sampled trajectories in ORIGINAL scale
        """
        B, C = past_target.shape
        H = future_time_features.size(1)
        device = past_target.device

        # Auto-scaling
        scale = past_target.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        past_scaled = past_target / scale

        # Item embedding
        if item_id_index is not None:
            item_emb = self.item_embedding(item_id_index).unsqueeze(1)  # (B, 1, E)
        else:
            item_emb = torch.zeros(B, 1, self.embedding_dim, device=device)

        # Age
        if past_age is not None:
            past_age_feat = torch.log1p(past_age).unsqueeze(-1)
        else:
            past_age_feat = torch.zeros(B, C, 1, device=device)
        if future_age is not None:
            future_age_feat = torch.log1p(future_age)
        else:
            future_age_feat = torch.zeros(B, H, device=device)

        # Encode past (once)
        past_lags = _extract_lags(past_scaled, self.lags)
        past_input = torch.cat([
            past_scaled.unsqueeze(-1),
            past_lags,
            past_time_features,
            item_emb.expand(-1, C, -1),
            past_age_feat,
        ], dim=-1)
        _, (h0, c0) = self.lstm(past_input)

        all_trajectories = []

        for s in range(n_samples):
            h, c_state = h0.clone(), c0.clone()
            full_series = past_scaled.clone()
            prev_value = past_scaled[:, -1:]
            trajectory = []

            for t in range(H):
                step_lags = self._get_step_lags(full_series, self.lags)
                time_feat = future_time_features[:, t:t+1, :]
                age_feat = future_age_feat[:, t:t+1].unsqueeze(-1)

                step_input = torch.cat([
                    prev_value.unsqueeze(-1),
                    step_lags,
                    time_feat,
                    item_emb,
                    age_feat,
                ], dim=-1)

                out, (h, c_state) = self.lstm(step_input, (h, c_state))
                params = self.dist_head(out)
                squeezed = tuple(p.squeeze(1) for p in params)

                # SAMPLE (not mean!)
                sample = self.dist_head.sample(squeezed, n_samples=1).squeeze(0)
                trajectory.append(sample)

                prev_value = sample.unsqueeze(1)
                full_series = torch.cat([full_series, prev_value], dim=1)

            traj = torch.stack(trajectory, dim=1)
            all_trajectories.append(traj)

        samples = torch.stack(all_trajectories, dim=0) * scale.unsqueeze(0)
        return samples

    def predict_quantiles(
        self,
        past_target: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        quantile_levels: Sequence[float],
        n_samples: int = 100,
        item_id_index: torch.Tensor | None = None,
        past_age: torch.Tensor | None = None,
        future_age: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate quantile forecasts via trajectory sampling."""
        samples = self.sample_trajectories(
            past_target, past_time_features, future_time_features, n_samples,
            item_id_index=item_id_index, past_age=past_age, future_age=future_age,
        )

        mean = samples.mean(dim=0)
        quantiles = {}
        for q in quantile_levels:
            quantiles[q] = torch.quantile(samples, q, dim=0)

        return {"mean": mean, "quantiles": quantiles}


# ---------------------------------------------------------------------------
# DeepAR Model
# ---------------------------------------------------------------------------

@register_model("DeepAR")
class DeepARModel(AbstractDLModel):
    """DeepAR: Autoregressive probabilistic forecasting with LSTM.

    Includes auto-scaling, lagged features, item embedding, age covariate,
    and trajectory sampling as described in the original paper.

    Other Parameters
    ----------------
    hidden_size : int
        LSTM hidden dimension (default 64).
    num_layers : int
        Number of LSTM layers (default 2).
    dropout : float
        Dropout rate (default 0.1).
    distribution : str
        ``"gaussian"``, ``"student_t"``, or ``"negative_binomial"``
        (default ``"gaussian"``).
    n_samples : int
        Number of sample trajectories for quantile estimation (default 100).
    lags : list of int or None
        Lag indices. None = auto-detect from freq.
    embedding_dim : int
        Item embedding dimension (default 10).
    """

    # DeepAR handles its own per-item mean_abs scaling internally
    _uses_own_scaling = True

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "hidden_size": 40,           # AG default: 40
        "num_layers": 2,
        "dropout": 0.1,
        "distribution": "gaussian",
        "n_samples": 100,
        "lags": None,
        "embedding_dim": 10,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "grad_clip": 1.0,
    }

    def _fit(self, train_data, val_data=None, time_limit=None):
        # Store item_id_to_idx mapping for consistent encoding across train/predict
        self._item_id_to_idx = {
            item_id: idx for idx, item_id in enumerate(sorted(train_data.item_ids))
        }
        self._n_items = len(self._item_id_to_idx)
        super()._fit(train_data, val_data, time_limit)

    def _build_network(self, context_length, prediction_length):
        lags = self.get_hyperparameter("lags")
        if lags is None:
            lags = _get_lags_for_freq(self.freq)
        lags = [l for l in lags if l <= context_length]
        if not lags:
            lags = [1]

        return DeepARNetwork(
            hidden_size=self.get_hyperparameter("hidden_size"),
            num_layers=self.get_hyperparameter("num_layers"),
            dropout=self.get_hyperparameter("dropout"),
            n_time_features=5,
            lags=lags,
            distribution=self.get_hyperparameter("distribution"),
            n_items=self._n_items,
            embedding_dim=self.get_hyperparameter("embedding_dim"),
        )

    def _train_step(self, batch):
        out = self._network(
            past_target=batch["past_target"],
            past_time_features=batch["past_time_features"],
            future_time_features=batch["future_time_features"],
            future_target=batch["future_target"],
            item_id_index=batch.get("item_id_index"),
            past_age=batch.get("past_age"),
            future_age=batch.get("future_age"),
        )
        return out["loss"]

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        return self._network.predict_quantiles(
            past_target=batch["past_target"],
            past_time_features=batch["past_time_features"],
            future_time_features=batch["future_time_features"],
            quantile_levels=quantile_levels,
            n_samples=self.get_hyperparameter("n_samples"),
            item_id_index=batch.get("item_id_index"),
            past_age=batch.get("past_age"),
            future_age=batch.get("future_age"),
        )
