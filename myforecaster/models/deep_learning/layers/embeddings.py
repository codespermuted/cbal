"""
Embeddings for deep learning time series models.

- Cyclic date embeddings (sin/cos) for calendar features
- Positional encoding for Transformer-based models
- Value embedding (linear projection of input)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class CyclicDateEmbedding(nn.Module):
    """Encode calendar features as sin/cos pairs.

    For each temporal field (hour, dow, month, ...), produces a 2D
    embedding ``[sin(2π * value / period), cos(2π * value / period)]``.
    Total output dim = 2 * number of fields.

    Parameters
    ----------
    freq : str or None
        Time series frequency — determines which fields to extract.
    embed_dim : int or None
        If given, project concatenated sin/cos features to this dim.
    """

    # field_name -> (period, accessor_attr)
    _FIELD_MAP = {
        "second":      (60,  "second"),
        "minute":      (60,  "minute"),
        "hour":        (24,  "hour"),
        "day_of_week": (7,   "dayofweek"),
        "day_of_month":(31,  "day"),
        "day_of_year": (366, "dayofyear"),
        "month":       (12,  "month"),
        "week_of_year":(53,  None),  # special: isocalendar
    }

    # Which fields to use per freq tier
    _FREQ_FIELDS = {
        "second":     ["second", "minute", "hour", "day_of_week", "month"],
        "minute":     ["minute", "hour", "day_of_week", "month"],
        "hourly":     ["hour", "day_of_week", "day_of_month", "month"],
        "daily_plus": ["day_of_week", "day_of_month", "month", "week_of_year"],
    }

    def __init__(self, freq: str | None = None, embed_dim: int | None = None):
        super().__init__()
        self.freq = freq
        tier = self._freq_to_tier(freq)
        self.fields = self._FREQ_FIELDS[tier]
        raw_dim = 2 * len(self.fields)

        self.proj = None
        if embed_dim is not None and embed_dim != raw_dim:
            self.proj = nn.Linear(raw_dim, embed_dim)

        self.output_dim = embed_dim if embed_dim is not None else raw_dim

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Parameters
        ----------
        timestamps : Tensor of shape (batch, seq_len)
            Integer-encoded timestamps (Unix epoch seconds or similar).
            **Alternative**: pass a dict of pre-extracted fields instead
            via :meth:`forward_from_fields`.

        Returns
        -------
        Tensor of shape (batch, seq_len, output_dim)
        """
        # Expect a dict {field_name: tensor (batch, seq_len)} passed via forward_from_fields
        raise NotImplementedError(
            "Use forward_from_fields() with pre-extracted calendar tensors."
        )

    def forward_from_fields(self, fields: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode from pre-extracted calendar field tensors.

        Parameters
        ----------
        fields : dict mapping field name -> Tensor (batch, seq_len)
            Values should be floats (e.g. hour=0..23).

        Returns
        -------
        Tensor of shape (batch, seq_len, output_dim)
        """
        parts = []
        for fname in self.fields:
            period, _ = self._FIELD_MAP[fname]
            val = fields[fname]  # (batch, seq_len)
            angle = 2 * math.pi * val / period
            parts.append(torch.sin(angle).unsqueeze(-1))
            parts.append(torch.cos(angle).unsqueeze(-1))

        x = torch.cat(parts, dim=-1)  # (batch, seq_len, 2*n_fields)
        if self.proj is not None:
            x = self.proj(x)
        return x

    @staticmethod
    def _freq_to_tier(freq: str | None) -> str:
        if freq is None:
            return "daily_plus"
        f = freq.upper().rstrip("0123456789")
        if f in ("S",):
            return "second"
        if f in ("T", "MIN"):
            return "minute"
        if f in ("H", "BH"):
            return "hourly"
        return "daily_plus"


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformers.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq_len, d_model)."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ValueEmbedding(nn.Module):
    """Project scalar time series values to a d_model-dimensional space.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (1 for univariate).
    d_model : int
        Target embedding dimension.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)."""
        return self.proj(x)
