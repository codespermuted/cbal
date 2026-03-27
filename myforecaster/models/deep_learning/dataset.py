"""
PyTorch Dataset for time series models.

Converts a ``TimeSeriesDataFrame`` into sliding-window samples suitable
for training deep learning models.

Each sample contains:
- ``past_target``: (context_length,) — historical target values
- ``future_target``: (prediction_length,) — ground truth future (for training)
- ``past_time_features``: (context_length, n_time_feat) — calendar features
- ``future_time_features``: (prediction_length, n_time_feat) — future calendar
- ``item_id``: str — series identifier
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from myforecaster.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame


class TimeSeriesDataset(Dataset):
    """Windowed dataset for deep learning time series models.

    Parameters
    ----------
    data : TimeSeriesDataFrame
        Input data with ``(item_id, timestamp)`` MultiIndex.
    context_length : int
        Number of historical time steps to use as input.
    prediction_length : int
        Number of future time steps to predict.
    freq : str or None
        Used for generating calendar features.
    mode : str
        ``"train"`` — yields random windows with future targets.
        ``"predict"`` — yields only the last window per item (no future).
    stride : int
        Step size between consecutive windows in training mode.
    item_id_to_idx : dict or None
        Mapping from item_id str → integer index. If None, built automatically.
        Pass the training dataset's mapping when creating predict/val datasets
        so item indices are consistent.
    """

    def __init__(
        self,
        data: TimeSeriesDataFrame,
        context_length: int,
        prediction_length: int,
        freq: str | None = None,
        mode: str = "train",
        stride: int = 1,
        item_id_to_idx: dict[str, int] | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.mode = mode
        self.stride = stride

        # Build or reuse item_id → integer index mapping
        if item_id_to_idx is not None:
            self.item_id_to_idx = item_id_to_idx
        else:
            self.item_id_to_idx = {
                item_id: idx for idx, item_id in enumerate(sorted(data.item_ids))
            }
        self.n_items = max(len(self.item_id_to_idx), 1)

        self._samples: list[dict[str, Any]] = []
        self._prepare(data)

    def _prepare(self, data: TimeSeriesDataFrame):
        """Extract all valid windows from the data."""
        for item_id in data.item_ids:
            item_df = data.loc[item_id]
            target = item_df[TARGET].values.astype(np.float64)
            timestamps = item_df.index.get_level_values(TIMESTAMP)
            n = len(target)

            # Item index for embedding lookup
            item_idx = self.item_id_to_idx.get(item_id, 0)

            if self.mode == "train":
                min_len = self.context_length + self.prediction_length
                if n < min_len:
                    continue
                for start in range(0, n - min_len + 1, self.stride):
                    end_ctx = start + self.context_length
                    end_fut = end_ctx + self.prediction_length
                    # Age: distance from first observation (0-indexed)
                    past_age = np.arange(start, end_ctx, dtype=np.float64)
                    future_age = np.arange(end_ctx, end_fut, dtype=np.float64)
                    self._samples.append({
                        "item_id": item_id,
                        "item_id_index": item_idx,
                        "past_target": target[start:end_ctx],
                        "future_target": target[end_ctx:end_fut],
                        "past_timestamps": timestamps[start:end_ctx],
                        "future_timestamps": timestamps[end_ctx:end_fut],
                        "past_age": past_age,
                        "future_age": future_age,
                    })
            elif self.mode == "predict":
                if n < self.context_length:
                    pad_len = self.context_length - n
                    past = np.concatenate([np.zeros(pad_len), target])
                    ts_freq = self.freq or "D"
                    pad_ts = pd.date_range(
                        end=timestamps[0] - pd.tseries.frequencies.to_offset(ts_freq),
                        periods=pad_len, freq=ts_freq
                    )
                    past_ts = pad_ts.append(timestamps)
                    past_age = np.arange(-pad_len, n, dtype=np.float64)
                    past_age = np.maximum(past_age, 0)  # clip negative
                else:
                    past = target[-self.context_length:]
                    past_ts = timestamps[-self.context_length:]
                    start_idx = n - self.context_length
                    past_age = np.arange(start_idx, n, dtype=np.float64)

                ts_freq = self.freq or "D"
                future_ts = pd.date_range(
                    start=timestamps[-1] + pd.tseries.frequencies.to_offset(ts_freq),
                    periods=self.prediction_length, freq=ts_freq
                )
                future_age = np.arange(n, n + self.prediction_length, dtype=np.float64)

                self._samples.append({
                    "item_id": item_id,
                    "item_id_index": item_idx,
                    "past_target": past,
                    "future_target": np.zeros(self.prediction_length),
                    "past_timestamps": past_ts,
                    "future_timestamps": future_ts,
                    "past_age": past_age,
                    "future_age": future_age,
                })

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self._samples[idx]

        # Extract calendar fields for cyclic embedding
        past_feats = _extract_time_features(sample["past_timestamps"])
        future_feats = _extract_time_features(sample["future_timestamps"])

        return {
            "item_id": sample["item_id"],
            "item_id_index": torch.tensor(sample["item_id_index"], dtype=torch.long),
            "past_target": torch.tensor(sample["past_target"], dtype=torch.float32),
            "future_target": torch.tensor(sample["future_target"], dtype=torch.float32),
            "past_time_features": past_feats,
            "future_time_features": future_feats,
            "past_age": torch.tensor(sample["past_age"], dtype=torch.float32),
            "future_age": torch.tensor(sample["future_age"], dtype=torch.float32),
        }


def _extract_time_features(timestamps: pd.DatetimeIndex) -> torch.Tensor:
    """Extract normalized calendar features from timestamps.

    Returns a (seq_len, 5) tensor with:
    [second_of_minute, minute_of_hour, hour_of_day, day_of_week, month_of_year]
    Each normalized to [0, 1] range.
    """
    feats = np.column_stack([
        timestamps.second / 59.0 if hasattr(timestamps, 'second') else np.zeros(len(timestamps)),
        timestamps.minute / 59.0 if hasattr(timestamps, 'minute') else np.zeros(len(timestamps)),
        timestamps.hour / 23.0,
        timestamps.dayofweek / 6.0,
        (timestamps.month - 1) / 11.0,
    ]).astype(np.float32)
    return torch.tensor(feats, dtype=torch.float32)
