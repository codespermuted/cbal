"""
Feature engineering for tabular time series models.

Converts raw time series into supervised learning features:
- Lag features (y_{t-1}, y_{t-2}, ...)
- Rolling statistics (mean, std over rolling windows)
- Date/calendar features — **freq-aware automatic encoding**:
  - cardinality ≤ 24 → one-hot
  - cardinality > 24 → ordinal

For deep learning models, use ``encoding="cyclic"`` to get sin/cos pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# Lag features
# ============================================================================

def create_lag_features(
    series: np.ndarray,
    lags: list[int],
) -> dict[str, np.ndarray]:
    """Create lag features from a 1-D series.

    Parameters
    ----------
    series : np.ndarray, shape (T,)
    lags : list of int
        Lag offsets (e.g. [1, 2, 3, 7, 14]).

    Returns
    -------
    dict mapping feature name -> array of shape (T,) with NaN for unavailable lags.
    """
    T = len(series)
    features = {}
    for lag in lags:
        feat = np.full(T, np.nan)
        if lag < T:
            feat[lag:] = series[:T - lag]
        features[f"lag_{lag}"] = feat
    return features


# ============================================================================
# Rolling features
# ============================================================================

def create_rolling_features(
    series: np.ndarray,
    windows: list[int],
    min_periods: int = 1,
) -> dict[str, np.ndarray]:
    """Create rolling mean, std, min, max features.

    All rolling stats are computed on ``shift(1)`` to avoid data leakage
    (the current value is never included in its own window).
    """
    s = pd.Series(series)
    features = {}
    for w in windows:
        roll = s.shift(1).rolling(window=w, min_periods=min_periods)
        features[f"rolling_mean_{w}"] = roll.mean().values
        features[f"rolling_std_{w}"] = roll.std().values
        features[f"rolling_min_{w}"] = roll.min().values
        features[f"rolling_max_{w}"] = roll.max().values
    return features


# ============================================================================
# Calendar / date features
# ============================================================================

# Threshold: one-hot if cardinality <= this, ordinal otherwise
ONEHOT_MAX_CARDINALITY = 24


def _onehot(values: np.ndarray, prefix: str, n_classes: int, start: int = 0) -> dict[str, np.ndarray]:
    """One-hot encode integer values into binary columns."""
    return {
        f"{prefix}_{i}": (values == i).astype(np.float32)
        for i in range(start, start + n_classes)
    }


def _cyclic(values: np.ndarray, prefix: str, period: int) -> dict[str, np.ndarray]:
    """Cyclic (sin/cos) encode for deep learning models."""
    angle = 2 * np.pi * values.astype(np.float64) / period
    return {
        f"{prefix}_sin": np.sin(angle).astype(np.float32),
        f"{prefix}_cos": np.cos(angle).astype(np.float32),
    }


def _normalized(values: np.ndarray, prefix: str, cardinality: int) -> dict[str, np.ndarray]:
    """AG-style normalized encoding: value / (num - 1) - 0.5 → range [-0.5, 0.5].

    Better than one-hot for tree models: fewer features, preserves ordinality.
    """
    denom = max(cardinality - 1, 1)
    return {prefix: (values.astype(np.float32) / denom - 0.5)}


def _encode(
    values: np.ndarray,
    prefix: str,
    cardinality: int,
    encoding: str,
    period: int | None = None,
    start: int = 0,
) -> dict[str, np.ndarray]:
    """Encode a single calendar field using the chosen strategy.

    Parameters
    ----------
    encoding : str
        ``"auto"`` — AG-style normalized [-0.5, 0.5] (default for tree models).
        ``"onehot"`` — one-hot if card ≤ 24, ordinal otherwise (legacy).
        ``"cyclic"`` — sin/cos pairs (for deep learning).
    start : int
        Starting value for one-hot range (0 for dow/hour, 1 for month).
    """
    if encoding == "cyclic":
        return _cyclic(values, prefix, period or cardinality)
    if encoding == "onehot":
        if cardinality <= ONEHOT_MAX_CARDINALITY:
            return _onehot(values, prefix, cardinality, start=start)
        return {prefix: values.astype(np.float32)}
    # encoding == "auto" → AG-style normalized (fewer features, better for trees)
    return _normalized(values, prefix, cardinality)


def create_date_features(
    timestamps: pd.DatetimeIndex,
    freq: str | None = None,
    encoding: str = "auto",
) -> dict[str, np.ndarray]:
    """Extract calendar features with freq-aware field selection.

    Which fields are extracted depends on ``freq``:

    - **Daily or coarser**: day_of_week (7), month (12), day_of_month (31), week_of_year (53)
    - **Hourly**: hour (24), day_of_week (7), month (12), day_of_month (31)
    - **Minute**: hour (24), minute (60), day_of_week (7), month (12)
    - **Second**: hour (24), minute (60), second (60), day_of_week (7), month (12)

    Encoding per field:

    - ``"auto"`` (default, for tree models): one-hot if cardinality ≤ 24, ordinal otherwise.
    - ``"cyclic"`` (for deep learning): sin/cos pairs for all fields.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
    freq : str or None
        Time series frequency. None defaults to daily.
    encoding : str
        ``"auto"`` or ``"cyclic"``.

    Returns
    -------
    dict mapping feature name -> np.ndarray of shape (T,).
    """
    features: dict[str, np.ndarray] = {}
    tier = _freq_to_tier(freq)

    dow = timestamps.dayofweek.values                        # 0-6   (card 7)
    month = timestamps.month.values                          # 1-12  (card 12)
    dom = timestamps.day.values                              # 1-31  (card 31)
    woy = timestamps.isocalendar().week.values.astype(int)   # 1-53  (card 53)

    # is_weekend — always useful, encoding-independent
    features["is_weekend"] = (dow >= 5).astype(np.float32)

    if tier == "daily_plus":
        features.update(_encode(dow, "day_of_week", 7, encoding, period=7))
        features.update(_encode(month, "month", 12, encoding, period=12, start=1))
        features.update(_encode(dom, "day_of_month", 31, encoding, period=31))
        features.update(_encode(woy, "week_of_year", 53, encoding, period=53))

    elif tier == "hourly":
        hour = timestamps.hour.values  # 0-23 (card 24)
        features.update(_encode(hour, "hour", 24, encoding, period=24))
        features.update(_encode(dow, "day_of_week", 7, encoding, period=7))
        features.update(_encode(month, "month", 12, encoding, period=12, start=1))
        features.update(_encode(dom, "day_of_month", 31, encoding, period=31))

    elif tier in ("minute", "second"):
        hour = timestamps.hour.values      # 0-23 (card 24)
        minute = timestamps.minute.values  # 0-59 (card 60)
        features.update(_encode(hour, "hour", 24, encoding, period=24))
        features.update(_encode(minute, "minute", 60, encoding, period=60))
        features.update(_encode(dow, "day_of_week", 7, encoding, period=7))
        features.update(_encode(month, "month", 12, encoding, period=12, start=1))

        if tier == "second":
            second = timestamps.second.values  # 0-59 (card 60)
            features.update(_encode(second, "second", 60, encoding, period=60))

    return features


def _freq_to_tier(freq: str | None) -> str:
    """Map a pandas frequency string to a tier."""
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


# ============================================================================
# Diff / momentum features
# ============================================================================

def create_diff_features(
    series: np.ndarray,
    lags: list[int],
) -> dict[str, np.ndarray]:
    """Create differencing features: y_t - y_{t-k} (momentum/change).

    Captures short-term trend direction and magnitude.
    Tree models benefit significantly from these.
    """
    T = len(series)
    features = {}
    for lag in lags:
        feat = np.full(T, np.nan)
        if lag < T:
            feat[lag:] = series[lag:] - series[:T - lag]
        features[f"diff_{lag}"] = feat
    return features


# ============================================================================
# Lag ratio features
# ============================================================================

def create_ratio_features(
    series: np.ndarray,
    lags: list[int],
) -> dict[str, np.ndarray]:
    """Create ratio features: y_t / y_{t-k} (relative change).

    Useful for multiplicative patterns (e.g., percentage growth).
    """
    T = len(series)
    features = {}
    for lag in lags:
        feat = np.full(T, np.nan)
        if lag < T:
            prev = series[:T - lag]
            safe_prev = np.where(np.abs(prev) > 1e-8, prev, 1e-8)
            feat[lag:] = series[lag:] / safe_prev
        features[f"ratio_{lag}"] = feat
    return features


# ============================================================================
# EWM (exponential weighted) features
# ============================================================================

def create_ewm_features(
    series: np.ndarray,
    spans: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Exponentially weighted moving average/std (shift(1) to avoid leakage).

    EWM puts more weight on recent observations — better than simple rolling
    for capturing recent regime changes.
    """
    s = pd.Series(series).shift(1)
    if spans is None:
        spans = [7, 14, 28]
    features = {}
    for span in spans:
        features[f"ewm_mean_{span}"] = s.ewm(span=span, min_periods=1).mean().values
        features[f"ewm_std_{span}"] = s.ewm(span=span, min_periods=2).std().values
    return features


# ============================================================================
# Default lags / windows
# ============================================================================

def get_default_lags(seasonal_period: int, freq: str | None = None) -> list[int]:
    """Return AG-style lag indices based on frequency.

    AutoGluon uses 16-40 lags depending on frequency, covering:
    - Short-term: 1..7
    - Seasonal: sp-1, sp, sp+1, 2*sp-1, 2*sp, 2*sp+1, ...
    - Long-term: yearly lags for daily/hourly data
    """
    lags = set(range(1, 8))  # always include 1-7

    # Seasonal neighbourhood: sp ± 1
    for mult in [1, 2, 3]:
        center = seasonal_period * mult
        for offset in [-1, 0, 1]:
            lag = center + offset
            if 1 <= lag <= 1100:
                lags.add(lag)

    # Frequency-specific long-term lags (AG-style)
    _f = (freq or "").upper().rstrip("0123456789")
    if _f in ("H", "BH", ""):
        # Hourly: add daily/weekly/monthly boundaries
        for v in [23, 24, 25, 47, 48, 49, 167, 168, 169, 335, 336, 337,
                  719, 720, 721]:
            lags.add(v)
    elif _f == "D":
        # Daily: add weekly/monthly/yearly boundaries
        for v in [13, 14, 15, 20, 21, 22, 27, 28, 29, 30, 31,
                  56, 84, 363, 364, 365, 727, 728, 729]:
            lags.add(v)
    elif _f in ("MS", "ME", "M"):
        # Monthly: 1-7, 11-13, 23-25, 35-37
        for v in [11, 12, 13, 23, 24, 25, 35, 36, 37]:
            lags.add(v)
    elif _f in ("W",):
        for v in [51, 52, 53, 103, 104, 105]:
            lags.add(v)

    return sorted(lags)


def get_default_windows(seasonal_period: int) -> list[int]:
    """Return reasonable rolling window sizes."""
    windows = [7]
    if seasonal_period > 1:
        windows.append(seasonal_period)
    if seasonal_period * 2 <= 365:
        windows.append(seasonal_period * 2)
    windows.append(28)
    return sorted(set(w for w in windows if w >= 2))


# ============================================================================
# Batch feature builder (mlforecast-style, vectorized)
# ============================================================================

def build_batch_features(
    data: "TimeSeriesDataFrame",
    lags: list[int],
    windows: list[int],
    include_date_features: bool = True,
    freq: str | None = None,
) -> pd.DataFrame:
    """Build features for ALL items at once using groupby (mlforecast-style).

    10-50x faster than per-item build_feature_matrix for multi-item datasets.
    Also efficient for single-item datasets with long series.

    Parameters
    ----------
    data : TimeSeriesDataFrame
        Input data with (item_id, timestamp) index and 'target' column.
    lags, windows : lists
    include_date_features : bool
    freq : str or None

    Returns
    -------
    pd.DataFrame with feature columns, same index as input.
    """
    # Reset to flat DataFrame for groupby operations
    df = data.reset_index()
    target = df["target"].values.astype(np.float64)
    item_ids = df["item_id"]

    features = {}

    # --- Lag features (vectorized groupby shift) ---
    grouped = df.groupby("item_id", sort=False)["target"]
    for lag in lags:
        features[f"lag_{lag}"] = grouped.shift(lag).values

    # --- Rolling features (vectorized groupby rolling) ---
    shifted = grouped.shift(1)
    for w in windows:
        roll = shifted.rolling(window=w, min_periods=1)
        features[f"rolling_mean_{w}"] = roll.mean().values
        features[f"rolling_std_{w}"] = roll.std().values
        features[f"rolling_min_{w}"] = roll.min().values
        features[f"rolling_max_{w}"] = roll.max().values

    # --- Diff features ---
    diff_lags = [l for l in lags if l <= 7] or lags[:3]
    for lag in diff_lags:
        features[f"diff_{lag}"] = (target - grouped.shift(lag).values)

    # --- EWM features ---
    for span in [7, 14, 28]:
        features[f"ewm_mean_{span}"] = shifted.ewm(span=span, min_periods=1).mean().values

    # --- Date features ---
    if include_date_features:
        timestamps = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
        features.update(create_date_features(timestamps, freq=freq))

    result = pd.DataFrame(features, index=df.index)
    return result


# ============================================================================
# Main entry point (per-item, legacy)
# ============================================================================

def build_feature_matrix(
    series: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lags: list[int],
    windows: list[int],
    include_date_features: bool = True,
    freq: str | None = None,
    date_encoding: str = "auto",
    include_diff: bool = True,
    include_ratio: bool = True,
    include_ewm: bool = True,
) -> pd.DataFrame:
    """Build a complete feature matrix for one time series.

    Parameters
    ----------
    series : np.ndarray, shape (T,)
    timestamps : pd.DatetimeIndex, shape (T,)
    lags, windows : list of int
    include_date_features : bool
    freq : str or None
        Determines which date fields to extract.
    date_encoding : str
        ``"auto"`` (tree models) or ``"cyclic"`` (deep learning).
    include_diff : bool
        Add differencing features ``y_t - y_{t-k}`` (default True).
    include_ratio : bool
        Add ratio features ``y_t / y_{t-k}`` (default True).
    include_ewm : bool
        Add exponentially weighted moving average/std (default True).

    Returns
    -------
    pd.DataFrame of shape (T, num_features).
    """
    features = {}
    features.update(create_lag_features(series, lags))
    features.update(create_rolling_features(series, windows))

    if include_diff:
        # Use a subset of lags for diffs to limit feature count
        diff_lags = [l for l in lags if l <= 7] or lags[:3]
        features.update(create_diff_features(series, diff_lags))

    if include_ratio:
        ratio_lags = [l for l in lags if l <= 7] or lags[:3]
        features.update(create_ratio_features(series, ratio_lags))

    if include_ewm:
        features.update(create_ewm_features(series))

    if include_date_features:
        features.update(create_date_features(timestamps, freq=freq, encoding=date_encoding))

    return pd.DataFrame(features, index=range(len(series)))
