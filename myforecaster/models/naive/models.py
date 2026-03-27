"""
Naive baseline models for time series forecasting.

All models produce point forecasts (``mean``) and quantile forecasts via
residual-based prediction intervals assuming normally distributed errors.

Models
------
- **NaiveModel**: Last observed value repeated.
- **SeasonalNaiveModel**: Last season's values repeated.
- **AverageModel**: Historical mean repeated.
- **SeasonalAverageModel**: Per-season historical mean.
- **DriftModel**: Linear extrapolation from first to last value.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from myforecaster.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame
from myforecaster.models.abstract_model import AbstractTimeSeriesModel
from myforecaster.models import register_model

logger = logging.getLogger(__name__)


# ============================================================================
# Shared helpers
# ============================================================================

def _build_quantiles(
    mean: np.ndarray,
    residual_std: float,
    horizon_steps: np.ndarray,
    quantile_levels: Sequence[float],
) -> dict[str, np.ndarray]:
    """Build quantile forecasts from point forecast + residual std.

    Uses the standard prediction-interval formula for expanding uncertainty::

        q(tau, h) = mean(h) + z(tau) * sigma * sqrt(h)

    where ``z(tau)`` is the normal quantile function.

    Parameters
    ----------
    mean : np.ndarray, shape (prediction_length,)
        Point forecasts.
    residual_std : float
        Estimated standard deviation of in-sample residuals.
    horizon_steps : np.ndarray, shape (prediction_length,)
        Step index 1, 2, ..., H  (used for expanding intervals).
    quantile_levels : sequence of float
        Quantile levels (e.g. [0.1, 0.5, 0.9]).

    Returns
    -------
    dict mapping column name -> np.ndarray
    """
    result = {"mean": mean}
    if residual_std < 1e-12:
        # No variability => all quantiles = mean
        for q in quantile_levels:
            result[str(q)] = mean.copy()
        return result

    for q in quantile_levels:
        z = stats.norm.ppf(q)
        spread = z * residual_std * np.sqrt(horizon_steps)
        result[str(q)] = mean + spread
    return result


def _residual_std(residuals: np.ndarray) -> float:
    """Estimate standard deviation from residuals, handling edge cases."""
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 2:
        return 0.0
    return float(np.std(residuals, ddof=1))


# ============================================================================
# NaiveModel
# ============================================================================

@register_model("Naive")
class NaiveModel(AbstractTimeSeriesModel):
    """Forecast = last observed value.

    The simplest possible baseline.  Prediction intervals expand with
    the square root of the forecast horizon, scaled by the standard
    deviation of one-step naive residuals (first differences).

    Reference: Hyndman & Athanasopoulos, *Forecasting: Principles and
    Practice*, §5.2.
    """

    def _fit(self, train_data, val_data=None, time_limit=None):
        self._last_values: dict[str, float] = {}
        self._residual_stds: dict[str, float] = {}

        for item_id in train_data.item_ids:
            series = train_data.loc[item_id][TARGET].values.astype(np.float64)
            self._last_values[item_id] = series[-1]

            # Residuals of one-step naive: e_t = y_t - y_{t-1}
            if len(series) > 1:
                residuals = np.diff(series)
                self._residual_stds[item_id] = _residual_std(residuals)
            else:
                self._residual_stds[item_id] = 0.0

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        horizon = np.arange(1, self.prediction_length + 1, dtype=np.float64)

        rows = []
        for item_id in data.item_ids:
            last_val = self._last_values.get(
                item_id, float(data.loc[item_id][TARGET].iloc[-1])
            )
            sigma = self._residual_stds.get(item_id, 0.0)
            mean = np.full(self.prediction_length, last_val)

            cols = _build_quantiles(mean, sigma, horizon, quantile_levels)
            timestamps = self._make_future_timestamps(data, item_id)

            for i in range(self.prediction_length):
                row = {ITEMID: item_id, TIMESTAMP: timestamps[i]}
                for k, v in cols.items():
                    row[k] = v[i]
                rows.append(row)

        return self._rows_to_tsdf(rows)


# ============================================================================
# SeasonalNaiveModel
# ============================================================================

@register_model("SeasonalNaive")
class SeasonalNaiveModel(AbstractTimeSeriesModel):
    """Forecast = last season's values repeated.

    For daily data with ``seasonal_period=7``, the forecast for next Monday
    is last Monday's value, etc.

    Other Parameters
    ----------------
    seasonal_period : int or None
        If ``None``, inferred from frequency.
    """

    _default_hyperparameters = {"seasonal_period": None}

    def _fit(self, train_data, val_data=None, time_limit=None):
        sp = self.get_hyperparameter("seasonal_period")
        self._seasonal_period = sp if sp is not None else self._get_seasonal_period()

        self._last_season: dict[str, np.ndarray] = {}
        self._residual_stds: dict[str, float] = {}

        for item_id in train_data.item_ids:
            series = train_data.loc[item_id][TARGET].values.astype(np.float64)
            m = self._seasonal_period

            # Store last m values (the "last season")
            if len(series) >= m:
                self._last_season[item_id] = series[-m:]
            else:
                self._last_season[item_id] = series.copy()

            # Seasonal naive residuals: e_t = y_t - y_{t-m}
            if len(series) > m:
                residuals = series[m:] - series[:-m]
                self._residual_stds[item_id] = _residual_std(residuals)
            else:
                self._residual_stds[item_id] = 0.0

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        m = self._seasonal_period

        rows = []
        for item_id in data.item_ids:
            season = self._last_season.get(item_id)
            if season is None:
                series = data.loc[item_id][TARGET].values.astype(np.float64)
                season = series[-m:] if len(series) >= m else series

            # Tile the season to cover prediction_length
            n_tiles = (self.prediction_length // len(season)) + 1
            mean = np.tile(season, n_tiles)[:self.prediction_length]

            sigma = self._residual_stds.get(item_id, 0.0)
            # For seasonal naive, variance grows as k (number of full seasons ahead)
            k_seasons = np.array([(h - 1) // m + 1 for h in range(1, self.prediction_length + 1)],
                                 dtype=np.float64)
            cols = _build_quantiles(mean, sigma, k_seasons, quantile_levels)
            timestamps = self._make_future_timestamps(data, item_id)

            for i in range(self.prediction_length):
                row = {ITEMID: item_id, TIMESTAMP: timestamps[i]}
                for k, v in cols.items():
                    row[k] = v[i]
                rows.append(row)

        return self._rows_to_tsdf(rows)


# ============================================================================
# AverageModel
# ============================================================================

@register_model("Average")
class AverageModel(AbstractTimeSeriesModel):
    """Forecast = historical mean of the series.

    A simple but often competitive baseline for stationary data.
    """

    def _fit(self, train_data, val_data=None, time_limit=None):
        self._means: dict[str, float] = {}
        self._residual_stds: dict[str, float] = {}

        for item_id in train_data.item_ids:
            series = train_data.loc[item_id][TARGET].values.astype(np.float64)
            self._means[item_id] = float(np.nanmean(series))

            # Residuals: e_t = y_t - mean
            residuals = series - self._means[item_id]
            self._residual_stds[item_id] = _residual_std(residuals)

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        rows = []
        for item_id in data.item_ids:
            avg = self._means.get(item_id, float(np.nanmean(data.loc[item_id][TARGET].values)))
            sigma = self._residual_stds.get(item_id, 0.0)
            mean = np.full(self.prediction_length, avg)

            # For the mean method, prediction interval is constant (no horizon scaling)
            # σ_h = σ * sqrt(1 + 1/T)  ≈ σ for large T
            n_train = len(self._means)
            horizon = np.ones(self.prediction_length, dtype=np.float64)
            cols = _build_quantiles(mean, sigma, horizon, quantile_levels)
            timestamps = self._make_future_timestamps(data, item_id)

            for i in range(self.prediction_length):
                row = {ITEMID: item_id, TIMESTAMP: timestamps[i]}
                for k, v in cols.items():
                    row[k] = v[i]
                rows.append(row)

        return self._rows_to_tsdf(rows)


# ============================================================================
# SeasonalAverageModel
# ============================================================================

@register_model("SeasonalAverage")
class SeasonalAverageModel(AbstractTimeSeriesModel):
    """Forecast = average of all historical values for the same seasonal index.

    For daily data with ``seasonal_period=7``, the Monday forecast is the
    average of all past Mondays.

    Other Parameters
    ----------------
    seasonal_period : int or None
        If ``None``, inferred from frequency.
    """

    _default_hyperparameters = {"seasonal_period": None}

    def _fit(self, train_data, val_data=None, time_limit=None):
        sp = self.get_hyperparameter("seasonal_period")
        self._seasonal_period = sp if sp is not None else self._get_seasonal_period()

        self._seasonal_means: dict[str, np.ndarray] = {}
        self._residual_stds: dict[str, float] = {}

        for item_id in train_data.item_ids:
            series = train_data.loc[item_id][TARGET].values.astype(np.float64)
            m = self._seasonal_period

            # Compute mean for each seasonal index 0, 1, ..., m-1
            season_means = np.zeros(m)
            for s in range(m):
                values = series[s::m]
                season_means[s] = np.nanmean(values) if len(values) > 0 else 0.0
            self._seasonal_means[item_id] = season_means

            # Residuals
            n = len(series)
            fitted = np.array([season_means[t % m] for t in range(n)])
            residuals = series - fitted
            self._residual_stds[item_id] = _residual_std(residuals)

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        m = self._seasonal_period

        rows = []
        for item_id in data.item_ids:
            season_means = self._seasonal_means.get(item_id)
            if season_means is None:
                season_means = np.zeros(m)

            # Determine the seasonal index of the first forecast step
            series = data.loc[item_id][TARGET].values
            start_idx = len(series) % m

            mean = np.array([season_means[(start_idx + h) % m]
                             for h in range(self.prediction_length)])

            sigma = self._residual_stds.get(item_id, 0.0)
            horizon = np.ones(self.prediction_length, dtype=np.float64)
            cols = _build_quantiles(mean, sigma, horizon, quantile_levels)
            timestamps = self._make_future_timestamps(data, item_id)

            for i in range(self.prediction_length):
                row = {ITEMID: item_id, TIMESTAMP: timestamps[i]}
                for k, v in cols.items():
                    row[k] = v[i]
                rows.append(row)

        return self._rows_to_tsdf(rows)


# ============================================================================
# DriftModel
# ============================================================================

@register_model("Drift")
class DriftModel(AbstractTimeSeriesModel):
    """Forecast = linear extrapolation from first to last value.

    Equivalent to drawing a line from ``y_1`` to ``y_T`` and extending it.
    The drift is ``(y_T - y_1) / (T - 1)``::

        ŷ_{T+h} = y_T + h * drift

    Reference: Hyndman & Athanasopoulos, *Forecasting: Principles and
    Practice*, §5.2.
    """

    def _fit(self, train_data, val_data=None, time_limit=None):
        self._last_values: dict[str, float] = {}
        self._drifts: dict[str, float] = {}
        self._residual_stds: dict[str, float] = {}

        for item_id in train_data.item_ids:
            series = train_data.loc[item_id][TARGET].values.astype(np.float64)
            T = len(series)
            self._last_values[item_id] = series[-1]

            if T > 1:
                drift = (series[-1] - series[0]) / (T - 1)
                self._drifts[item_id] = drift

                # Fitted values: y_1 + t * drift  for t = 0, 1, ..., T-1
                fitted = series[0] + np.arange(T) * drift
                residuals = series - fitted
                self._residual_stds[item_id] = _residual_std(residuals)
            else:
                self._drifts[item_id] = 0.0
                self._residual_stds[item_id] = 0.0

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        rows = []
        for item_id in data.item_ids:
            last = self._last_values.get(item_id, float(data.loc[item_id][TARGET].iloc[-1]))
            drift = self._drifts.get(item_id, 0.0)
            sigma = self._residual_stds.get(item_id, 0.0)

            h = np.arange(1, self.prediction_length + 1, dtype=np.float64)
            mean = last + h * drift

            cols = _build_quantiles(mean, sigma, h, quantile_levels)
            timestamps = self._make_future_timestamps(data, item_id)

            for i in range(self.prediction_length):
                row = {ITEMID: item_id, TIMESTAMP: timestamps[i]}
                for k, v in cols.items():
                    row[k] = v[i]
                rows.append(row)

        return self._rows_to_tsdf(rows)



