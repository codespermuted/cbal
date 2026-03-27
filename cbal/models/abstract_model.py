"""
AbstractTimeSeriesModel — Base class for all forecasting models.

Every model in the Model Zoo inherits from this class and implements
``_fit()`` and ``_predict()``.  The public ``fit()`` / ``predict()`` wrappers
handle validation, timing, logging, and error handling.

Usage (implementing a custom model)::

    from cbal.models.abstract_model import AbstractTimeSeriesModel

    class MyModel(AbstractTimeSeriesModel):
        def _fit(self, train_data, val_data=None, time_limit=None):
            self._last_values = train_data.groupby("item_id")["target"].last()

        def _predict(self, data, known_covariates=None, quantile_levels=None):
            # Return a DataFrame with predictions
            ...
"""

from __future__ import annotations

import abc
import copy
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cbal.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame
from cbal.metrics.scorers import DEFAULT_METRIC, TimeSeriesScorer, get_metric

logger = logging.getLogger(__name__)


class AbstractTimeSeriesModel(abc.ABC):
    """Abstract base class for time series forecasting models.

    Parameters
    ----------
    freq : str or None
        Time series frequency (e.g. ``'D'``, ``'h'``, ``'MS'``).
        Inferred from training data if ``None``.
    prediction_length : int
        Number of future time steps to forecast.
    path : str or None
        Directory to save model artifacts.  Created on ``save()``.
    name : str or None
        Human-readable model name.  Defaults to class name.
    eval_metric : str or TimeSeriesScorer or None
        Metric used for validation scoring.  Defaults to ``MASE``.
    hyperparameters : dict or None
        Model-specific hyperparameters.
    """

    # Subclasses can set default hyperparameters here
    _default_hyperparameters: dict[str, Any] = {}

    def __init__(
        self,
        freq: str | None = None,
        prediction_length: int = 1,
        path: str | None = None,
        name: str | None = None,
        eval_metric: str | TimeSeriesScorer | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ):
        self.freq = freq
        self.prediction_length = prediction_length
        self.path = path
        self.name = name or self.__class__.__name__
        self.eval_metric = get_metric(eval_metric or DEFAULT_METRIC)

        # Merge default + user hyperparameters
        self._hyperparameters: dict[str, Any] = {
            **copy.deepcopy(self._default_hyperparameters),
            **(hyperparameters or {}),
        }

        # State
        self._is_fitted: bool = False
        self._fit_time: float | None = None
        self._val_score: float | None = None
        self._train_item_ids: np.ndarray | None = None
        self._train_target_tail: dict[str, np.ndarray] = {}  # item_id -> last values for scoring

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        time_limit: float | None = None,
    ) -> "AbstractTimeSeriesModel":
        """Train the model.

        Parameters
        ----------
        train_data : TimeSeriesDataFrame
            Training data.
        val_data : TimeSeriesDataFrame, optional
            Validation data (full series including future).
        time_limit : float, optional
            Maximum training time in seconds.

        Returns
        -------
        self
        """
        logger.info(f"Fitting {self.name} (prediction_length={self.prediction_length})")

        # Infer frequency if not provided
        if self.freq is None:
            self.freq = train_data.freq
            if self.freq is None:
                logger.warning("Could not infer frequency from training data.")

        # Store item IDs and tail values (needed for MASE-like metrics)
        self._train_item_ids = train_data.item_ids
        for item_id in self._train_item_ids:
            item_series = train_data.loc[item_id][TARGET].values
            self._train_target_tail[item_id] = item_series.copy()

        start_time = time.time()
        try:
            self._fit(train_data, val_data=val_data, time_limit=time_limit)
        except Exception as e:
            logger.error(f"Error fitting {self.name}: {e}")
            raise
        self._fit_time = time.time() - start_time
        self._is_fitted = True

        logger.info(f"Fitted {self.name} in {self._fit_time:.2f}s")

        # Score on validation data if provided
        if val_data is not None:
            try:
                self._val_score = self.score(val_data)
                logger.info(f"  Validation {self.eval_metric.name}: {self._val_score:.4f}")
            except Exception as e:
                logger.warning(f"  Could not compute validation score: {e}")

        return self

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: pd.DataFrame | None = None,
        quantile_levels: Sequence[float] | None = None,
    ) -> TimeSeriesDataFrame:
        """Generate forecasts.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Historical data (context) for each item.
        known_covariates : pd.DataFrame, optional
            Future values of known covariates.
        quantile_levels : list of float, optional
            Quantile levels to predict (e.g. ``[0.1, 0.5, 0.9]``).
            If ``None``, only the ``mean`` column is returned.

        Returns
        -------
        TimeSeriesDataFrame
            Predictions with columns ``mean`` and optionally quantile columns
            (e.g. ``0.1``, ``0.5``, ``0.9``).  Indexed by ``(item_id, timestamp)``.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} has not been fitted yet. Call fit() first.")

        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        raw = self._predict(data, known_covariates=known_covariates, quantile_levels=quantile_levels)

        # Normalize output to TimeSeriesDataFrame
        result = self._normalize_predictions(raw, data, quantile_levels)
        return result

    def score(
        self,
        data: TimeSeriesDataFrame,
        metric: str | TimeSeriesScorer | None = None,
    ) -> float:
        """Evaluate the model on ``data``.

        ``data`` should contain the **full** series (context + future).
        The last ``prediction_length`` steps per item are treated as ground
        truth; the rest is used as context for prediction.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Full data including the future window.
        metric : str or TimeSeriesScorer, optional
            Override the model's default eval_metric.

        Returns
        -------
        float
        """
        score, _ = self.score_with_predictions(data, metric=metric)
        return score

    def score_with_predictions(
        self,
        data: TimeSeriesDataFrame,
        metric: str | TimeSeriesScorer | None = None,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Evaluate the model and return both score and predictions.

        Same as ``score()``, but also returns per-item predictions for
        caching (avoids redundant predict calls in ensemble building).

        Returns
        -------
        score : float
        predictions : dict[str, np.ndarray]
            {item_id: predicted_values} for each item.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} has not been fitted yet.")

        scorer = get_metric(metric) if metric is not None else self.eval_metric

        # Batch predict: build context for ALL items at once, single predict call
        context_tsdf = data.slice_by_timestep(None, -self.prediction_length)
        context_tsdf._cached_freq = self.freq

        try:
            preds = self.predict(context_tsdf, quantile_levels=[0.5])
        except Exception:
            # Fallback to per-item predict if batch fails
            return self._score_per_item(data, scorer), {}

        # Score per item from batch prediction result
        scores = []
        item_predictions = {}
        for item_id in data.item_ids:
            item_data = data.loc[item_id]
            n = len(item_data)
            if n <= self.prediction_length:
                continue

            y_true = item_data[TARGET].values[-self.prediction_length:]

            try:
                item_pred = preds.loc[item_id]
                y_pred = item_pred["mean"].values if "mean" in preds.columns else item_pred.iloc[:, 0].values
            except (KeyError, IndexError):
                continue

            # Align lengths
            min_len = min(len(y_true), len(y_pred))
            y_true_s = y_true[:min_len]
            y_pred_s = y_pred[:min_len]

            item_predictions[item_id] = y_pred[:self.prediction_length]

            y_train = self._train_target_tail.get(item_id)
            try:
                s = scorer(y_true_s, y_pred_s, y_train=y_train)
                if np.isfinite(s):
                    scores.append(s)
            except Exception:
                pass

        if not scores:
            return float("inf"), item_predictions
        return float(np.mean(scores)), item_predictions

    def _score_per_item(self, data, scorer):
        """Fallback per-item scoring when batch predict fails."""
        scores = []
        for item_id in data.item_ids:
            item_data = data.loc[item_id]
            n = len(item_data)
            if n <= self.prediction_length:
                continue

            y_true = item_data[TARGET].values[-self.prediction_length:]

            context_df = data.loc[[item_id]].iloc[:n - self.prediction_length]
            context_tsdf = TimeSeriesDataFrame(context_df)
            context_tsdf._cached_freq = self.freq

            try:
                pred = self.predict(context_tsdf, quantile_levels=[0.5])
                y_pred = pred["mean"].values if "mean" in pred.columns else pred.iloc[:, 0].values
                min_len = min(len(y_true), len(y_pred))
                y_train = self._train_target_tail.get(item_id)
                s = scorer(y_true[:min_len], y_pred[:min_len], y_train=y_train)
                if np.isfinite(s):
                    scores.append(s)
            except Exception:
                pass

        return float(np.mean(scores)) if scores else float("inf")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    def get_hyperparameters(self) -> dict[str, Any]:
        """Return current hyperparameters."""
        return copy.deepcopy(self._hyperparameters)

    def get_hyperparameter(self, key: str, default: Any = None) -> Any:
        """Get a single hyperparameter value."""
        return self._hyperparameters.get(key, default)

    def set_hyperparameters(self, **kwargs) -> None:
        """Update hyperparameters."""
        self._hyperparameters.update(kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | None = None) -> str:
        """Save the model to disk.

        Parameters
        ----------
        path : str, optional
            Directory path.  Defaults to ``self.path``.

        Returns
        -------
        str
            Path where the model was saved.
        """
        save_path = path or self.path
        if save_path is None:
            raise ValueError("No save path specified. Set path= in constructor or save().")

        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, "model.pkl")

        with open(model_file, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved {self.name} to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: str) -> "AbstractTimeSeriesModel":
        """Load a saved model.

        Parameters
        ----------
        path : str
            Directory containing ``model.pkl``.

        Returns
        -------
        AbstractTimeSeriesModel
        """
        model_file = os.path.join(path, "model.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No model found at {model_file}")

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Loaded {model.name} from {path}")
        return model

    # ------------------------------------------------------------------
    # Info / repr
    # ------------------------------------------------------------------
    def model_info(self) -> dict[str, Any]:
        """Return a summary dict of model state."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "prediction_length": self.prediction_length,
            "freq": self.freq,
            "is_fitted": self._is_fitted,
            "fit_time_s": self._fit_time,
            "val_score": self._val_score,
            "eval_metric": self.eval_metric.name,
            "hyperparameters": self.get_hyperparameters(),
        }

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"prediction_length={self.prediction_length}, "
            f"freq={self.freq!r}, {status})"
        )

    # ------------------------------------------------------------------
    # Abstract methods (subclasses must implement)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        time_limit: float | None = None,
    ) -> None:
        """Train the model (internal).

        Subclass implementation should store any learned parameters as
        instance attributes.
        """
        ...

    @abc.abstractmethod
    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: pd.DataFrame | None = None,
        quantile_levels: Sequence[float] | None = None,
    ) -> pd.DataFrame | dict[str, np.ndarray]:
        """Generate predictions (internal).

        Must return **one of**:
        - A ``pd.DataFrame`` with ``(item_id, timestamp)`` index and at least
          a ``mean`` column.
        - A ``dict`` mapping ``item_id`` → ``np.ndarray`` of shape
          ``(prediction_length,)`` for point forecasts.
        """
        ...

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_predictions(
        self,
        raw: pd.DataFrame | dict[str, np.ndarray],
        data: TimeSeriesDataFrame,
        quantile_levels: Sequence[float],
    ) -> TimeSeriesDataFrame:
        """Convert raw model output to a standard TimeSeriesDataFrame."""
        if isinstance(raw, dict):
            raw = self._dict_to_prediction_df(raw, data, quantile_levels)

        # Ensure it's a TimeSeriesDataFrame
        if not isinstance(raw, TimeSeriesDataFrame):
            if isinstance(raw.index, pd.MultiIndex):
                raw = TimeSeriesDataFrame(raw)
                raw._cached_freq = self.freq
            else:
                raise ValueError(
                    f"Model {self.name} returned predictions without a "
                    f"(item_id, timestamp) MultiIndex."
                )

        # Ensure 'mean' column exists
        if "mean" not in raw.columns:
            if "0.5" in raw.columns:
                raw["mean"] = raw["0.5"]
            elif len(raw.columns) > 0:
                raw["mean"] = raw.iloc[:, 0]

        return raw

    def _dict_to_prediction_df(
        self,
        predictions: dict[str, np.ndarray],
        data: TimeSeriesDataFrame,
        quantile_levels: Sequence[float],
    ) -> TimeSeriesDataFrame:
        """Convert {item_id: array} predictions to a TimeSeriesDataFrame."""
        rows = []
        for item_id in predictions:
            values = np.asarray(predictions[item_id], dtype=np.float64)

            # Generate future timestamps
            item_timestamps = data.loc[item_id].index.get_level_values(TIMESTAMP)
            last_ts = item_timestamps.max()
            future_ts = pd.date_range(
                start=last_ts, periods=self.prediction_length + 1, freq=self.freq
            )[1:]  # exclude the last observed timestamp

            for t, v in zip(future_ts, values):
                row = {ITEMID: item_id, TIMESTAMP: t, "mean": v}
                # Simple quantile approximation: point forecast ± spread
                for q in quantile_levels:
                    row[str(q)] = v  # subclasses should override for proper quantiles
                rows.append(row)

        df = pd.DataFrame(rows)
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        df[ITEMID] = df[ITEMID].astype(str)
        df = df.set_index([ITEMID, TIMESTAMP])
        result = TimeSeriesDataFrame(df)
        result._cached_freq = self.freq
        return result

    def _get_last_values(self, data: TimeSeriesDataFrame) -> dict[str, float]:
        """Get the last observed target value for each item."""
        result = {}
        for item_id in data.item_ids:
            result[item_id] = float(data.loc[item_id][TARGET].iloc[-1])
        return result

    def _get_seasonal_period(self) -> int:
        """Infer a reasonable seasonal period from the frequency."""
        freq_map = {
            "h": 24,
            "D": 7,
            "W": 52,
            "MS": 12,
            "ME": 12,
            "M": 12,
            "QS": 4,
            "QE": 4,
            "Q": 4,
            "YS": 1,
            "YE": 1,
            "Y": 1,
            "B": 5,
            "min": 60,
            "T": 60,
            "s": 60,
            "S": 60,
        }
        if self.freq is None:
            return 1
        # Try exact match first, then prefix match
        if self.freq in freq_map:
            return freq_map[self.freq]
        for key in freq_map:
            if self.freq.endswith(key):
                return freq_map[key]
        return 1

    def _make_future_timestamps(self, data: TimeSeriesDataFrame, item_id: str) -> pd.DatetimeIndex:
        """Generate ``prediction_length`` future timestamps for an item."""
        item_ts = data.loc[item_id].index.get_level_values(TIMESTAMP)
        last_ts = item_ts.max()
        freq = self.freq or "D"
        return pd.date_range(start=last_ts, periods=self.prediction_length + 1, freq=freq)[1:]

    def _rows_to_tsdf(self, rows: list[dict]) -> TimeSeriesDataFrame:
        """Convert a list of row dicts to a TimeSeriesDataFrame (no 'target' required)."""
        df = pd.DataFrame(rows)
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        df[ITEMID] = df[ITEMID].astype(str)
        df = df.sort_values([ITEMID, TIMESTAMP]).set_index([ITEMID, TIMESTAMP])
        result = TimeSeriesDataFrame.__new__(TimeSeriesDataFrame)
        pd.DataFrame.__init__(result, df)
        result._cached_freq = self.freq
        return result
