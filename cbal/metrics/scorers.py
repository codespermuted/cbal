"""
Evaluation metrics for time series forecasting.

All scorers follow a unified interface::

    scorer = MASE()
    score = scorer(y_true, y_pred, y_train=y_train)

For quantile metrics::

    scorer = WQL()
    score = scorer(y_true, y_pred_quantiles, quantile_levels=[0.1, 0.5, 0.9])

Each scorer exposes:
- ``name``: human-readable name
- ``sign``: +1 if higher is better, -1 if lower is better
- ``optimum``: the perfect score (0.0 for error metrics, 1.0 for R², etc.)
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np
import pandas as pd


class TimeSeriesScorer(abc.ABC):
    """Abstract base class for time series evaluation metrics.

    Subclasses must implement :meth:`_score`.
    """

    name: str = "base_scorer"
    sign: int = -1  # -1 = lower is better (default for error metrics)
    optimum: float = 0.0

    def __call__(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series | pd.DataFrame,
        y_train: np.ndarray | pd.Series | None = None,
        quantile_levels: Sequence[float] | None = None,
        horizon_weight: np.ndarray | None = None,
    ) -> float:
        """Compute the metric.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like
        y_train : array-like, optional
        quantile_levels : list of float, optional
        horizon_weight : array-like, optional
            Per-step weight for the forecast horizon.  If provided,
            the per-step errors are multiplied by these weights before
            averaging.  Must have the same length as ``y_true``.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values.astype(np.float64)
        else:
            y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_train is not None:
            y_train = np.asarray(y_train, dtype=np.float64)

        score = self._score(y_true, y_pred, y_train=y_train,
                            quantile_levels=quantile_levels,
                            horizon_weight=horizon_weight)
        return score

    @abc.abstractmethod
    def _score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray | None = None,
        quantile_levels: Sequence[float] | None = None,
        **kwargs,
    ) -> float: ...

    def __repr__(self) -> str:
        direction = "higher_is_better" if self.sign == 1 else "lower_is_better"
        return f"{self.__class__.__name__}(name={self.name!r}, {direction})"


# ============================================================================
# Point-forecast metrics
# ============================================================================

class MAE(TimeSeriesScorer):
    """Mean Absolute Error."""
    name = "MAE"

    def _score(self, y_true, y_pred, **kwargs) -> float:
        errs = np.abs(y_true - _point(y_pred))
        w = kwargs.get("horizon_weight")
        if w is not None:
            return float(np.average(errs, weights=np.asarray(w)))
        return float(np.mean(errs))


class RMSE(TimeSeriesScorer):
    """Root Mean Squared Error."""
    name = "RMSE"

    def _score(self, y_true, y_pred, **kwargs) -> float:
        sq = (y_true - _point(y_pred)) ** 2
        w = kwargs.get("horizon_weight")
        if w is not None:
            return float(np.sqrt(np.average(sq, weights=np.asarray(w))))
        return float(np.sqrt(np.mean(sq)))


class MAPE(TimeSeriesScorer):
    """Mean Absolute Percentage Error.

    Warning: undefined when ``y_true`` contains zeros.
    """
    name = "MAPE"

    def _score(self, y_true, y_pred, **kwargs) -> float:
        p = _point(y_pred)
        mask = y_true != 0
        if not mask.any():
            return float("inf")
        return float(np.mean(np.abs((y_true[mask] - p[mask]) / y_true[mask])) * 100)


class sMAPE(TimeSeriesScorer):
    """Symmetric Mean Absolute Percentage Error."""
    name = "sMAPE"

    def _score(self, y_true, y_pred, **kwargs) -> float:
        p = _point(y_pred)
        denom = np.abs(y_true) + np.abs(p)
        mask = denom != 0
        if not mask.any():
            return 0.0
        return float(np.mean(2.0 * np.abs(y_true[mask] - p[mask]) / denom[mask]) * 100)


class MASE(TimeSeriesScorer):
    """Mean Absolute Scaled Error.

    Requires ``y_train`` to compute the naive-forecast scale.
    """
    name = "MASE"

    def __init__(self, seasonal_period: int = 1):
        self.seasonal_period = seasonal_period

    def _score(self, y_true, y_pred, y_train=None, **kwargs) -> float:
        if y_train is None:
            raise ValueError("MASE requires y_train to compute the scaling factor.")
        p = _point(y_pred)
        # Scale = mean absolute error of the seasonal naive forecast on training data
        naive_errors = np.abs(y_train[self.seasonal_period :] - y_train[: -self.seasonal_period])
        scale = np.mean(naive_errors)
        # AutoGluon convention: if scale ≈ 0 (constant series), use 1.0
        # This prevents inf scores from dominating the average
        if scale < 1e-12:
            scale = 1.0
        return float(np.mean(np.abs(y_true - p)) / scale)


class RMSSE(TimeSeriesScorer):
    """Root Mean Squared Scaled Error.

    Requires ``y_train``.  Used in M5 competition.
    """
    name = "RMSSE"

    def __init__(self, seasonal_period: int = 1):
        self.seasonal_period = seasonal_period

    def _score(self, y_true, y_pred, y_train=None, **kwargs) -> float:
        if y_train is None:
            raise ValueError("RMSSE requires y_train to compute the scaling factor.")
        p = _point(y_pred)
        naive_errors = y_train[self.seasonal_period :] - y_train[: -self.seasonal_period]
        scale_sq = np.mean(naive_errors ** 2)
        if scale_sq < 1e-12:
            scale_sq = 1.0
        return float(np.sqrt(np.mean((y_true - p) ** 2) / scale_sq))


# ============================================================================
# Quantile-forecast metrics
# ============================================================================

class WQL(TimeSeriesScorer):
    """Weighted Quantile Loss (a.k.a. scaled pinball loss).

    When ``y_pred`` is 1-D (point forecast), falls back to MAE (equivalent
    to quantile 0.5).  When ``y_pred`` is 2-D, ``quantile_levels`` must be
    provided.
    """
    name = "WQL"

    def _score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray | None = None,
        quantile_levels: Sequence[float] | None = None,
        **kwargs,
    ) -> float:
        # Point forecast fallback
        if y_pred.ndim == 1:
            return float(np.mean(np.abs(y_true - y_pred)))

        if quantile_levels is None:
            raise ValueError("WQL requires quantile_levels when y_pred is 2-D.")

        quantile_levels = np.asarray(quantile_levels, dtype=np.float64)
        n, q = y_pred.shape
        assert q == len(quantile_levels), (
            f"y_pred has {q} columns but {len(quantile_levels)} quantile_levels given."
        )

        total_loss = 0.0
        for i, tau in enumerate(quantile_levels):
            errors = y_true - y_pred[:, i]
            loss = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
            total_loss += np.mean(loss)

        # Average over quantiles, then normalise by mean absolute value of y_true
        avg_loss = total_loss / len(quantile_levels)
        scale = np.mean(np.abs(y_true))
        if scale < 1e-12:
            return float(avg_loss)
        return float(avg_loss / scale)


class QuantileLoss(TimeSeriesScorer):
    """Pinball loss for a single quantile level.

    Parameters
    ----------
    quantile : float
        The target quantile in (0, 1).
    """
    name = "QuantileLoss"

    def __init__(self, quantile: float = 0.5):
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self.name = f"QuantileLoss[{quantile}]"

    def _score(self, y_true, y_pred, **kwargs) -> float:
        p = _point(y_pred)
        errors = y_true - p
        loss = np.where(errors >= 0, self.quantile * errors, (self.quantile - 1) * errors)
        return float(np.mean(loss))


class WAPE(TimeSeriesScorer):
    """Weighted Absolute Percentage Error (volume-weighted)."""
    name = "WAPE"
    sign = -1
    optimum = 0.0

    def _score(self, y_true, y_pred, **kwargs) -> float:
        pt = _point(y_pred)
        denom = np.sum(np.abs(y_true))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(np.abs(y_true - pt)) / denom)


class Coverage(TimeSeriesScorer):
    """Prediction interval coverage — fraction of actuals within
    the (lower, upper) quantile band.

    Returns a value in [0, 1].  Higher is better but should match
    the nominal level (e.g. 0.8 for 0.1–0.9 band).
    """
    name = "Coverage"
    sign = 1   # higher is better
    optimum = 1.0

    def _score(self, y_true, y_pred, quantile_levels=None, **kwargs) -> float:
        if y_pred.ndim < 2 or y_pred.shape[1] < 2:
            return float("nan")
        lower = y_pred[:, 0]
        upper = y_pred[:, -1]
        covered = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(covered))


class SQL(TimeSeriesScorer):
    """Scaled Quantile Loss — WQL normalized by MAE of naive forecast."""
    name = "SQL"
    sign = -1
    optimum = 0.0

    def _score(self, y_true, y_pred, y_train=None,
               quantile_levels=None, **kwargs) -> float:
        wql_scorer = WQL()
        wql = wql_scorer._score(y_true, y_pred,
                                quantile_levels=quantile_levels)
        if y_train is not None and len(y_train) > 1:
            naive_mae = np.mean(np.abs(np.diff(y_train)))
            if naive_mae > 1e-12:
                return wql / naive_mae
        return wql


# ============================================================================
# Custom metric helpers — user-friendly API for domain-specific losses
# ============================================================================

class CustomMetric(TimeSeriesScorer):
    """Wrap a plain Python function into a TimeSeriesScorer.

    This is the easiest way to inject domain knowledge into model selection.
    The function receives ``(y_true, y_pred)`` arrays and optionally
    ``y_train`` and ``**kwargs``, and must return a single float.

    Parameters
    ----------
    func : callable
        ``func(y_true, y_pred, *, y_train=None, **kw) -> float``
    name : str
        Human-readable name shown on leaderboards.
    greater_is_better : bool
        ``True`` if higher values mean better forecasts (e.g. R²),
        ``False`` if lower is better (e.g. MAE, RMSE). Default ``False``.

    Examples
    --------
    Penalise under-prediction 3× more than over-prediction::

        def asymmetric_loss(y_true, y_pred, **kw):
            err = y_true - y_pred
            return np.mean(np.where(err > 0, 3 * err**2, err**2))

        predictor = TimeSeriesPredictor(
            prediction_length=7,
            eval_metric=CustomMetric(asymmetric_loss, "AsymLoss"),
        )

    Peak-hour weighted MAE (e.g. hours 8-20 matter 5× more)::

        weights = np.ones(24)
        weights[8:20] = 5.0

        predictor = TimeSeriesPredictor(
            prediction_length=24,
            eval_metric=CustomMetric(
                lambda yt, yp, **kw: np.average(np.abs(yt - yp), weights=weights[:len(yt)]),
                name="PeakMAE",
            ),
        )
    """

    def __init__(
        self,
        func: callable,
        name: str = "CustomMetric",
        greater_is_better: bool = False,
    ):
        self._func = func
        self.name = name
        self.sign = 1 if greater_is_better else -1
        self.optimum = 0.0

    def _score(self, y_true, y_pred, y_train=None, **kwargs) -> float:
        p = _point(y_pred)
        return float(self._func(y_true, p, y_train=y_train, **kwargs))


def make_scorer(
    func: callable,
    name: str = "CustomMetric",
    greater_is_better: bool = False,
) -> CustomMetric:
    """Create a scorer from a plain function — shorthand for ``CustomMetric()``.

    Parameters
    ----------
    func : callable
        ``func(y_true, y_pred, *, y_train=None, **kw) -> float``
    name : str
        Metric name for display.
    greater_is_better : bool
        Direction of metric improvement.

    Returns
    -------
    CustomMetric

    Examples
    --------
    ::

        from cbal.metrics import make_scorer

        # Simple: penalise big errors more
        my_metric = make_scorer(
            lambda yt, yp, **kw: np.mean((yt - yp)**4),
            name="L4Loss",
        )

        predictor = TimeSeriesPredictor(eval_metric=my_metric, prediction_length=7)
    """
    return CustomMetric(func, name=name, greater_is_better=greater_is_better)


class HorizonWeightedMetric(TimeSeriesScorer):
    """Wrap any base metric with per-horizon weights.

    Useful when certain forecast steps are more important than others
    (e.g. next-day matters more than day-7, or weekdays > weekends).

    Parameters
    ----------
    base_metric : str or TimeSeriesScorer
        The underlying metric (e.g. ``"MAE"``, ``"RMSE"``).
    horizon_weights : array-like
        Weight for each forecast step. Length must equal ``prediction_length``.
        Weights are normalised internally so they sum to 1.

    Examples
    --------
    First 3 days matter 5× more than the rest::

        weights = [5]*3 + [1]*4  # prediction_length=7
        metric = HorizonWeightedMetric("MAE", weights)
        predictor = TimeSeriesPredictor(eval_metric=metric, prediction_length=7)
    """

    def __init__(
        self,
        base_metric: "str | TimeSeriesScorer" = "MAE",
        horizon_weights: "Sequence[float] | np.ndarray" = (),
    ):
        self._base = get_metric(base_metric) if isinstance(base_metric, str) else base_metric
        w = np.asarray(horizon_weights, dtype=np.float64)
        self._weights = w / w.sum() if w.sum() > 0 else w
        self.name = f"HW_{self._base.name}"
        self.sign = self._base.sign
        self.optimum = self._base.optimum

    def _score(self, y_true, y_pred, y_train=None, **kwargs) -> float:
        kwargs.pop("horizon_weight", None)
        return self._base._score(
            y_true, y_pred, y_train=y_train,
            horizon_weight=self._weights[:len(y_true)], **kwargs,
        )


# ============================================================================
# Metric registry
# ============================================================================

METRIC_REGISTRY: dict[str, type[TimeSeriesScorer]] = {
    "MAE": MAE,
    "RMSE": RMSE,
    "MAPE": MAPE,
    "sMAPE": sMAPE,
    "MASE": MASE,
    "RMSSE": RMSSE,
    "WQL": WQL,
    "SQL": SQL,
    "WAPE": WAPE,
    "Coverage": Coverage,
}

DEFAULT_METRIC = "MASE"


def get_metric(
    name: "str | TimeSeriesScorer",
    seasonal_period: int | None = None,
) -> TimeSeriesScorer:
    """Get a scorer instance by name or pass through an existing scorer.

    Accepts:
    - A string name (e.g. ``"MAE"``, ``"MASE"``).
    - A ``TimeSeriesScorer`` instance (passed through unchanged).
    - A ``CustomMetric`` or ``make_scorer()`` result.
    - A plain callable — automatically wrapped via ``make_scorer()``.

    Parameters
    ----------
    name : str or TimeSeriesScorer or callable
    seasonal_period : int, optional
        Passed to MASE / RMSSE for proper scaling.
    """
    # Already a scorer instance (CustomMetric, HorizonWeightedMetric, etc.)
    if isinstance(name, TimeSeriesScorer):
        return name

    # Plain callable → wrap automatically
    if callable(name) and not isinstance(name, str):
        return make_scorer(name, name=getattr(name, "__name__", "CustomMetric"))

    key = name.upper() if name.upper() in METRIC_REGISTRY else name
    if key not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name!r}. Available: {list(METRIC_REGISTRY.keys())}")
    cls = METRIC_REGISTRY[key]
    if seasonal_period and key in ("MASE", "RMSSE"):
        return cls(seasonal_period=seasonal_period)
    return cls()


# ============================================================================
# Helpers
# ============================================================================

def _point(y_pred: np.ndarray) -> np.ndarray:
    """Extract point forecast: if 2-D, take the median column (middle)."""
    if y_pred.ndim == 1:
        return y_pred
    # Convention: middle column is the median (0.5 quantile)
    mid = y_pred.shape[1] // 2
    return y_pred[:, mid]
