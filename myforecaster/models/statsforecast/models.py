"""
Universal StatsForecast wrapper.

A single ``StatsForecastModel`` class that wraps **any** model from the
`statsforecast <https://github.com/Nixtla/statsforecast>`_ library.  Just
pass the model name as a string::

    # Any of the 38+ models:
    model = StatsForecastModel(model_name="AutoARIMA", freq="D", prediction_length=14)
    model = StatsForecastModel(model_name="AutoETS", ...)
    model = StatsForecastModel(model_name="CrostonSBA", ...)

For convenience, the most popular models are also registered individually
(``AutoARIMAModel``, ``AutoETSModel``, etc.) so they can be referenced by
short names in the Predictor's ``hyperparameters`` dict.

Supported models (auto-discovered from ``statsforecast.models``)::

    AutoARIMA, AutoETS, AutoTheta, AutoCES, AutoMFLES, AutoTBATS,
    ARIMA, MSTL, Theta, OptimizedTheta, DynamicTheta,
    DynamicOptimizedTheta, Holt, HoltWinters,
    SimpleExponentialSmoothing, SeasonalExponentialSmoothing,
    CrostonClassic, CrostonOptimized, CrostonSBA, ADIDA, IMAPA, TSB,
    HistoricAverage, WindowAverage, SeasonalWindowAverage, ARCH, GARCH,
    TBATS, MFLES, ... and any future additions.
"""

from __future__ import annotations

import inspect
import logging
import signal
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd


class _Timeout:
    """Context manager that raises TimeoutError after *seconds* (UNIX only)."""

    def __init__(self, seconds: float | None):
        self.seconds = int(seconds) if seconds else 0

    def _handler(self, signum, frame):
        raise TimeoutError(f"Timed out after {self.seconds}s")

    def __enter__(self):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *args):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)

from myforecaster.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame
from myforecaster.models.abstract_model import AbstractTimeSeriesModel
from myforecaster.models import register_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discover available models from statsforecast
# ---------------------------------------------------------------------------
_SF_MODEL_CLASSES: dict[str, type] = {}

try:
    import statsforecast.models as _sf_models

    for _name, _obj in inspect.getmembers(_sf_models, inspect.isclass):
        if _name.startswith("_"):
            continue
        if hasattr(_obj, "forecast") or hasattr(_obj, "predict"):
            _SF_MODEL_CLASSES[_name] = _obj
except ImportError:
    logger.warning(
        "statsforecast is not installed. "
        "Install with: pip install 'myforecaster[stats]'"
    )

# Models that do NOT accept season_length
_NO_SEASON_LENGTH = {
    "CrostonClassic", "CrostonOptimized", "CrostonSBA",
    "ADIDA", "IMAPA", "TSB",
    "HistoricAverage", "Naive", "SeasonalNaive",
    "RandomWalkWithDrift", "WindowAverage", "SeasonalWindowAverage",
    "ConstantModel", "NaNModel", "ZeroModel", "SklearnModel",
}


def list_statsforecast_models() -> list[str]:
    """Return names of all discovered StatsForecast model classes."""
    return sorted(_SF_MODEL_CLASSES.keys())


# ---------------------------------------------------------------------------
# Universal wrapper
# ---------------------------------------------------------------------------

class StatsForecastModel(AbstractTimeSeriesModel):
    """Universal wrapper for any ``statsforecast`` model.

    Parameters (via hyperparameters)
    --------------------------------
    model_name : str
        Name of the statsforecast model class (e.g. ``"AutoARIMA"``).
    season_length : int or None
        Passed to the underlying model. If ``None``, inferred from freq.
    n_jobs : int
        Number of parallel jobs for fitting across items (default 1).
    model_kwargs : dict
        Extra keyword arguments forwarded to the model constructor.

    Example::

        model = StatsForecastModel(
            freq="D", prediction_length=14,
            hyperparameters={"model_name": "AutoETS", "season_length": 7}
        )
    """

    _default_hyperparameters = {
        "model_name": "AutoETS",
        "season_length": None,
        "n_jobs": -1,  # Use all cores (AG default)
        "model_kwargs": {},
    }

    def __init__(self, model_name: str | None = None, **kwargs):
        # Allow model_name as a positional-style kwarg for convenience
        hp = kwargs.get("hyperparameters", {}) or {}
        if model_name is not None:
            hp.setdefault("model_name", model_name)
        kwargs["hyperparameters"] = hp

        super().__init__(**kwargs)

        actual_name = self.get_hyperparameter("model_name")
        if self.name == self.__class__.__name__:
            self.name = actual_name

    # ------------------------------------------------------------------
    # Core — Batch mode (AutoGluon-style, 10-50x faster)
    # ------------------------------------------------------------------
    def _fit(self, train_data, val_data=None, time_limit=None):
        model_name = self.get_hyperparameter("model_name")
        if model_name not in _SF_MODEL_CLASSES:
            available = list_statsforecast_models()
            raise ValueError(
                f"Unknown statsforecast model: {model_name!r}. "
                f"Available: {available}"
            )

        season_length = self.get_hyperparameter("season_length")
        if season_length is None:
            season_length = self._get_seasonal_period()
        n_jobs = self.get_hyperparameter("n_jobs")

        model_kwargs = dict(self.get_hyperparameter("model_kwargs") or {})

        # Inject season_length if the model supports it
        sf_cls = _SF_MODEL_CLASSES[model_name]
        if model_name not in _NO_SEASON_LENGTH:
            sig = inspect.signature(sf_cls.__init__)
            if "season_length" in sig.parameters:
                model_kwargs.setdefault("season_length", season_length)

        self._season_length = season_length
        self._model_kwargs = model_kwargs

        # Build long-format DataFrame for StatsForecast batch API
        from statsforecast import StatsForecast

        sf_df = self._to_sf_dataframe(train_data)
        sf_model_instance = sf_cls(**model_kwargs)

        self._sf = StatsForecast(
            models=[sf_model_instance],
            freq=self.freq,
            n_jobs=n_jobs,
        )
        self._sf.fit(sf_df)
        # Detect the column name SF uses for this model (may differ from class name)
        # e.g. AutoCES → "CES", AutoETS → "AutoETS"
        try:
            _probe = self._sf.predict(h=1)
            if isinstance(_probe.index, pd.MultiIndex):
                _probe = _probe.reset_index()
            elif "unique_id" not in _probe.columns:
                _probe = _probe.reset_index()
            # Find the column that is NOT unique_id or ds
            _meta_cols = {"unique_id", "ds"}
            _model_cols = [c for c in _probe.columns if c not in _meta_cols
                           and "-lo-" not in c and "-hi-" not in c]
            self._model_name_in_sf = _model_cols[0] if _model_cols else type(sf_model_instance).__name__
        except Exception:
            self._model_name_in_sf = type(sf_model_instance).__name__

    def _to_sf_dataframe(self, data: TimeSeriesDataFrame) -> pd.DataFrame:
        """Convert to StatsForecast's expected long-format DataFrame."""
        df = data.reset_index()
        # StatsForecast expects columns: unique_id, ds, y
        result = pd.DataFrame({
            "unique_id": df[ITEMID].values,
            "ds": pd.to_datetime(df[TIMESTAMP].values),
            "y": df[TARGET].values.astype(np.float64),
        })
        # Handle NaN
        result["y"] = result["y"].ffill().bfill()
        return result

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        levels = _quantiles_to_levels(quantile_levels)

        # Batch predict via StatsForecast API
        try:
            fc_df = self._sf.predict(h=self.prediction_length, level=levels)
        except Exception:
            # Some models (Croston etc.) don't support prediction_intervals
            fc_df = self._sf.predict(h=self.prediction_length)
            levels = []  # No quantile columns available

        # fc_df: index=unique_id, columns=[ds, ModelName, ModelName-lo-80, ...]
        if isinstance(fc_df.index, pd.MultiIndex):
            fc_df = fc_df.reset_index()
        elif "unique_id" not in fc_df.columns:
            fc_df = fc_df.reset_index()

        mn = self._model_name_in_sf  # e.g. "AutoETS"

        # Build output rows vectorized
        rows_list = []
        for item_id in data.item_ids:
            item_fc = fc_df[fc_df["unique_id"] == item_id]
            timestamps = self._make_future_timestamps(data, item_id)

            if len(item_fc) == 0:
                # Fallback
                last_val = float(data.loc[item_id][TARGET].iloc[-1])
                for i in range(self.prediction_length):
                    row = {ITEMID: item_id, TIMESTAMP: timestamps[i], "mean": last_val}
                    for q in quantile_levels:
                        row[str(q)] = last_val
                    rows_list.append(row)
                continue

            mean_vals = item_fc[mn].values[:self.prediction_length]

            for i in range(min(self.prediction_length, len(mean_vals))):
                row = {ITEMID: item_id, TIMESTAMP: timestamps[i], "mean": float(mean_vals[i])}
                for q in quantile_levels:
                    if q == 0.5:
                        row[str(q)] = float(mean_vals[i])
                        continue
                    tail = min(q, 1 - q)
                    level = round((1 - 2 * tail) * 100)
                    if q < 0.5:
                        key = f"{mn}-lo-{level}"
                    else:
                        key = f"{mn}-hi-{level}"
                    if key in item_fc.columns:
                        row[str(q)] = float(item_fc[key].values[i])
                    else:
                        row[str(q)] = float(mean_vals[i])
                rows_list.append(row)

        return self._rows_to_tsdf(rows_list)


# ---------------------------------------------------------------------------
# Quantile ↔ Level conversion helpers
# ---------------------------------------------------------------------------

def _quantiles_to_levels(quantile_levels: Sequence[float]) -> list[int]:
    """Convert quantile levels to statsforecast confidence levels.

    statsforecast uses symmetric confidence levels (e.g. level=80 means
    the 10th and 90th percentiles).  We find all unique symmetric pairs.
    """
    levels = set()
    for q in quantile_levels:
        if q == 0.5:
            continue  # median = mean
        # level = 100 - 2 * min(q, 1-q) * 100
        tail = min(q, 1 - q)
        level = round((1 - 2 * tail) * 100)
        if 1 <= level <= 99:
            levels.add(level)

    if not levels:
        levels.add(80)  # default

    return sorted(levels)


def _levels_to_quantile_arrays(
    fc: dict,
    quantile_levels: Sequence[float],
    levels: list[int],
    mean_vals: np.ndarray,
) -> dict[float, np.ndarray]:
    """Map statsforecast forecast dict to per-quantile arrays."""
    result = {}
    for q in quantile_levels:
        if q == 0.5:
            result[q] = mean_vals.copy()
            continue

        tail = min(q, 1 - q)
        level = round((1 - 2 * tail) * 100)

        if q < 0.5:
            key = f"lo-{level}"
        else:
            key = f"hi-{level}"

        if key in fc:
            result[q] = np.asarray(fc[key], dtype=np.float64)
        else:
            # Fallback: use mean
            result[q] = mean_vals.copy()

    return result


# ---------------------------------------------------------------------------
# Pre-registered popular models (convenience shortcuts)
# ---------------------------------------------------------------------------

def _make_shortcut_class(model_name: str, registry_name: str) -> type:
    """Create a shortcut class for a popular StatsForecast model."""

    class _Shortcut(StatsForecastModel):
        _default_hyperparameters = {
            **StatsForecastModel._default_hyperparameters,
            "model_name": model_name,
        }

        def __init__(self, **kwargs):
            hp = kwargs.get("hyperparameters", {}) or {}
            hp.setdefault("model_name", model_name)
            kwargs["hyperparameters"] = hp
            super().__init__(**kwargs)

    _Shortcut.__name__ = f"{model_name}Model"
    _Shortcut.__qualname__ = f"{model_name}Model"
    _Shortcut.__doc__ = f"Shortcut for ``StatsForecastModel(model_name={model_name!r}, ...)``."
    return _Shortcut


# Auto-register the most important models with friendly names
_POPULAR_MODELS = {
    "AutoARIMA": "AutoARIMA",
    "AutoETS": "AutoETS",
    "AutoTheta": "AutoTheta",
    "AutoCES": "AutoCES",
    "AutoMFLES": "AutoMFLES",
    "AutoTBATS": "AutoTBATS",
    "ARIMA": "ARIMA",
    "ETS": "AutoETS",       # common alias
    "Theta": "Theta",
    "CES": "AutoCES",       # common alias
    "MSTL": "MSTL",
    "DynamicOptimizedTheta": "DynamicOptimizedTheta",
    "CrostonSBA": "CrostonSBA",
    "CrostonClassic": "CrostonClassic",
    "CrostonOptimized": "CrostonOptimized",
    "ADIDA": "ADIDA",
    "IMAPA": "IMAPA",
    "TSB": "TSB",
}

# Register each shortcut in the MODEL_REGISTRY
for _reg_name, _sf_name in _POPULAR_MODELS.items():
    _cls = _make_shortcut_class(_sf_name, _reg_name)
    register_model(_reg_name)(_cls)
    # Also export at module level
    globals()[f"{_reg_name}Model"] = _cls
