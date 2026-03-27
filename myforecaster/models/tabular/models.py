"""
Tabular time series models — universal GBDT/ML wrapper.

Converts time series into a supervised learning problem via automatic
feature engineering, then uses any scikit-learn-compatible regressor.

Supported backends (auto-detected from string name):
- ``"LightGBM"`` — fastest, default
- ``"XGBoost"`` — GPU friendly
- ``"CatBoost"`` — auto handles categoricals
- ``"RandomForest"``, ``"ExtraTrees"`` — sklearn tree ensembles
- ``"LinearRegression"``, ``"Ridge"``, ``"ElasticNet"`` — linear models
- Any scikit-learn regressor instance passed via ``model_kwargs``

Two prediction strategies:
- **RecursiveTabularModel**: One model, feeds predictions back as lags.
- **DirectTabularModel**: One model per horizon step.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from myforecaster.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame
from myforecaster.models.abstract_model import AbstractTimeSeriesModel
from myforecaster.models import register_model
from myforecaster.models.tabular.features import (
    build_feature_matrix,
    build_batch_features,
    get_default_lags,
    get_default_windows,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry: string name -> constructor function
# ---------------------------------------------------------------------------
_BACKEND_REGISTRY: dict[str, Any] = {}


def _register_backends():
    """Lazy-register backends to avoid import errors for missing packages."""
    if _BACKEND_REGISTRY:
        return

    # LightGBM — AG uses num_boost_round=10000 with early stopping
    try:
        import lightgbm as lgb
        _BACKEND_REGISTRY["LightGBM"] = lambda p: lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 2000),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 128),
            min_child_samples=p.get("min_child_samples", 5),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_alpha=p.get("reg_alpha", 0.1),
            reg_lambda=p.get("reg_lambda", 1.0),
            verbosity=-1,
            n_jobs=p.get("n_jobs", -1),
            force_col_wise=True,
        )
    except ImportError:
        pass

    # XGBoost
    try:
        import xgboost as xgb
        _BACKEND_REGISTRY["XGBoost"] = lambda p: xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 200),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 6),
            verbosity=0,
            n_jobs=p.get("n_jobs", 1),
        )
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        _BACKEND_REGISTRY["CatBoost"] = lambda p: CatBoostRegressor(
            iterations=p.get("n_estimators", 200),
            learning_rate=p.get("learning_rate", 0.05),
            depth=p.get("max_depth", 6),
            verbose=0,
        )
    except ImportError:
        pass

    # sklearn models
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor

    _BACKEND_REGISTRY["RandomForest"] = lambda p: RandomForestRegressor(
        n_estimators=p.get("n_estimators", 100),
        n_jobs=p.get("n_jobs", 1),
        random_state=p.get("random_state", 42),
    )
    _BACKEND_REGISTRY["ExtraTrees"] = lambda p: ExtraTreesRegressor(
        n_estimators=p.get("n_estimators", 100),
        n_jobs=p.get("n_jobs", 1),
        random_state=p.get("random_state", 42),
    )
    _BACKEND_REGISTRY["GradientBoosting"] = lambda p: GradientBoostingRegressor(
        n_estimators=p.get("n_estimators", 100),
        learning_rate=p.get("learning_rate", 0.05),
        max_depth=p.get("max_depth", 5),
    )
    _BACKEND_REGISTRY["LinearRegression"] = lambda p: LinearRegression()
    _BACKEND_REGISTRY["Ridge"] = lambda p: Ridge(alpha=p.get("alpha", 1.0))
    _BACKEND_REGISTRY["ElasticNet"] = lambda p: ElasticNet(
        alpha=p.get("alpha", 1.0), l1_ratio=p.get("l1_ratio", 0.5)
    )
    _BACKEND_REGISTRY["KNN"] = lambda p: KNeighborsRegressor(
        n_neighbors=p.get("n_neighbors", 5),
        n_jobs=p.get("n_jobs", 1),
    )


def list_tabular_backends() -> list[str]:
    """Return available tabular backend names."""
    _register_backends()
    return sorted(_BACKEND_REGISTRY.keys())


def _create_backend(backend_name: str, params: dict) -> Any:
    """Instantiate a backend regressor by name."""
    _register_backends()
    if backend_name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown tabular backend: {backend_name!r}. "
            f"Available: {list_tabular_backends()}"
        )
    return _BACKEND_REGISTRY[backend_name](params)


# ---------------------------------------------------------------------------
# Shared training logic
# ---------------------------------------------------------------------------

def _build_training_data(
    train_data: TimeSeriesDataFrame,
    lags: list[int],
    windows: list[int],
    include_dates: bool,
    horizon_offset: int = 0,
    freq: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build global (X, y) across all items.

    Parameters
    ----------
    horizon_offset : int
        For Direct models: target is y_{t + horizon_offset}.
        0 means next-step (same as Recursive).
    freq : str or None
        Passed to date feature encoding.
    """
    max_lag = max(lags) if lags else 1
    X_all, y_all = [], []

    for item_id in train_data.item_ids:
        item_df = train_data.loc[item_id]
        series = item_df[TARGET].values.astype(np.float64)
        timestamps = item_df.index.get_level_values(TIMESTAMP)

        X = build_feature_matrix(series, timestamps, lags, windows, include_dates, freq=freq)
        n = len(series)

        if horizon_offset == 0:
            y = series.copy()
            valid = X.notna().all(axis=1)
        else:
            # Target shifted forward by horizon_offset
            if n <= horizon_offset:
                continue
            y = np.full(n, np.nan)
            y[:n - horizon_offset] = series[horizon_offset:]
            valid = X.notna().all(axis=1) & ~np.isnan(y)

        X_all.append(X[valid])
        y_all.append(y[valid.values] if isinstance(valid, pd.Series) else y[valid])

    if not X_all:
        raise ValueError("No valid training samples after feature engineering.")

    X_train = pd.concat(X_all, ignore_index=True)
    y_train = np.concatenate(y_all)
    feature_names = list(X_train.columns)

    # Fill remaining NaN (edge cases from rolling on short series)
    X_train = X_train.fillna(0)

    return X_train, y_train, feature_names


# ---------------------------------------------------------------------------
# RecursiveTabularModel
# ---------------------------------------------------------------------------

@register_model("RecursiveTabular")
class RecursiveTabularModel(AbstractTimeSeriesModel):
    """Recursive multi-step forecasting with any tabular ML backend.

    Trains one model to predict the next step, then feeds predictions
    back as lag features for subsequent steps.

    Other Parameters
    ----------------
    backend : str
        Backend name (default ``"LightGBM"``).
    lags : list of int or None
        Lag indices. ``None`` = auto from frequency.
    rolling_windows : list of int or None
        Rolling window sizes. ``None`` = auto.
    include_date_features : bool
        Add calendar features (default ``True``).
    n_estimators : int
        Number of trees/rounds (default 200).
    learning_rate : float
        Learning rate for boosting (default 0.05).
    """

    _default_hyperparameters = {
        "backend": "LightGBM",
        "lags": None,
        "rolling_windows": None,
        "include_date_features": True,
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_child_samples": 5,
        "n_jobs": -1,
        "target_scaler": "standard",  # AG default: standard for Recursive
        "differences": None,  # Auto: [seasonal_period] for seasonal differencing
        "early_stopping_rounds": 50,  # Stop if no improvement for 50 rounds
    }

    def _fit(self, train_data, val_data=None, time_limit=None):
        sp = self._get_seasonal_period()
        # Filter lags by minimum series length to avoid empty feature matrices
        min_len = min(len(train_data.loc[iid]) for iid in train_data.item_ids)
        all_lags = self.get_hyperparameter("lags") or get_default_lags(sp, freq=self.freq)
        self._lags = [l for l in all_lags if l < min_len - 1] or [1]
        self._windows = [w for w in (self.get_hyperparameter("rolling_windows") or get_default_windows(sp))
                         if w < min_len] or [2]
        self._include_dates = self.get_hyperparameter("include_date_features")

        # --- Seasonal differencing (AG-style) ---
        # AG only diffs when series is long enough: min_len > 3*sp + prediction_length
        # For many short series (like M3), differencing hurts more than it helps
        diff_cfg = self.get_hyperparameter("differences")
        n_items = train_data.num_items
        if diff_cfg is None:
            if sp > 1 and min_len > sp * 4 + self.prediction_length and n_items <= 50:
                # Only diff for few long series (like ETT)
                diff_cfg = [sp]
            else:
                diff_cfg = []
        self._differences = diff_cfg or []
        self._diff_tails: dict[str, list[np.ndarray]] = {}  # store tails for inverse

        # Per-item target scaling (AG default: standard for Recursive)
        scaler_method = self.get_hyperparameter("target_scaler")
        self._item_scales: dict[str, tuple[float, float]] = {}
        scaled_data = train_data
        if scaler_method and scaler_method != "none":
            from myforecaster.models.wrappers import TargetScaler
            self._target_scaler_obj = TargetScaler(method=scaler_method)
            scaled_data = self._target_scaler_obj.fit_transform(train_data)
            self._item_scales = {
                iid: (self._target_scaler_obj._loc.get(iid, 0.0),
                      self._target_scaler_obj._scale.get(iid, 1.0))
                for iid in train_data.item_ids
            }
        else:
            self._target_scaler_obj = None

        # Apply differencing after scaling — use fast numpy, avoid slow DataFrame ops
        if self._differences:
            from myforecaster.dataset.ts_dataframe import TARGET, TIMESTAMP
            total_trim = sum(self._differences)
            keep_rows = []
            for item_id in scaled_data.item_ids:
                item_df = scaled_data.loc[item_id]
                series = item_df[TARGET].values.astype(np.float64)
                tails = []
                diffed = series.copy()
                for d in self._differences:
                    tails.append(diffed[-d:].copy())
                    diffed = diffed[d:] - diffed[:-d]
                self._diff_tails[item_id] = tails
                # Build trimmed rows as plain dict for speed
                ts_vals = item_df.index.get_level_values(TIMESTAMP)[total_trim:]
                for t, v in zip(ts_vals, diffed):
                    keep_rows.append({ITEMID: item_id, TIMESTAMP: t, TARGET: v})
            diff_df = pd.DataFrame(keep_rows)
            scaled_data = TimeSeriesDataFrame.from_data_frame(diff_df)
            scaled_data._cached_freq = self.freq

        X_train, y_train, self._feature_names = _build_training_data(
            scaled_data, self._lags, self._windows, self._include_dates, freq=self.freq
        )

        backend_name = self.get_hyperparameter("backend")
        self._backend_name = backend_name
        early_stopping_rounds = self.get_hyperparameter("early_stopping_rounds")

        # Split for early stopping if using LightGBM/XGBoost
        if early_stopping_rounds and backend_name in ("LightGBM", "XGBoost") and len(X_train) > 200:
            split_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

            self._model = _create_backend(backend_name, self._hyperparameters)
            if backend_name == "LightGBM":
                self._model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        __import__("lightgbm").early_stopping(early_stopping_rounds, verbose=False),
                        __import__("lightgbm").log_evaluation(0),
                    ],
                )
            else:  # XGBoost
                self._model.set_params(early_stopping_rounds=early_stopping_rounds)
                self._model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self._model = _create_backend(backend_name, self._hyperparameters)
            self._model.fit(X_train, y_train)

        # Residual std for prediction intervals
        preds = self._model.predict(X_train)
        residuals = y_train - preds
        self._residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        # Scale input data
        predict_data = data
        if self._target_scaler_obj is not None:
            predict_data = self._target_scaler_obj.transform(data)

        rows = []
        for item_id in predict_data.item_ids:
            item_df = predict_data.loc[item_id]
            raw_series = item_df[TARGET].values.astype(np.float64)
            timestamps = item_df.index.get_level_values(TIMESTAMP)
            future_ts = self._make_future_timestamps(data, item_id)

            # Apply differencing to input series for prediction
            original_series = raw_series.copy()
            series_for_pred = raw_series.copy()
            if self._differences:
                for d in self._differences:
                    if len(series_for_pred) > d:
                        series_for_pred = series_for_pred[d:] - series_for_pred[:-d]
                total_trim = sum(self._differences)
                timestamps = timestamps[total_trim:]

            # Context for prediction — cap to keep predict fast
            max_lag = max(self._lags) if self._lags else 1
            max_win = max(self._windows) if self._windows else 7
            context_size = min(len(series_for_pred), max(max_lag, max_win) + 10)
            series_arr = series_for_pred[-context_size:].copy()
            ctx_timestamps = timestamps[-context_size:]

            # Get inverse scale params
            loc, sc = self._item_scales.get(item_id, (0.0, 1.0))

            # Strategy: for short context (<500) do full rebuild each step (accurate)
            # For long context, use incremental update (fast)
            use_incremental = context_size > 500
            buf = list(series_arr)

            if use_incremental:
                # Build feature matrix ONCE, then incremental update
                X0 = build_feature_matrix(np.array(buf), ctx_timestamps,
                                           self._lags, self._windows,
                                           self._include_dates, freq=self.freq)
                base_row = X0.iloc[-1][self._feature_names].fillna(0).values.astype(np.float64).copy()
                feat_to_idx = {name: i for i, name in enumerate(self._feature_names)}
                lag_indices = [(feat_to_idx[f"lag_{lag}"], lag) for lag in self._lags
                               if f"lag_{lag}" in feat_to_idx]
                diff_indices = [(feat_to_idx[f"diff_{lag}"], lag) for lag in self._lags
                                if lag <= 7 and f"diff_{lag}" in feat_to_idx]
                x_df = pd.DataFrame([base_row], columns=self._feature_names)

            predictions = []
            for h in range(self.prediction_length):
                if use_incremental:
                    x_df.iloc[0] = base_row
                    pred_val = float(self._model.predict(x_df)[0])
                else:
                    # Full rebuild — accurate for short series
                    arr = np.array(buf, dtype=np.float64)
                    ts_ext = ctx_timestamps.append(future_ts[:h]) if h > 0 else ctx_timestamps
                    X = build_feature_matrix(arr, ts_ext, self._lags, self._windows,
                                              self._include_dates, freq=self.freq)
                    x_row = X.iloc[[-1]][self._feature_names].fillna(0)
                    pred_val = float(self._model.predict(x_row)[0])

                predictions.append(pred_val)
                buf.append(pred_val)

                if use_incremental:
                    n = len(buf)
                    for idx, lag in lag_indices:
                        base_row[idx] = buf[n - 1 - lag] if lag < n else 0.0
                    for idx, lag in diff_indices:
                        if lag < n:
                            base_row[idx] = buf[-1] - buf[-(1 + lag)]

            mean_diff = np.array(predictions)

            # Inverse differencing
            if self._differences:
                reconstructed = mean_diff.copy()
                for d_idx in range(len(self._differences) - 1, -1, -1):
                    d = self._differences[d_idx]
                    undiffed = np.empty(len(reconstructed))
                    for i in range(len(reconstructed)):
                        if i < d:
                            ref_idx = len(raw_series) - d + i
                            undiffed[i] = reconstructed[i] + raw_series[ref_idx]
                        else:
                            undiffed[i] = reconstructed[i] + undiffed[i - d]
                    reconstructed = undiffed
                mean = reconstructed
            else:
                mean = mean_diff

            # Inverse scale
            mean_orig = mean * sc + loc
            sigma_orig = self._residual_std * sc

            for i in range(self.prediction_length):
                row = {ITEMID: item_id, TIMESTAMP: future_ts[i], "mean": mean_orig[i]}
                h_factor = np.sqrt(i + 1)
                for q in quantile_levels:
                    z = norm.ppf(q)
                    row[str(q)] = mean_orig[i] + z * sigma_orig * h_factor
                rows.append(row)

        return self._rows_to_tsdf(rows)


# ---------------------------------------------------------------------------
# DirectTabularModel
# ---------------------------------------------------------------------------

@register_model("DirectTabular")
class DirectTabularModel(AbstractTimeSeriesModel):
    """AG-style Direct multi-step forecasting.

    Trains ONE model that predicts all horizon steps simultaneously.
    During training, randomly masks some lag features to teach the model
    to handle unknown future values (AG's key innovation).

    This is much faster than per-horizon models and produces more
    consistent forecasts across the horizon.
    """

    _default_hyperparameters = {
        **RecursiveTabularModel._default_hyperparameters,
        "n_estimators": 2000,
        "target_scaler": "mean_abs",  # AG default: mean_abs for Direct
        "differences": None,  # No differencing for Direct (AG default)
    }

    def _fit(self, train_data, val_data=None, time_limit=None):
        sp = self._get_seasonal_period()
        min_len = min(len(train_data.loc[iid]) for iid in train_data.item_ids)
        all_lags = self.get_hyperparameter("lags") or get_default_lags(sp, freq=self.freq)
        self._lags = [l for l in all_lags if l < min_len - 1] or [1]
        self._windows = [w for w in (self.get_hyperparameter("rolling_windows") or get_default_windows(sp))
                         if w < min_len] or [2]
        self._include_dates = self.get_hyperparameter("include_date_features")

        # Per-item target scaling (AG default: mean_abs)
        scaler_method = self.get_hyperparameter("target_scaler")
        self._item_scales: dict[str, tuple[float, float]] = {}
        scaled_data = train_data
        if scaler_method and scaler_method != "none":
            from myforecaster.models.wrappers import TargetScaler
            self._target_scaler_obj = TargetScaler(method=scaler_method)
            scaled_data = self._target_scaler_obj.fit_transform(train_data)
            self._item_scales = {
                iid: (self._target_scaler_obj._loc.get(iid, 0.0),
                      self._target_scaler_obj._scale.get(iid, 1.0))
                for iid in train_data.item_ids
            }
        else:
            self._target_scaler_obj = None

        # Build training data: one model for all horizons (AG Direct-style)
        # Uses per-item feature matrix + vectorized lag masking + horizon tiling
        rng = np.random.RandomState(123)
        H = self.prediction_length
        max_lag = max(self._lags) if self._lags else 1
        n_lag_feats = len(self._lags)

        X_parts, y_parts = [], []
        feat_names_ref = None

        for item_id in scaled_data.item_ids:
            item_df = scaled_data.loc[item_id]
            series = item_df[TARGET].values.astype(np.float64)
            timestamps = item_df.index.get_level_values(TIMESTAMP)

            X = build_feature_matrix(series, timestamps, self._lags, self._windows,
                                     self._include_dates, freq=self.freq)
            if feat_names_ref is None:
                feat_names_ref = list(X.columns)

            X_np = X.values
            n_valid = len(series) - H - max_lag
            if n_valid <= 0:
                continue

            valid_indices = np.arange(max_lag, max_lag + n_valid)
            n_samples = len(valid_indices)

            # Vectorized lag masking
            X_base = X_np[valid_indices].copy()  # (n_samples, F)
            num_hidden = rng.randint(0, H + 1, size=n_samples)
            if n_lag_feats > 0:
                mask = np.arange(n_lag_feats)[None, :] < num_hidden[:, None]
                X_base[:, :n_lag_feats][mask] = np.nan

            # Tile for all horizons: (n_samples*H, F+1)
            X_tiled = np.repeat(X_base, H, axis=0)
            h_col = np.tile(np.arange(H, dtype=np.float64), n_samples)
            X_block = np.column_stack([X_tiled, h_col])

            # Vectorized targets
            idx_matrix = valid_indices[:, None] + np.arange(1, H + 1)[None, :]
            idx_matrix = np.clip(idx_matrix, 0, len(series) - 1)
            y_block = series[idx_matrix].ravel()

            X_parts.append(X_block)
            y_parts.append(y_block)

        if not X_parts:
            raise ValueError("No valid training samples.")

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)
        self._feature_names = feat_names_ref + ["_horizon"]

        # Fill NaN from masking (LightGBM handles NaN natively)
        backend_name = self.get_hyperparameter("backend")
        self._backend_name = backend_name
        early_stopping_rounds = self.get_hyperparameter("early_stopping_rounds")

        if early_stopping_rounds and backend_name in ("LightGBM", "XGBoost") and len(X_train) > 200:
            split_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

            self._model = _create_backend(backend_name, self._hyperparameters)
            if backend_name == "LightGBM":
                self._model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        __import__("lightgbm").early_stopping(early_stopping_rounds, verbose=False),
                        __import__("lightgbm").log_evaluation(0),
                    ],
                )
            else:
                self._model.set_params(early_stopping_rounds=early_stopping_rounds)
                self._model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self._model = _create_backend(backend_name, self._hyperparameters)
            self._model.fit(X_train, y_train)

        # Residual std for prediction intervals
        preds = self._model.predict(X_train)
        residuals = y_train - preds
        self._residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        predict_data = data
        if getattr(self, "_target_scaler_obj", None) is not None:
            predict_data = self._target_scaler_obj.transform(data)

        n_lag_feats = len(self._lags)
        H = self.prediction_length

        rows = []
        for item_id in predict_data.item_ids:
            item_df = predict_data.loc[item_id]
            series = item_df[TARGET].values.astype(np.float64)
            timestamps = item_df.index.get_level_values(TIMESTAMP)
            future_ts = self._make_future_timestamps(data, item_id)
            loc, sc = self._item_scales.get(item_id, (0.0, 1.0))

            X = build_feature_matrix(series, timestamps, self._lags, self._windows,
                                     self._include_dates, freq=self.freq)
            x_base = X.iloc[-1].values.copy()

            # Batch predict: tile base features for all H horizons
            X_pred = np.tile(x_base, (H, 1))  # (H, F)
            for h in range(H):
                for lag_idx, lag_val in enumerate(self._lags):
                    if lag_val <= h and lag_idx < n_lag_feats:
                        X_pred[h, lag_idx] = np.nan
            h_col = np.arange(H, dtype=np.float64).reshape(-1, 1)
            X_pred_full = np.hstack([X_pred, h_col])

            preds_scaled = self._model.predict(X_pred_full)
            preds_orig = preds_scaled * sc + loc
            sigma = self._residual_std * sc

            for i in range(H):
                row = {ITEMID: item_id, TIMESTAMP: future_ts[i], "mean": float(preds_orig[i])}
                for q in quantile_levels:
                    z = norm.ppf(q)
                    row[str(q)] = float(preds_orig[i]) + z * sigma
                rows.append(row)

        return self._rows_to_tsdf(rows)


# ---------------------------------------------------------------------------
# Convenience shortcut registrations
# ---------------------------------------------------------------------------

# These let users write hyperparameters={"LightGBM": {}, "XGBoost": {}}
# in the predictor, with Recursive as default strategy.
_BACKEND_SHORTCUTS = {
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
}

for _reg_name, _backend in _BACKEND_SHORTCUTS.items():
    # Only register if not already taken by another model
    from myforecaster.models import MODEL_REGISTRY
    if _reg_name not in MODEL_REGISTRY:

        def _make_shortcut(backend_name):
            class _SC(RecursiveTabularModel):
                _default_hyperparameters = {
                    **RecursiveTabularModel._default_hyperparameters,
                    "backend": backend_name,
                }
            _SC.__name__ = f"{backend_name}Model"
            _SC.__qualname__ = f"{backend_name}Model"
            return _SC

        register_model(_reg_name)(_make_shortcut(_backend))
