"""
Model wrappers — TargetScaler and CovariateRegressor.

These are decorator-pattern wrappers that add preprocessing capabilities
to any base forecasting model, matching AutoGluon's architecture.

TargetScaler
    Wraps fit/predict: scale target → delegate → inverse-scale predictions.
    Makes DL models converge faster on heterogeneous-scale data.

CovariateRegressor
    Wraps fit/predict: train a regression model (LightGBM) on covariates,
    subtract covariate effect → delegate base model → add back.
    Makes ANY model covariate-aware, even univariate ones.

Usage::

    from cbal.models.wrappers import TargetScaler, CovariateRegressor

    # Wrap any model
    model = TargetScaler(base_model, method="standard")
    model = CovariateRegressor(base_model, regressor_backend="lightgbm")
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cbal.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame

logger = logging.getLogger(__name__)


# =====================================================================
# TargetScaler
# =====================================================================

class TargetScaler:
    """Scale the target per-item before training, inverse-scale predictions.

    Supports four scaling methods (same as AutoGluon):
    - ``"standard"`` — zero mean, unit variance: (y - mean) / std
    - ``"mean_abs"`` — divide by mean absolute value: y / mean(|y|)
    - ``"robust"``   — median / IQR: (y - median) / IQR
    - ``"min_max"``  — into [0, 1]: (y - min) / (max - min)

    Parameters
    ----------
    method : str
        One of ``"standard"``, ``"mean_abs"``, ``"robust"``, ``"min_max"``.
    """

    def __init__(self, method: str = "standard"):
        if method not in ("standard", "mean_abs", "robust", "min_max"):
            raise ValueError(
                f"Unknown scaler method '{method}'. "
                f"Choose from: standard, mean_abs, robust, min_max"
            )
        self.method = method
        self._loc: dict[str, float] = {}   # item_id → location
        self._scale: dict[str, float] = {} # item_id → scale

    def fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Compute per-item scale params and return scaled data."""
        scaled = data.copy()
        for item_id in data.item_ids:
            y = data.loc[item_id][TARGET].values.astype(float)
            loc, scale = self._compute_params(y)
            self._loc[item_id] = loc
            self._scale[item_id] = scale
            mask = scaled.index.get_level_values(ITEMID) == item_id
            scaled.loc[mask, TARGET] = (
                scaled.loc[mask, TARGET].values - loc
            ) / max(scale, 1e-8)
        data._propagate_metadata(scaled)
        return scaled

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Scale data using previously fitted params."""
        scaled = data.copy()
        for item_id in data.item_ids:
            loc = self._loc.get(item_id, 0.0)
            scale = self._scale.get(item_id, 1.0)
            mask = scaled.index.get_level_values(ITEMID) == item_id
            scaled.loc[mask, TARGET] = (
                scaled.loc[mask, TARGET].values - loc
            ) / max(scale, 1e-8)
        data._propagate_metadata(scaled)
        return scaled

    def inverse_transform_predictions(
        self, preds: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        """Inverse-scale prediction columns (mean + quantiles)."""
        result = preds.copy()
        for item_id in preds.item_ids:
            loc = self._loc.get(item_id, 0.0)
            scale = self._scale.get(item_id, 1.0)
            mask = result.index.get_level_values(ITEMID) == item_id
            for col in result.columns:
                result.loc[mask, col] = (
                    result.loc[mask, col].values * scale + loc
                )
        return result

    def _compute_params(self, y: np.ndarray) -> tuple[float, float]:
        y = y[np.isfinite(y)]
        if len(y) == 0:
            return 0.0, 1.0
        if self.method == "standard":
            return float(np.mean(y)), float(max(np.std(y), 1e-8))
        elif self.method == "mean_abs":
            return 0.0, float(max(np.mean(np.abs(y)), 1e-8))
        elif self.method == "robust":
            q25, q50, q75 = np.percentile(y, [25, 50, 75])
            iqr = max(q75 - q25, 1e-8)
            return float(q50), float(iqr)
        elif self.method == "min_max":
            mn, mx = float(np.min(y)), float(np.max(y))
            return mn, max(mx - mn, 1e-8)
        return 0.0, 1.0


# =====================================================================
# CovariateRegressor
# =====================================================================

class CovariateRegressor:
    """Make any model covariate-aware by residual regression.

    Architecture (same as AutoGluon's covariate_regressor):
    1. Train a regression model on covariates → predict target
    2. Subtract covariate effect from target → residuals
    3. Base forecasting model learns residuals (covariate-free)
    4. At predict time: base_prediction + covariate_prediction

    Parameters
    ----------
    known_covariates_names : list of str
        Columns used as known-future covariates.
    past_covariates_names : list of str
        Columns used as past-only covariates.
    static_features_names : list of str
        Static feature columns.
    backend : str
        Regressor backend: ``"lightgbm"`` or ``"linear"``.
    """

    def __init__(
        self,
        known_covariates_names: list[str] | None = None,
        past_covariates_names: list[str] | None = None,
        static_features_names: list[str] | None = None,
        backend: str = "lightgbm",
    ):
        self.known_names = known_covariates_names or []
        self.past_names = past_covariates_names or []
        self.static_names = static_features_names or []
        self.backend = backend
        self._regressor = None
        self._is_fitted = False

    def _build_features(
        self,
        data: TimeSeriesDataFrame,
        static_features: pd.DataFrame | None = None,
        future: bool = False,
    ) -> pd.DataFrame:
        """Build feature matrix from covariates + static features."""
        features = {}

        # Time-varying covariates present in data columns
        for col in self.known_names:
            if col in data.columns:
                features[col] = data[col].values

        if not future:
            for col in self.past_names:
                if col in data.columns:
                    features[col] = data[col].values

        # Static features: broadcast to each row
        if static_features is not None and self.static_names:
            item_ids = data.index.get_level_values(ITEMID)
            for col in self.static_names:
                if col in static_features.columns:
                    mapping = static_features[col].to_dict()
                    features[f"static_{col}"] = [
                        mapping.get(iid, np.nan) for iid in item_ids
                    ]

        if not features:
            return pd.DataFrame(index=data.index)

        feat_df = pd.DataFrame(features, index=data.index)
        # Convert categoricals to codes
        for col in feat_df.select_dtypes(include=["object", "category"]).columns:
            feat_df[col] = feat_df[col].astype("category").cat.codes
        return feat_df

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        static_features: pd.DataFrame | None = None,
    ):
        """Fit the covariate regressor on training data."""
        X = self._build_features(train_data, static_features)
        if X.empty or X.shape[1] == 0:
            logger.info("CovariateRegressor: no covariates found, skipping.")
            self._is_fitted = False
            return

        y = train_data[TARGET].values
        mask = np.isfinite(y)
        X_clean = X.iloc[mask]
        y_clean = y[mask]

        if self.backend == "lightgbm":
            try:
                import lightgbm as lgb
                self._regressor = lgb.LGBMRegressor(
                    n_estimators=100, learning_rate=0.1,
                    max_depth=5, verbose=-1, n_jobs=-1,
                )
                self._regressor.fit(X_clean, y_clean)
            except ImportError:
                logger.warning("CovariateRegressor: lightgbm not installed, using linear.")
                self.backend = "linear"

        if self.backend == "linear":
            from sklearn.linear_model import Ridge
            self._regressor = Ridge(alpha=1.0)
            self._regressor.fit(X_clean, y_clean)

        self._is_fitted = True
        logger.info(f"CovariateRegressor fitted ({self.backend}, {X.shape[1]} features)")

    def remove_covariate_effect(
        self,
        data: TimeSeriesDataFrame,
        static_features: pd.DataFrame | None = None,
    ) -> TimeSeriesDataFrame:
        """Subtract covariate prediction from target → residuals."""
        if not self._is_fitted:
            return data

        X = self._build_features(data, static_features)
        if X.empty:
            return data

        residual_data = data.copy()
        cov_pred = self._regressor.predict(X)
        residual_data[TARGET] = data[TARGET].values - cov_pred
        data._propagate_metadata(residual_data)
        return residual_data

    def add_covariate_effect(
        self,
        preds: TimeSeriesDataFrame,
        future_data: TimeSeriesDataFrame | None = None,
        known_covariates: TimeSeriesDataFrame | None = None,
        static_features: pd.DataFrame | None = None,
    ) -> TimeSeriesDataFrame:
        """Add covariate prediction back to base model predictions."""
        if not self._is_fitted or known_covariates is None:
            return preds

        X = self._build_features(known_covariates, static_features, future=True)
        if X.empty:
            return preds

        cov_pred = self._regressor.predict(X)
        result = preds.copy()

        # Add covariate effect to all prediction columns
        for col in result.columns:
            if col in ("mean",) or col.replace(".", "").replace("-", "").isdigit():
                # Align by position within each item
                idx = 0
                for item_id in result.item_ids:
                    item_mask = result.index.get_level_values(ITEMID) == item_id
                    n = item_mask.sum()
                    result.loc[item_mask, col] = (
                        result.loc[item_mask, col].values + cov_pred[idx:idx+n]
                    )
                    idx += n

        return result


# =====================================================================
# CovariateScaler
# =====================================================================

class CovariateScaler:
    """Scale covariates and static features before model training.

    AutoGluon's ``"global"`` covariate_scaler applies:
    - QuantileTransform for skewed numeric features
    - StandardScaler for other numeric features
    - Passthrough for boolean features

    Parameters
    ----------
    method : str
        ``"global"`` — auto-detect skewed vs normal.
        ``"standard"`` — StandardScaler for all numeric.
    """

    def __init__(self, method: str = "global"):
        self.method = method
        self._scalers: dict = {}
        self._is_fitted = False

    def fit_transform(self, data, static_features=None):
        """Fit scalers and transform covariates."""
        from sklearn.preprocessing import StandardScaler, QuantileTransformer

        result = data.copy()
        cov_cols = [c for c in data.columns if c != TARGET]

        for col in cov_cols:
            if data[col].dtype == bool:
                continue
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            vals = data[col].values.reshape(-1, 1)
            finite_mask = np.isfinite(vals.ravel())
            if finite_mask.sum() < 10:
                continue

            if self.method == "global":
                skew = float(pd.Series(vals[finite_mask].ravel()).skew())
                if abs(skew) > 1.0:
                    scaler = QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=min(1000, int(finite_mask.sum())),
                    )
                else:
                    scaler = StandardScaler()
            else:
                scaler = StandardScaler()

            clean_vals = np.where(finite_mask.reshape(-1, 1), vals, 0.0)
            scaler.fit(clean_vals)
            transformed = scaler.transform(clean_vals)
            transformed[~finite_mask.reshape(-1, 1)] = np.nan
            result[col] = transformed.ravel()
            self._scalers[col] = scaler

        result_static = None
        if static_features is not None:
            result_static = static_features.copy()
            for col in static_features.columns:
                if not pd.api.types.is_numeric_dtype(static_features[col]):
                    continue
                vals = static_features[col].values.reshape(-1, 1)
                scaler = StandardScaler()
                scaler.fit(vals)
                result_static[col] = scaler.transform(vals).ravel()
                self._scalers[f"static_{col}"] = scaler

        self._is_fitted = True
        data._propagate_metadata(result)
        return result, result_static

    def transform(self, data, static_features=None):
        """Transform using fitted scalers."""
        if not self._is_fitted:
            return data, static_features
        result = data.copy()
        for col, scaler in self._scalers.items():
            if col.startswith("static_"):
                continue
            if col in result.columns:
                vals = result[col].values.reshape(-1, 1)
                finite = np.isfinite(vals.ravel())
                clean = np.where(finite.reshape(-1, 1), vals, 0.0)
                transformed = scaler.transform(clean)
                transformed[~finite.reshape(-1, 1)] = np.nan
                result[col] = transformed.ravel()

        result_static = static_features
        if static_features is not None:
            result_static = static_features.copy()
            for col in static_features.columns:
                key = f"static_{col}"
                if key in self._scalers:
                    vals = static_features[col].values.reshape(-1, 1)
                    result_static[col] = self._scalers[key].transform(vals).ravel()

        data._propagate_metadata(result)
        return result, result_static
