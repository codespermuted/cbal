"""Step 5-c: Tabular models verification tests."""

import numpy as np
import pandas as pd
import pytest

from cbal.dataset import TimeSeriesDataFrame
from cbal.models import MODEL_REGISTRY, list_models
from cbal.models.tabular import (
    RecursiveTabularModel,
    DirectTabularModel,
    list_tabular_backends,
)
from cbal.models.tabular.features import (
    build_feature_matrix,
    create_lag_features,
    create_rolling_features,
    create_date_features,
    get_default_lags,
    get_default_windows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_tsdf():
    """2 items, 200 daily obs, trend + seasonality + noise."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    rows = []
    for item_id in ["A", "B"]:
        base = 100 if item_id == "A" else 200
        for i, d in enumerate(dates):
            val = base + i * 0.3 + 10 * np.sin(2 * np.pi * i / 7) + np.random.randn() * 2
            rows.append({"item_id": item_id, "timestamp": d, "target": val})
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


@pytest.fixture
def pred_length():
    return 7


@pytest.fixture
def train_test(daily_tsdf, pred_length):
    return daily_tsdf.train_test_split(pred_length)


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_lag_features(self):
        series = np.arange(10, dtype=float)
        lags = create_lag_features(series, [1, 3])
        assert "lag_1" in lags
        assert "lag_3" in lags
        assert np.isnan(lags["lag_1"][0])
        assert lags["lag_1"][1] == 0.0

    def test_rolling_features(self):
        series = np.arange(20, dtype=float)
        rolling = create_rolling_features(series, [3, 7])
        assert "rolling_mean_3" in rolling
        assert "rolling_std_7" in rolling
        assert "rolling_min_3" in rolling
        assert "rolling_max_3" in rolling

    # --- Daily: AG-style normalized encoding ---
    def test_date_features_daily(self):
        ts = pd.date_range("2023-01-01", periods=10, freq="D")
        feats = create_date_features(ts, freq="D")
        assert "day_of_week" in feats     # normalized [-0.5, 0.5]
        assert "month" in feats
        assert "day_of_month" in feats
        assert "week_of_year" in feats
        assert "is_weekend" in feats

    # --- Hourly ---
    def test_date_features_hourly(self):
        ts = pd.date_range("2023-01-01", periods=48, freq="h")
        feats = create_date_features(ts, freq="h")
        assert "hour" in feats
        assert "day_of_week" in feats
        assert "month" in feats
        assert "day_of_month" in feats

    # --- Minute ---
    def test_date_features_minute(self):
        ts = pd.date_range("2023-01-01", periods=120, freq="min")
        feats = create_date_features(ts, freq="min")
        assert "hour" in feats
        assert "minute" in feats
        assert "day_of_week" in feats

    # --- Second ---
    def test_date_features_second(self):
        ts = pd.date_range("2023-01-01", periods=120, freq="s")
        feats = create_date_features(ts, freq="s")
        assert "second" in feats
        assert "minute" in feats
        assert "hour" in feats

    # --- Cyclic encoding (for DL) ---
    def test_date_features_cyclic(self):
        ts = pd.date_range("2023-01-01", periods=48, freq="h")
        feats = create_date_features(ts, freq="h", encoding="cyclic")
        assert "hour_sin" in feats
        assert "hour_cos" in feats
        assert "day_of_week_sin" in feats

    # --- None freq defaults to daily ---
    def test_date_features_none_freq(self):
        ts = pd.date_range("2023-01-01", periods=10, freq="D")
        feats = create_date_features(ts, freq=None)
        assert "day_of_week" in feats

    # --- build_feature_matrix ---
    def test_build_feature_matrix_daily(self):
        series = np.arange(50, dtype=float)
        ts = pd.date_range("2023-01-01", periods=50, freq="D")
        X = build_feature_matrix(series, ts, lags=[1, 7], windows=[7], freq="D")
        assert len(X) == 50
        assert "lag_1" in X.columns
        assert "rolling_mean_7" in X.columns
        assert "day_of_week" in X.columns

    def test_build_feature_matrix_hourly(self):
        series = np.arange(100, dtype=float)
        ts = pd.date_range("2023-01-01", periods=100, freq="h")
        X = build_feature_matrix(series, ts, lags=[1, 24], windows=[24], freq="h")
        assert "hour" in X.columns

    def test_default_lags(self):
        lags = get_default_lags(7)
        assert 1 in lags
        assert 7 in lags

    def test_default_windows(self):
        windows = get_default_windows(7)
        assert 7 in windows


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------
class TestBackendDiscovery:
    def test_list_backends(self):
        backends = list_tabular_backends()
        assert "LightGBM" in backends
        assert "XGBoost" in backends
        assert "CatBoost" in backends
        assert "RandomForest" in backends
        assert "LinearRegression" in backends

    def test_shortcut_registrations(self):
        registered = list_models()
        assert "RecursiveTabular" in registered
        assert "DirectTabular" in registered
        assert "LightGBM" in registered
        assert "XGBoost" in registered
        assert "CatBoost" in registered


# ---------------------------------------------------------------------------
# RecursiveTabularModel — GBDT backends
# ---------------------------------------------------------------------------
GBDT_BACKENDS = ["LightGBM", "XGBoost", "CatBoost"]


class TestRecursiveGBDT:
    @pytest.mark.parametrize("backend", GBDT_BACKENDS)
    def test_fit_predict_shape(self, backend, train_test, pred_length):
        train, _ = train_test
        m = RecursiveTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": backend, "n_estimators": 20},
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 2 * pred_length

    @pytest.mark.parametrize("backend", GBDT_BACKENDS)
    def test_has_mean_and_quantiles(self, backend, train_test, pred_length):
        train, _ = train_test
        m = RecursiveTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": backend, "n_estimators": 20},
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert "mean" in pred.columns
        assert "0.1" in pred.columns
        assert "0.9" in pred.columns

    @pytest.mark.parametrize("backend", GBDT_BACKENDS)
    def test_quantile_ordering(self, backend, train_test, pred_length):
        train, _ = train_test
        m = RecursiveTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": backend, "n_estimators": 20},
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert (pred["0.1"].values <= pred["0.5"].values + 1e-6).all()
        assert (pred["0.5"].values <= pred["0.9"].values + 1e-6).all()

    @pytest.mark.parametrize("backend", GBDT_BACKENDS)
    def test_score_is_finite(self, backend, train_test, pred_length):
        train, test = train_test
        m = RecursiveTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": backend, "n_estimators": 20},
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)
        assert score > 0


# ---------------------------------------------------------------------------
# RecursiveTabularModel — sklearn backends
# ---------------------------------------------------------------------------
SKLEARN_BACKENDS = ["RandomForest", "ExtraTrees", "Ridge", "LinearRegression"]


class TestRecursiveSklearn:
    @pytest.mark.parametrize("backend", SKLEARN_BACKENDS)
    def test_fit_predict(self, backend, train_test, pred_length):
        train, _ = train_test
        m = RecursiveTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": backend, "n_estimators": 20},
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 2 * pred_length
        assert "mean" in pred.columns


# ---------------------------------------------------------------------------
# DirectTabularModel
# ---------------------------------------------------------------------------
class TestDirect:
    def test_fit_predict_lightgbm(self, train_test, pred_length):
        train, _ = train_test
        m = DirectTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": "LightGBM", "n_estimators": 20},
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 2 * pred_length
        assert "mean" in pred.columns

    def test_fit_predict_xgboost(self, train_test, pred_length):
        train, _ = train_test
        m = DirectTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": "XGBoost", "n_estimators": 20},
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 2 * pred_length

    def test_trains_single_model(self, train_test, pred_length):
        """AG-style Direct trains ONE model with horizon indicator."""
        train, _ = train_test
        m = DirectTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": "LightGBM", "n_estimators": 10},
        )
        m.fit(train)
        assert m._model is not None  # single model, not per-horizon

    def test_quantile_ordering(self, train_test, pred_length):
        train, _ = train_test
        m = DirectTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": "LightGBM", "n_estimators": 20},
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert (pred["0.1"].values <= pred["0.5"].values + 1e-6).all()
        assert (pred["0.5"].values <= pred["0.9"].values + 1e-6).all()

    def test_score(self, train_test, pred_length):
        train, test = train_test
        m = DirectTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": "LightGBM", "n_estimators": 20},
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Future timestamps
# ---------------------------------------------------------------------------
class TestTimestamps:
    def test_recursive_future_timestamps(self, train_test, pred_length):
        train, _ = train_test
        m = RecursiveTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": "LightGBM", "n_estimators": 10},
        )
        m.fit(train)
        pred = m.predict(train)
        for item_id in train.item_ids:
            last_ts = train.loc[item_id].index.get_level_values("timestamp").max()
            pred_ts = pred.loc[item_id].index.get_level_values("timestamp")
            assert pred_ts.min() > last_ts

    def test_direct_future_timestamps(self, train_test, pred_length):
        train, _ = train_test
        m = DirectTabularModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"backend": "LightGBM", "n_estimators": 10},
        )
        m.fit(train)
        pred = m.predict(train)
        for item_id in train.item_ids:
            last_ts = train.loc[item_id].index.get_level_values("timestamp").max()
            pred_ts = pred.loc[item_id].index.get_level_values("timestamp")
            assert pred_ts.min() > last_ts


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrors:
    def test_unknown_backend_raises(self, train_test):
        train, _ = train_test
        m = RecursiveTabularModel(
            freq="D", prediction_length=7,
            hyperparameters={"backend": "NonexistentBackend"},
        )
        with pytest.raises(ValueError, match="Unknown tabular backend"):
            m.fit(train)


# ---------------------------------------------------------------------------
# Shortcut via MODEL_REGISTRY
# ---------------------------------------------------------------------------
class TestShortcuts:
    def test_lightgbm_shortcut(self, train_test, pred_length):
        train, _ = train_test
        cls = MODEL_REGISTRY["LightGBM"]
        m = cls(freq="D", prediction_length=pred_length,
                hyperparameters={"n_estimators": 10})
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 2 * pred_length
