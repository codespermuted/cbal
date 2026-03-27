"""Step 5-b: StatsForecast wrapper verification tests."""

import numpy as np
import pandas as pd
import pytest

from myforecaster.dataset import TimeSeriesDataFrame
from myforecaster.models import MODEL_REGISTRY, list_models
from myforecaster.models.statsforecast import StatsForecastModel, list_statsforecast_models


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_tsdf():
    """2 items, 100 daily observations, trend + seasonality + noise."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rows = []
    for item_id in ["A", "B"]:
        base = 100 if item_id == "A" else 200
        for i, d in enumerate(dates):
            val = base + i * 0.3 + 15 * np.sin(2 * np.pi * i / 7) + np.random.randn() * 2
            rows.append({"item_id": item_id, "timestamp": d, "target": val})
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


@pytest.fixture
def train_test(daily_tsdf):
    return daily_tsdf.train_test_split(14)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
class TestDiscovery:
    def test_list_statsforecast_models(self):
        models = list_statsforecast_models()
        assert len(models) > 20
        assert "AutoARIMA" in models
        assert "AutoETS" in models
        assert "AutoTheta" in models

    def test_popular_models_registered(self):
        registered = list_models()
        for name in ["AutoARIMA", "AutoETS", "AutoTheta", "AutoCES", "MSTL",
                      "CrostonSBA", "ADIDA", "Theta", "ETS"]:
            assert name in registered, f"{name} not registered"


# ---------------------------------------------------------------------------
# Core wrapper: fit + predict structure
# ---------------------------------------------------------------------------
CORE_MODELS = ["AutoETS", "AutoTheta", "AutoCES"]


class TestCoreFitPredict:
    @pytest.mark.parametrize("model_name", CORE_MODELS)
    def test_fit_predict_shape(self, model_name, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            model_name=model_name, freq="D", prediction_length=14,
            hyperparameters={"model_name": model_name}
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 2 * 14  # 2 items * 14 steps

    @pytest.mark.parametrize("model_name", CORE_MODELS)
    def test_has_mean_column(self, model_name, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            model_name=model_name, freq="D", prediction_length=14,
            hyperparameters={"model_name": model_name}
        )
        m.fit(train)
        pred = m.predict(train)
        assert "mean" in pred.columns

    @pytest.mark.parametrize("model_name", CORE_MODELS)
    def test_quantile_columns(self, model_name, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            model_name=model_name, freq="D", prediction_length=14,
            hyperparameters={"model_name": model_name}
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert "0.1" in pred.columns
        assert "0.5" in pred.columns
        assert "0.9" in pred.columns

    @pytest.mark.parametrize("model_name", CORE_MODELS)
    def test_quantile_ordering(self, model_name, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            model_name=model_name, freq="D", prediction_length=14,
            hyperparameters={"model_name": model_name}
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert (pred["0.1"].values <= pred["0.5"].values + 1e-6).all()
        assert (pred["0.5"].values <= pred["0.9"].values + 1e-6).all()

    @pytest.mark.parametrize("model_name", CORE_MODELS)
    def test_future_timestamps(self, model_name, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            model_name=model_name, freq="D", prediction_length=14,
            hyperparameters={"model_name": model_name}
        )
        m.fit(train)
        pred = m.predict(train)
        for item_id in train.item_ids:
            last_ts = train.loc[item_id].index.get_level_values("timestamp").max()
            pred_ts = pred.loc[item_id].index.get_level_values("timestamp")
            assert pred_ts.min() > last_ts


# ---------------------------------------------------------------------------
# AutoARIMA (most important model — tested separately)
# ---------------------------------------------------------------------------
class TestAutoARIMA:
    def test_fit_predict(self, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            model_name="AutoARIMA", freq="D", prediction_length=14,
            hyperparameters={"model_name": "AutoARIMA"}
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 28
        assert "mean" in pred.columns

    def test_score(self, train_test):
        train, test = train_test
        m = StatsForecastModel(
            model_name="AutoARIMA", freq="D", prediction_length=14,
            hyperparameters={"model_name": "AutoARIMA"}
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)
        assert score > 0

    def test_with_model_kwargs(self, train_test):
        """Pass extra kwargs to the underlying model."""
        train, _ = train_test
        m = StatsForecastModel(
            freq="D", prediction_length=14,
            hyperparameters={
                "model_name": "AutoARIMA",
                "model_kwargs": {"stepwise": True, "max_p": 3, "max_q": 3},
            }
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 28


# ---------------------------------------------------------------------------
# Shortcut classes via MODEL_REGISTRY
# ---------------------------------------------------------------------------
class TestShortcuts:
    def test_registry_lookup(self, train_test):
        """Access model via registry string name."""
        train, _ = train_test
        cls = MODEL_REGISTRY["AutoETS"]
        m = cls(freq="D", prediction_length=14)
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 28

    def test_ets_alias(self, train_test):
        """'ETS' should map to AutoETS."""
        train, _ = train_test
        cls = MODEL_REGISTRY["ETS"]
        m = cls(freq="D", prediction_length=14)
        m.fit(train)
        assert m.get_hyperparameter("model_name") == "AutoETS"

    def test_ces_alias(self, train_test):
        """'CES' should map to AutoCES."""
        train, _ = train_test
        cls = MODEL_REGISTRY["CES"]
        m = cls(freq="D", prediction_length=14)
        m.fit(train)
        assert m.get_hyperparameter("model_name") == "AutoCES"


# ---------------------------------------------------------------------------
# Intermittent demand models
# ---------------------------------------------------------------------------
class TestIntermittentModels:
    @pytest.fixture
    def sparse_tsdf(self):
        """Sparse time series with many zeros (intermittent demand)."""
        np.random.seed(42)
        n = 100
        values = np.zeros(n)
        # Random demand events
        demand_idx = np.random.choice(n, size=20, replace=False)
        values[demand_idx] = np.random.poisson(5, size=20).astype(float)
        df = pd.DataFrame({
            "item_id": ["sparse"] * n,
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="D"),
            "target": values,
        })
        return TimeSeriesDataFrame.from_data_frame(df)

    @pytest.mark.parametrize("model_name", ["CrostonSBA", "CrostonClassic", "ADIDA"])
    def test_intermittent_fit_predict(self, model_name, sparse_tsdf):
        m = StatsForecastModel(
            model_name=model_name, freq="D", prediction_length=7,
            hyperparameters={"model_name": model_name}
        )
        m.fit(sparse_tsdf)
        pred = m.predict(sparse_tsdf)
        assert len(pred) == 7
        assert "mean" in pred.columns
        # Predictions should be non-negative for demand
        assert (pred["mean"].values >= -1e-6).all()

    def test_tsb_with_params(self, sparse_tsdf):
        """TSB requires alpha_d and alpha_p parameters."""
        m = StatsForecastModel(
            model_name="TSB", freq="D", prediction_length=7,
            hyperparameters={
                "model_name": "TSB",
                "model_kwargs": {"alpha_d": 0.1, "alpha_p": 0.1},
            }
        )
        m.fit(sparse_tsdf)
        pred = m.predict(sparse_tsdf)
        assert len(pred) == 7


# ---------------------------------------------------------------------------
# MSTL (multiple seasonalities)
# ---------------------------------------------------------------------------
class TestMSTL:
    def test_mstl_fit_predict(self, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            freq="D", prediction_length=14,
            hyperparameters={"model_name": "MSTL", "season_length": 7}
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 28


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_unknown_model_raises(self, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            freq="D", prediction_length=14,
            hyperparameters={"model_name": "NonexistentModel"}
        )
        with pytest.raises(ValueError, match="Unknown statsforecast model"):
            m.fit(train)

    def test_auto_name_from_model_name(self):
        """Model name should default to the statsforecast model name."""
        m = StatsForecastModel(
            freq="D", prediction_length=7,
            hyperparameters={"model_name": "AutoTheta"}
        )
        assert m.name == "AutoTheta"


# ---------------------------------------------------------------------------
# Season length inference
# ---------------------------------------------------------------------------
class TestSeasonLength:
    def test_auto_infer_daily(self, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            freq="D", prediction_length=7,
            hyperparameters={"model_name": "AutoETS"}
        )
        m.fit(train)
        assert m._season_length == 7  # daily -> weekly

    def test_explicit_override(self, train_test):
        train, _ = train_test
        m = StatsForecastModel(
            freq="D", prediction_length=7,
            hyperparameters={"model_name": "AutoETS", "season_length": 30}
        )
        m.fit(train)
        assert m._season_length == 30
