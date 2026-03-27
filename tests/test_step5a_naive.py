"""Step 5-a: Naive models verification tests."""

import numpy as np
import pandas as pd
import pytest

from myforecaster.dataset import TimeSeriesDataFrame
from myforecaster.models import MODEL_REGISTRY, list_models
from myforecaster.models.naive import (
    AverageModel,
    DriftModel,
    NaiveModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_tsdf():
    """2 items, 50 daily observations each, with a trend + seasonality."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    rows = []
    for item_id in ["A", "B"]:
        base = 100 if item_id == "A" else 200
        for i, d in enumerate(dates):
            val = base + i * 0.5 + 10 * np.sin(2 * np.pi * i / 7) + np.random.randn()
            rows.append({"item_id": item_id, "timestamp": d, "target": val})
    df = pd.DataFrame(rows)
    return TimeSeriesDataFrame.from_data_frame(df)


@pytest.fixture
def pred_length():
    return 7


@pytest.fixture
def train_test(daily_tsdf, pred_length):
    return daily_tsdf.train_test_split(pred_length)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegistration:
    def test_all_models_registered(self):
        names = list_models()
        for expected in ["Naive", "SeasonalNaive", "Average", "SeasonalAverage", "Drift"]:
            assert expected in names, f"{expected} not in MODEL_REGISTRY"

    def test_registry_returns_classes(self):
        assert MODEL_REGISTRY["Naive"] is NaiveModel
        assert MODEL_REGISTRY["SeasonalNaive"] is SeasonalNaiveModel
        assert MODEL_REGISTRY["Average"] is AverageModel
        assert MODEL_REGISTRY["SeasonalAverage"] is SeasonalAverageModel
        assert MODEL_REGISTRY["Drift"] is DriftModel


# ---------------------------------------------------------------------------
# Shared prediction structure tests (run for all 5 models)
# ---------------------------------------------------------------------------
ALL_NAIVE = [NaiveModel, SeasonalNaiveModel, AverageModel, SeasonalAverageModel, DriftModel]


class TestPredictionStructure:
    @pytest.mark.parametrize("ModelClass", ALL_NAIVE)
    def test_predict_shape(self, ModelClass, train_test, pred_length):
        train, _ = train_test
        m = ModelClass(freq="D", prediction_length=pred_length)
        m.fit(train)
        pred = m.predict(train)
        # 2 items * 7 steps = 14 rows
        assert len(pred) == 2 * pred_length

    @pytest.mark.parametrize("ModelClass", ALL_NAIVE)
    def test_predict_has_mean(self, ModelClass, train_test, pred_length):
        train, _ = train_test
        m = ModelClass(freq="D", prediction_length=pred_length)
        m.fit(train)
        pred = m.predict(train)
        assert "mean" in pred.columns

    @pytest.mark.parametrize("ModelClass", ALL_NAIVE)
    def test_predict_has_quantiles(self, ModelClass, train_test, pred_length):
        train, _ = train_test
        m = ModelClass(freq="D", prediction_length=pred_length)
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.05, 0.5, 0.95])
        assert "0.05" in pred.columns
        assert "0.5" in pred.columns
        assert "0.95" in pred.columns

    @pytest.mark.parametrize("ModelClass", ALL_NAIVE)
    def test_quantile_ordering(self, ModelClass, train_test, pred_length):
        """Lower quantile <= median <= upper quantile."""
        train, _ = train_test
        m = ModelClass(freq="D", prediction_length=pred_length)
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert (pred["0.1"].values <= pred["0.5"].values + 1e-10).all()
        assert (pred["0.5"].values <= pred["0.9"].values + 1e-10).all()

    @pytest.mark.parametrize("ModelClass", ALL_NAIVE)
    def test_future_timestamps(self, ModelClass, train_test, pred_length):
        """All prediction timestamps are after the last training timestamp."""
        train, _ = train_test
        m = ModelClass(freq="D", prediction_length=pred_length)
        m.fit(train)
        pred = m.predict(train)
        for item_id in train.item_ids:
            last_train = train.loc[item_id].index.get_level_values("timestamp").max()
            pred_ts = pred.loc[item_id].index.get_level_values("timestamp")
            assert pred_ts.min() > last_train

    @pytest.mark.parametrize("ModelClass", ALL_NAIVE)
    def test_score_is_finite(self, ModelClass, train_test, pred_length):
        train, test = train_test
        m = ModelClass(freq="D", prediction_length=pred_length)
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)
        assert score >= 0


# ---------------------------------------------------------------------------
# NaiveModel — specific
# ---------------------------------------------------------------------------
class TestNaiveModel:
    def test_predicts_last_value(self):
        """Naive should repeat the last observed value."""
        df = pd.DataFrame({
            "item_id": ["A"] * 10,
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = NaiveModel(freq="D", prediction_length=3)
        m.fit(tsdf)
        pred = m.predict(tsdf)
        np.testing.assert_array_almost_equal(pred["mean"].values, [10, 10, 10])

    def test_expanding_intervals(self):
        """Prediction intervals should widen over the horizon."""
        np.random.seed(0)
        df = pd.DataFrame({
            "item_id": ["A"] * 100,
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "target": np.random.randn(100).cumsum(),
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = NaiveModel(freq="D", prediction_length=10)
        m.fit(tsdf)
        pred = m.predict(tsdf, quantile_levels=[0.1, 0.9])
        widths = pred["0.9"].values - pred["0.1"].values
        # Width at h=10 should be > width at h=1
        assert widths[-1] > widths[0]


# ---------------------------------------------------------------------------
# SeasonalNaiveModel — specific
# ---------------------------------------------------------------------------
class TestSeasonalNaiveModel:
    def test_repeats_last_season(self):
        """With period=7, should repeat last 7 values."""
        values = list(range(1, 15))  # 14 values, 2 weeks
        df = pd.DataFrame({
            "item_id": ["A"] * 14,
            "timestamp": pd.date_range("2023-01-01", periods=14, freq="D"),
            "target": values,
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = SeasonalNaiveModel(freq="D", prediction_length=7,
                               hyperparameters={"seasonal_period": 7})
        m.fit(tsdf)
        pred = m.predict(tsdf)
        # Last 7 values: 8,9,10,11,12,13,14
        np.testing.assert_array_almost_equal(pred["mean"].values, [8, 9, 10, 11, 12, 13, 14])

    def test_wraps_around(self):
        """prediction_length > seasonal_period should tile the season."""
        values = [10, 20, 30]  # period=3
        df = pd.DataFrame({
            "item_id": ["A"] * 6,
            "timestamp": pd.date_range("2023-01-01", periods=6, freq="D"),
            "target": values * 2,
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = SeasonalNaiveModel(freq="D", prediction_length=7,
                               hyperparameters={"seasonal_period": 3})
        m.fit(tsdf)
        pred = m.predict(tsdf)
        expected = [10, 20, 30, 10, 20, 30, 10]
        np.testing.assert_array_almost_equal(pred["mean"].values, expected)

    def test_custom_seasonal_period(self):
        m = SeasonalNaiveModel(freq="D", prediction_length=5,
                               hyperparameters={"seasonal_period": 12})
        assert m.get_hyperparameter("seasonal_period") == 12


# ---------------------------------------------------------------------------
# AverageModel — specific
# ---------------------------------------------------------------------------
class TestAverageModel:
    def test_predicts_mean(self):
        df = pd.DataFrame({
            "item_id": ["A"] * 5,
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
            "target": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = AverageModel(freq="D", prediction_length=3)
        m.fit(tsdf)
        pred = m.predict(tsdf)
        np.testing.assert_array_almost_equal(pred["mean"].values, [30, 30, 30])

    def test_constant_intervals(self):
        """Average model intervals should NOT expand with horizon."""
        np.random.seed(0)
        df = pd.DataFrame({
            "item_id": ["A"] * 100,
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "target": np.random.randn(100) + 50,
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = AverageModel(freq="D", prediction_length=10)
        m.fit(tsdf)
        pred = m.predict(tsdf, quantile_levels=[0.1, 0.9])
        widths = pred["0.9"].values - pred["0.1"].values
        # All widths should be equal (constant uncertainty)
        np.testing.assert_array_almost_equal(widths, widths[0])


# ---------------------------------------------------------------------------
# SeasonalAverageModel — specific
# ---------------------------------------------------------------------------
class TestSeasonalAverageModel:
    def test_per_season_means(self):
        """With period=3, averages values at positions 0, 1, 2 independently."""
        # pattern: [10, 20, 30] repeated 4 times = 12 values
        values = [10, 20, 30] * 4
        df = pd.DataFrame({
            "item_id": ["A"] * 12,
            "timestamp": pd.date_range("2023-01-01", periods=12, freq="D"),
            "target": values,
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = SeasonalAverageModel(freq="D", prediction_length=6,
                                 hyperparameters={"seasonal_period": 3})
        m.fit(tsdf)
        pred = m.predict(tsdf)
        # 12 % 3 = 0, so forecast starts at season index 0
        expected = [10, 20, 30, 10, 20, 30]
        np.testing.assert_array_almost_equal(pred["mean"].values, expected)


# ---------------------------------------------------------------------------
# DriftModel — specific
# ---------------------------------------------------------------------------
class TestDriftModel:
    def test_linear_extrapolation(self):
        """Perfect linear series => drift model extrapolates exactly."""
        df = pd.DataFrame({
            "item_id": ["A"] * 10,
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "target": [10 + 2 * i for i in range(10)],  # 10, 12, ..., 28
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = DriftModel(freq="D", prediction_length=3)
        m.fit(tsdf)
        pred = m.predict(tsdf)
        # Last = 28, drift = (28-10)/9 = 2.0
        expected = [30.0, 32.0, 34.0]
        np.testing.assert_array_almost_equal(pred["mean"].values, expected, decimal=5)

    def test_flat_series_no_drift(self):
        """Constant series => drift = 0."""
        df = pd.DataFrame({
            "item_id": ["A"] * 10,
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "target": [42.0] * 10,
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = DriftModel(freq="D", prediction_length=5)
        m.fit(tsdf)
        pred = m.predict(tsdf)
        np.testing.assert_array_almost_equal(pred["mean"].values, [42] * 5)

    def test_drift_expanding_intervals(self):
        """Drift intervals should expand with horizon."""
        np.random.seed(42)
        values = np.arange(100, dtype=float) + np.random.randn(100) * 5
        df = pd.DataFrame({
            "item_id": ["A"] * 100,
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "target": values,
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        m = DriftModel(freq="D", prediction_length=10)
        m.fit(tsdf)
        pred = m.predict(tsdf, quantile_levels=[0.1, 0.9])
        widths = pred["0.9"].values - pred["0.1"].values
        assert widths[-1] > widths[0]


# ---------------------------------------------------------------------------
# Multi-item consistency
# ---------------------------------------------------------------------------
class TestMultiItem:
    @pytest.mark.parametrize("ModelClass", ALL_NAIVE)
    def test_each_item_independent(self, ModelClass):
        """Predictions for item A shouldn't change when item B is added."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df_a = pd.DataFrame({
            "item_id": ["A"] * 20,
            "timestamp": dates,
            "target": np.arange(20, dtype=float),
        })
        df_ab = pd.DataFrame({
            "item_id": ["A"] * 20 + ["B"] * 20,
            "timestamp": list(dates) * 2,
            "target": list(np.arange(20, dtype=float)) + list(np.arange(100, 120, dtype=float)),
        })
        tsdf_a = TimeSeriesDataFrame.from_data_frame(df_a)
        tsdf_ab = TimeSeriesDataFrame.from_data_frame(df_ab)

        m1 = ModelClass(freq="D", prediction_length=5)
        m1.fit(tsdf_a)
        pred_a = m1.predict(tsdf_a)

        m2 = ModelClass(freq="D", prediction_length=5)
        m2.fit(tsdf_ab)
        pred_ab = m2.predict(tsdf_ab)

        # Item A predictions should be the same
        np.testing.assert_array_almost_equal(
            pred_a.loc["A"]["mean"].values,
            pred_ab.loc["A"]["mean"].values,
        )
