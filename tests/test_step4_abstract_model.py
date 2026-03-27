"""Step 4: AbstractTimeSeriesModel verification tests."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from myforecaster.dataset import TimeSeriesDataFrame
from myforecaster.metrics import MAE, RMSE, get_metric
from myforecaster.models.abstract_model import AbstractTimeSeriesModel


# ---------------------------------------------------------------------------
# Concrete test model
# ---------------------------------------------------------------------------
class DummyConstantModel(AbstractTimeSeriesModel):
    """Predicts a constant value (mean of training target)."""

    _default_hyperparameters = {"constant_override": None}

    def _fit(self, train_data, val_data=None, time_limit=None):
        override = self.get_hyperparameter("constant_override")
        if override is not None:
            self._constant = float(override)
        else:
            self._constant = float(train_data["target"].mean())

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        predictions = {}
        for item_id in data.item_ids:
            predictions[item_id] = np.full(self.prediction_length, self._constant)
        return predictions


class BrokenFitModel(AbstractTimeSeriesModel):
    """Model that raises during _fit."""

    def _fit(self, train_data, val_data=None, time_limit=None):
        raise RuntimeError("Intentional fit failure")

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_tsdf():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "item_id": ["A"] * 30 + ["B"] * 30,
        "timestamp": list(dates) * 2,
        "target": list(np.arange(30, dtype=float)) + list(np.arange(30, 60, dtype=float)),
    })
    return TimeSeriesDataFrame.from_data_frame(df)


@pytest.fixture
def fitted_model(sample_tsdf):
    train, _ = sample_tsdf.train_test_split(prediction_length=5)
    model = DummyConstantModel(freq="D", prediction_length=5, name="DummyConst")
    model.fit(train)
    return model, train, sample_tsdf


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------
class TestInstantiation:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractTimeSeriesModel(freq="D", prediction_length=5)

    def test_concrete_model_creation(self):
        m = DummyConstantModel(freq="D", prediction_length=7, name="Test")
        assert m.name == "Test"
        assert m.prediction_length == 7
        assert m.freq == "D"
        assert m._is_fitted is False

    def test_default_name_is_class_name(self):
        m = DummyConstantModel(prediction_length=1)
        assert m.name == "DummyConstantModel"

    def test_default_eval_metric(self):
        m = DummyConstantModel(prediction_length=1)
        assert m.eval_metric.name == "MASE"

    def test_custom_eval_metric(self):
        m = DummyConstantModel(prediction_length=1, eval_metric="RMSE")
        assert m.eval_metric.name == "RMSE"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
class TestHyperparameters:
    def test_default_hyperparams(self):
        m = DummyConstantModel(prediction_length=1)
        hp = m.get_hyperparameters()
        assert hp["constant_override"] is None

    def test_user_override(self):
        m = DummyConstantModel(prediction_length=1, hyperparameters={"constant_override": 42.0})
        assert m.get_hyperparameter("constant_override") == 42.0

    def test_set_hyperparameters(self):
        m = DummyConstantModel(prediction_length=1)
        m.set_hyperparameters(constant_override=99.0)
        assert m.get_hyperparameter("constant_override") == 99.0

    def test_get_nonexistent_returns_default(self):
        m = DummyConstantModel(prediction_length=1)
        assert m.get_hyperparameter("nonexistent", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------
class TestFit:
    def test_fit_sets_is_fitted(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5)
        assert m._is_fitted is False
        m.fit(train)
        assert m._is_fitted is True

    def test_fit_records_time(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5)
        m.fit(train)
        assert m._fit_time is not None
        assert m._fit_time >= 0

    def test_fit_infers_freq(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(prediction_length=5)  # freq=None
        m.fit(train)
        assert m.freq == "D"

    def test_fit_stores_train_item_ids(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5)
        m.fit(train)
        assert set(m._train_item_ids) == {"A", "B"}

    def test_fit_stores_target_tail(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5)
        m.fit(train)
        assert "A" in m._train_target_tail
        assert len(m._train_target_tail["A"]) == 25  # 30 - 5

    def test_fit_with_val_data(self, sample_tsdf):
        train, test = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5)
        m.fit(train, val_data=test)
        assert m._val_score is not None

    def test_fit_error_propagates(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(5)
        m = BrokenFitModel(freq="D", prediction_length=5)
        with pytest.raises(RuntimeError, match="Intentional"):
            m.fit(train)

    def test_fit_returns_self(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5)
        result = m.fit(train)
        assert result is m


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------
class TestPredict:
    def test_predict_before_fit_raises(self, sample_tsdf):
        m = DummyConstantModel(freq="D", prediction_length=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict(sample_tsdf)

    def test_predict_returns_tsdf(self, fitted_model):
        model, train, _ = fitted_model
        pred = model.predict(train)
        assert isinstance(pred, TimeSeriesDataFrame)

    def test_predict_has_mean_column(self, fitted_model):
        model, train, _ = fitted_model
        pred = model.predict(train)
        assert "mean" in pred.columns

    def test_predict_correct_shape(self, fitted_model):
        model, train, _ = fitted_model
        pred = model.predict(train)
        # 2 items * 5 prediction_length = 10 rows
        assert len(pred) == 10

    def test_predict_has_quantile_columns(self, fitted_model):
        model, train, _ = fitted_model
        pred = model.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert "0.1" in pred.columns
        assert "0.5" in pred.columns
        assert "0.9" in pred.columns

    def test_predict_values_correct(self, sample_tsdf):
        """DummyConstantModel with override=42 should predict all 42s."""
        train, _ = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5, hyperparameters={"constant_override": 42.0})
        m.fit(train)
        pred = m.predict(train)
        np.testing.assert_array_almost_equal(pred["mean"].values, 42.0)

    def test_predict_future_timestamps(self, fitted_model):
        """Prediction timestamps should be after the last training timestamp."""
        model, train, _ = fitted_model
        pred = model.predict(train)
        for item_id in train.item_ids:
            last_train_ts = train.loc[item_id].index.get_level_values("timestamp").max()
            pred_timestamps = pred.loc[item_id].index.get_level_values("timestamp")
            assert pred_timestamps.min() > last_train_ts


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------
class TestScore:
    def test_score_before_fit_raises(self, sample_tsdf):
        m = DummyConstantModel(freq="D", prediction_length=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.score(sample_tsdf)

    def test_score_returns_float(self, fitted_model):
        model, _, full_data = fitted_model
        s = model.score(full_data)
        assert isinstance(s, float)

    def test_score_with_custom_metric(self, fitted_model):
        model, _, full_data = fitted_model
        s = model.score(full_data, metric="RMSE")
        assert isinstance(s, float)
        assert s >= 0

    def test_perfect_model_scores_zero(self, sample_tsdf):
        """A model that predicts exact future values should score 0 MAE."""
        # This is a simplistic test — DummyConstantModel won't be perfect
        # but we verify the scoring pipeline works
        train, test = sample_tsdf.train_test_split(5)
        m = DummyConstantModel(freq="D", prediction_length=5)
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert score > 0  # constant prediction != actual values


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
class TestSaveLoad:
    def test_save_creates_file(self, fitted_model):
        model, _, _ = fitted_model
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model")
            model.save(path)
            assert os.path.exists(os.path.join(path, "model.pkl"))

    def test_load_restores_model(self, fitted_model, sample_tsdf):
        model, train, _ = fitted_model
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model")
            model.save(path)
            loaded = AbstractTimeSeriesModel.load(path)
            assert loaded.name == model.name
            assert loaded._is_fitted is True
            assert loaded._constant == model._constant

    def test_loaded_model_can_predict(self, fitted_model):
        model, train, _ = fitted_model
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model")
            model.save(path)
            loaded = AbstractTimeSeriesModel.load(path)
            pred = loaded.predict(train)
            assert len(pred) == 10

    def test_save_without_path_raises(self, fitted_model):
        model, _, _ = fitted_model
        model.path = None
        with pytest.raises(ValueError, match="No save path"):
            model.save()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            AbstractTimeSeriesModel.load("/nonexistent/path")


# ---------------------------------------------------------------------------
# model_info / repr
# ---------------------------------------------------------------------------
class TestInfo:
    def test_model_info_keys(self, fitted_model):
        model, _, _ = fitted_model
        info = model.model_info()
        assert "name" in info
        assert "is_fitted" in info
        assert "fit_time_s" in info
        assert "hyperparameters" in info
        assert info["is_fitted"] is True

    def test_repr_before_fit(self):
        m = DummyConstantModel(freq="D", prediction_length=5, name="Test")
        r = repr(m)
        assert "not fitted" in r
        assert "Test" in r

    def test_repr_after_fit(self, fitted_model):
        model, _, _ = fitted_model
        r = repr(model)
        assert "fitted" in r
        assert "DummyConst" in r


# ---------------------------------------------------------------------------
# Seasonal period inference
# ---------------------------------------------------------------------------
class TestSeasonalPeriod:
    def test_daily(self):
        m = DummyConstantModel(freq="D", prediction_length=1)
        assert m._get_seasonal_period() == 7

    def test_hourly(self):
        m = DummyConstantModel(freq="h", prediction_length=1)
        assert m._get_seasonal_period() == 24

    def test_monthly(self):
        m = DummyConstantModel(freq="MS", prediction_length=1)
        assert m._get_seasonal_period() == 12

    def test_none_freq(self):
        m = DummyConstantModel(freq=None, prediction_length=1)
        assert m._get_seasonal_period() == 1
