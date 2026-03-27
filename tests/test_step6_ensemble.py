"""Step 6: Ensemble model verification tests.

Run on your server:
    cd cbal-project
    pytest tests/test_step6_ensemble.py -v
"""

import numpy as np
import pandas as pd
import pytest

from cbal.dataset import TimeSeriesDataFrame
from cbal.models.ensemble import (
    WeightedEnsemble,
    SimpleAverageEnsemble,
    greedy_ensemble_selection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_tsdf():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rows = []
    for item_id in ["A", "B", "C"]:
        base = {"A": 100, "B": 200, "C": 50}[item_id]
        for i, d in enumerate(dates):
            val = base + i * 0.3 + 10 * np.sin(2 * np.pi * i / 7) + np.random.randn() * 3
            rows.append({"item_id": item_id, "timestamp": d, "target": val})
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


@pytest.fixture
def pred_length():
    return 7


@pytest.fixture
def train_val_test(daily_tsdf, pred_length):
    train_val, test = daily_tsdf.train_test_split(pred_length)
    train, val = train_val.train_test_split(pred_length)
    return train, val, test


@pytest.fixture
def fitted_base_models(train_val_test, pred_length):
    train, val, test = train_val_test
    from cbal.models.naive.models import NaiveModel, AverageModel, SeasonalNaiveModel

    models = {}
    for ModelClass in [NaiveModel, AverageModel, SeasonalNaiveModel]:
        m = ModelClass(freq="D", prediction_length=pred_length)
        m.fit(train)
        models[m.name] = m
    return models


# ===========================================================================
# Registration
# ===========================================================================
class TestRegistration:
    def test_weighted_ensemble_registered(self):
        from cbal.models import MODEL_REGISTRY
        from cbal.models.ensemble import WeightedEnsemble  # noqa
        assert "WeightedEnsemble" in MODEL_REGISTRY

    def test_simple_average_registered(self):
        from cbal.models import MODEL_REGISTRY
        from cbal.models.ensemble import SimpleAverageEnsemble  # noqa
        assert "SimpleAverage" in MODEL_REGISTRY


# ===========================================================================
# SimpleAverage
# ===========================================================================
class TestSimpleAverage:
    def test_fit_predict(self, train_val_test, fitted_base_models, pred_length):
        train, val, test = train_val_test
        ens = SimpleAverageEnsemble(freq="D", prediction_length=pred_length)
        ens.fit(train, base_models=fitted_base_models)
        pred = ens.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert np.isfinite(pred["mean"].values).all()

    def test_average_is_between_extremes(self, train_val_test, fitted_base_models, pred_length):
        train, _, _ = train_val_test
        ens = SimpleAverageEnsemble(freq="D", prediction_length=pred_length)
        ens.fit(train, base_models=fitted_base_models)
        ens_pred = ens.predict(train)

        individual = []
        for name, model in fitted_base_models.items():
            individual.append(model.predict(train)["mean"].values)

        ens_mean = ens_pred["mean"].values
        assert (ens_mean >= np.min(individual, axis=0) - 0.01).all()
        assert (ens_mean <= np.max(individual, axis=0) + 0.01).all()

    def test_requires_base_models(self, train_val_test, pred_length):
        train, _, _ = train_val_test
        ens = SimpleAverageEnsemble(freq="D", prediction_length=pred_length)
        with pytest.raises(ValueError, match="base_models"):
            ens.fit(train)

    def test_score(self, train_val_test, fitted_base_models, pred_length):
        train, _, test = train_val_test
        ens = SimpleAverageEnsemble(freq="D", prediction_length=pred_length)
        ens.fit(train, base_models=fitted_base_models)
        assert np.isfinite(ens.score(test, metric="MAE"))


# ===========================================================================
# WeightedEnsemble
# ===========================================================================
class TestWeightedEnsemble:
    def test_fit_predict(self, train_val_test, fitted_base_models, pred_length):
        train, val, _ = train_val_test
        ens = WeightedEnsemble(
            freq="D", prediction_length=pred_length,
            hyperparameters={"ensemble_size": 10, "metric": "MAE"},
        )
        ens.fit(train, val_data=val, base_models=fitted_base_models)
        assert ens._is_fitted
        pred = ens.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert np.isfinite(pred["mean"].values).all()

    def test_weights_sum_to_one(self, train_val_test, fitted_base_models, pred_length):
        train, val, _ = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length,
                               hyperparameters={"ensemble_size": 10})
        ens.fit(train, val_data=val, base_models=fitted_base_models)
        assert abs(sum(ens.weights.values()) - 1.0) < 1e-6

    def test_selected_models_subset(self, train_val_test, fitted_base_models, pred_length):
        train, val, _ = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length,
                               hyperparameters={"ensemble_size": 5})
        ens.fit(train, val_data=val, base_models=fitted_base_models)
        for name in ens.selected_models:
            assert name in fitted_base_models

    def test_ensemble_score_finite(self, train_val_test, fitted_base_models, pred_length):
        train, val, _ = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length,
                               hyperparameters={"ensemble_size": 10})
        ens.fit(train, val_data=val, base_models=fitted_base_models)
        assert np.isfinite(ens.ensemble_score)

    def test_score_on_test(self, train_val_test, fitted_base_models, pred_length):
        train, val, test = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length,
                               hyperparameters={"ensemble_size": 10})
        ens.fit(train, val_data=val, base_models=fitted_base_models)
        assert np.isfinite(ens.score(test, metric="MAE"))

    def test_requires_val_data(self, train_val_test, fitted_base_models, pred_length):
        train, _, _ = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length)
        with pytest.raises(ValueError, match="val_data"):
            ens.fit(train, base_models=fitted_base_models)

    def test_requires_base_models(self, train_val_test, pred_length):
        train, val, _ = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length)
        with pytest.raises(ValueError, match="base_models"):
            ens.fit(train, val_data=val)

    def test_model_info(self, train_val_test, fitted_base_models, pred_length):
        train, val, _ = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length,
                               hyperparameters={"ensemble_size": 5})
        ens.fit(train, val_data=val, base_models=fitted_base_models)
        info = ens.model_info()
        assert "weights" in info
        assert "ensemble_score" in info
        assert info["n_base_models"] == 3

    def test_repr(self, train_val_test, fitted_base_models, pred_length):
        train, val, _ = train_val_test
        ens = WeightedEnsemble(freq="D", prediction_length=pred_length,
                               hyperparameters={"ensemble_size": 5})
        ens.fit(train, val_data=val, base_models=fitted_base_models)
        assert "WeightedEnsemble" in repr(ens)


# ===========================================================================
# Greedy Selection
# ===========================================================================
class TestGreedySelection:
    def test_single_model(self, train_val_test, fitted_base_models, pred_length):
        _, val, _ = train_val_test
        from cbal.metrics.scorers import get_metric
        from cbal.models.ensemble import _compute_per_item_predictions

        single = {k: v for k, v in list(fitted_base_models.items())[:1]}
        name = list(single.keys())[0]
        preds = {name: _compute_per_item_predictions(single[name], val)}

        weights, score = greedy_ensemble_selection(
            preds, val, pred_length, get_metric("MAE"), ensemble_size=5
        )
        assert name in weights
        assert abs(weights[name] - 1.0) < 1e-6

    def test_weights_sum_to_one(self, train_val_test, fitted_base_models, pred_length):
        _, val, _ = train_val_test
        from cbal.metrics.scorers import get_metric
        from cbal.models.ensemble import _compute_per_item_predictions

        preds = {}
        for name, model in fitted_base_models.items():
            preds[name] = _compute_per_item_predictions(model, val)

        weights, score = greedy_ensemble_selection(
            preds, val, pred_length, get_metric("MAE"), ensemble_size=15
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert np.isfinite(score)
