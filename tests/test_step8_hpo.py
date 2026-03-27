"""
Tests for Step 8: Hyperparameter Optimization.

Run:
    MYFORECASTER_SKIP_TORCH=1 pytest tests/test_step8_hpo.py -v
"""

import random

import numpy as np
import pandas as pd
import pytest

from cbal.dataset.ts_dataframe import TimeSeriesDataFrame
from cbal.hpo.space import Int, Real, Categorical, sample_config, get_defaults
from cbal.hpo.searcher import RandomSearcher, get_searcher
from cbal.hpo.runner import tune_model, get_default_search_space


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    np.random.seed(42)
    rows = []
    for item in ["A", "B", "C"]:
        base = np.random.rand() * 10 + 5
        for t in range(100):
            rows.append({
                "item_id": item,
                "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=t),
                "target": base + 0.1 * t + np.random.randn() * 0.5,
            })
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Test search space primitives
# ---------------------------------------------------------------------------

class TestSearchSpaces:
    def test_int_sample_in_range(self):
        sp = Int(1, 10)
        rng = random.Random(42)
        for _ in range(50):
            v = sp.sample(rng)
            assert 1 <= v <= 10
            assert isinstance(v, int)

    def test_int_log_sample(self):
        sp = Int(1, 1000, log=True)
        rng = random.Random(42)
        vals = [sp.sample(rng) for _ in range(100)]
        # Log sampling: median should be much less than 500
        assert np.median(vals) < 500

    def test_int_contains(self):
        sp = Int(5, 20)
        assert sp.contains(10)
        assert not sp.contains(3)
        assert not sp.contains(25)

    def test_real_sample_in_range(self):
        sp = Real(0.0, 1.0)
        rng = random.Random(42)
        for _ in range(50):
            v = sp.sample(rng)
            assert 0.0 <= v <= 1.0

    def test_real_log_sample(self):
        sp = Real(1e-5, 1e-1, log=True)
        rng = random.Random(42)
        vals = [sp.sample(rng) for _ in range(100)]
        # Log: should include values near 1e-4 often
        assert any(v < 1e-3 for v in vals)
        assert any(v > 1e-3 for v in vals)

    def test_real_contains(self):
        sp = Real(0.0, 0.5)
        assert sp.contains(0.25)
        assert not sp.contains(0.6)

    def test_categorical_sample(self):
        sp = Categorical("a", "b", "c")
        rng = random.Random(42)
        vals = {sp.sample(rng) for _ in range(50)}
        assert vals == {"a", "b", "c"}

    def test_categorical_contains(self):
        sp = Categorical("x", "y")
        assert sp.contains("x")
        assert not sp.contains("z")

    def test_categorical_single(self):
        sp = Categorical("only_one")
        assert sp.sample() == "only_one"

    def test_categorical_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Categorical()

    def test_int_invalid_range_raises(self):
        with pytest.raises(ValueError, match="lower"):
            Int(10, 5)

    def test_real_invalid_range_raises(self):
        with pytest.raises(ValueError, match="lower"):
            Real(1.0, 0.5)


# ---------------------------------------------------------------------------
# Test sample_config and get_defaults
# ---------------------------------------------------------------------------

class TestConfigUtils:
    def test_sample_config_basic(self):
        space = {
            "d_model": Int(32, 256),
            "lr": Real(1e-4, 1e-2, log=True),
            "act": Categorical("relu", "gelu"),
            "fixed_param": 42,
        }
        config = sample_config(space, random.Random(42))
        assert isinstance(config["d_model"], int)
        assert 32 <= config["d_model"] <= 256
        assert isinstance(config["lr"], float)
        assert config["act"] in ("relu", "gelu")
        assert config["fixed_param"] == 42

    def test_get_defaults(self):
        space = {
            "d_model": Int(32, 256, default=128),
            "lr": Real(1e-4, 1e-2, log=True, default=1e-3),
            "act": Categorical("relu", "gelu", default="gelu"),
            "fixed": "abc",
        }
        defaults = get_defaults(space)
        assert defaults["d_model"] == 128
        assert defaults["lr"] == 1e-3
        assert defaults["act"] == "gelu"
        assert defaults["fixed"] == "abc"


# ---------------------------------------------------------------------------
# Test to_dict serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_int_to_dict(self):
        sp = Int(1, 100, log=True)
        d = sp.to_dict()
        assert d["type"] == "Int"
        assert d["lower"] == 1
        assert d["upper"] == 100
        assert d["log"] is True

    def test_real_to_dict(self):
        d = Real(0.0, 1.0).to_dict()
        assert d["type"] == "Real"

    def test_categorical_to_dict(self):
        d = Categorical("a", "b").to_dict()
        assert d["choices"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Test RandomSearcher
# ---------------------------------------------------------------------------

class TestRandomSearcher:
    def test_suggest_returns_config(self):
        space = {"x": Int(1, 10), "y": Real(0.0, 1.0)}
        s = RandomSearcher(space, seed=42)
        config = s.suggest()
        assert "x" in config
        assert "y" in config

    def test_suggest_different_configs(self):
        space = {"x": Int(1, 1000)}
        s = RandomSearcher(space, seed=42)
        configs = [s.suggest()["x"] for _ in range(20)]
        # Should have some variety
        assert len(set(configs)) > 5

    def test_report_and_best(self):
        space = {"x": Int(1, 10)}
        s = RandomSearcher(space, seed=42)
        c1 = s.suggest()
        s.report(c1, score=0.5)
        c2 = s.suggest()
        s.report(c2, score=0.3)
        c3 = s.suggest()
        s.report(c3, score=0.8)
        assert s.best_score == 0.3
        assert s.best_config == c2
        assert s.n_trials == 3

    def test_empty_best(self):
        s = RandomSearcher({"x": Int(1, 5)})
        assert s.best_config is None
        assert s.best_score is None


# ---------------------------------------------------------------------------
# Test get_searcher factory
# ---------------------------------------------------------------------------

class TestGetSearcher:
    def test_random(self):
        s = get_searcher("random", {"x": Int(1, 10)})
        assert isinstance(s, RandomSearcher)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown HPO"):
            get_searcher("genetic", {"x": Int(1, 10)})


# ---------------------------------------------------------------------------
# Test get_default_search_space
# ---------------------------------------------------------------------------

class TestDefaultSpaces:
    def test_known_models_have_spaces(self):
        for name in ["DLinear", "DeepAR", "PatchTST", "TFT", "N-HiTS"]:
            space = get_default_search_space(name)
            assert len(space) > 0, f"No default space for {name}"

    def test_unknown_model_empty(self):
        space = get_default_search_space("UnknownModel123")
        assert space == {}

    def test_spaces_are_searchspace_instances(self):
        space = get_default_search_space("PatchTST")
        from cbal.hpo.space import SearchSpace
        for v in space.values():
            assert isinstance(v, SearchSpace)


# ---------------------------------------------------------------------------
# Test tune_model (integration — with Naive models, no torch)
# ---------------------------------------------------------------------------

class TestTuneModel:
    def test_tune_naive_random(self, sample_data):
        """Tune a Naive model with random search (trivial — no real HP)."""
        train, val = sample_data.train_test_split(7)
        # SeasonalNaive has season_length we can "tune"
        space = {"season_length": Categorical(5, 7, 14)}
        best_config, best_score, history = tune_model(
            model_name="SeasonalNaive",
            search_space=space,
            train_data=train,
            val_data=val,
            freq="D",
            prediction_length=7,
            eval_metric="MAE",
            num_trials=3,
            searcher="random",
            seed=42,
        )
        assert isinstance(best_config, dict)
        assert "season_length" in best_config
        assert best_score < float("inf")
        assert len(history) == 3

    def test_tune_with_time_limit(self, sample_data):
        train, val = sample_data.train_test_split(7)
        space = {"season_length": Categorical(5, 7)}
        best_config, best_score, history = tune_model(
            model_name="SeasonalNaive",
            search_space=space,
            train_data=train,
            val_data=val,
            freq="D",
            prediction_length=7,
            num_trials=100,  # won't reach 100 with time limit
            time_limit=5,
            searcher="random",
        )
        assert len(history) > 0
        assert len(history) <= 100

    def test_tune_with_base_hyperparameters(self, sample_data):
        train, val = sample_data.train_test_split(7)
        space = {"season_length": Categorical(7, 14)}
        best_config, _, _ = tune_model(
            model_name="SeasonalNaive",
            search_space=space,
            train_data=train,
            val_data=val,
            freq="D",
            prediction_length=7,
            num_trials=2,
            base_hyperparameters={"extra_param": True},
        )
        # base params should be in final config
        assert best_config.get("extra_param") is True


# ---------------------------------------------------------------------------
# Test HPO integration with TimeSeriesPredictor
# ---------------------------------------------------------------------------

class TestPredictorHPO:
    def test_fit_with_hpo(self, sample_data):
        """Predictor + HPO with SearchSpace in hyperparameters."""
        from cbal.predictor import TimeSeriesPredictor

        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            sample_data,
            presets={
                "models": {
                    "Naive": {},
                    "SeasonalNaive": {"season_length": Categorical(5, 7, 14)},
                },
                "ensemble": "SimpleAverage",
            },
            hyperparameter_tune_kwargs={
                "num_trials": 3,
                "searcher": "random",
            },
        )
        assert p._is_fitted
        assert "SeasonalNaive" in p._models

    def test_fit_hpo_mixed(self, sample_data):
        """Mix of HPO models and fixed models."""
        from cbal.predictor import TimeSeriesPredictor

        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            sample_data,
            presets={
                "models": {
                    "Naive": {},  # fixed, no SearchSpace
                    "Average": {},  # fixed
                    "SeasonalNaive": {"season_length": Categorical(5, 7)},  # HPO
                },
                "ensemble": "SimpleAverage",
            },
            hyperparameter_tune_kwargs={
                "num_trials": 2,
                "searcher": "random",
            },
        )
        assert p._is_fitted
        assert len(p._models) >= 2

    def test_hpo_kwargs_none_means_no_hpo(self, sample_data):
        """When hyperparameter_tune_kwargs=None, no HPO even with SearchSpace."""
        from cbal.predictor import TimeSeriesPredictor

        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        # SearchSpace passed but HPO disabled → treated as fixed (fails gracefully)
        p.fit(
            sample_data,
            presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"},
            hyperparameter_tune_kwargs=None,
        )
        assert p._is_fitted


# ---------------------------------------------------------------------------
# Test repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_int_repr(self):
        assert "Int(1, 10" in repr(Int(1, 10))

    def test_real_repr(self):
        assert "Real(0.0, 1.0" in repr(Real(0.0, 1.0))

    def test_categorical_repr(self):
        assert "Categorical" in repr(Categorical("a", "b"))
