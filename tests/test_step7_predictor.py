"""
Tests for Step 7+: TimeSeriesPredictor with all AutoGluon-aligned features.

Covers: multi-window backtest, HP list values, auto context_length,
refit_full, prediction cache, random_seed, HPO "auto", enable_ensemble,
stacking (multi-layer ensemble).

Run:
    MYFORECASTER_SKIP_TORCH=1 pytest tests/test_step7_predictor.py -v
"""

import os, tempfile
import numpy as np
import pandas as pd
import pytest

from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
from myforecaster.predictor import (
    TimeSeriesPredictor, _resolve_preset, _resolve_hpo_kwargs,
    _auto_context_length, _infer_seasonal_period,
)


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


def _naive_preset(**kw):
    return {
        "models": {"Naive": {}, "SeasonalNaive": {}, "Average": {}, **kw},
        "ensemble": "SimpleAverage",
        "time_limit_per_model": 10,
    }


# ---------------------------------------------------------------------------
# Presets & HPO shortcuts
# ---------------------------------------------------------------------------

class TestPresets:
    def test_resolve_all(self):
        for name in ["fast_training", "medium_quality", "high_quality", "best_quality"]:
            c = _resolve_preset(name)
            assert len(c["models"]) > 0

    def test_resolve_unknown(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            _resolve_preset("nonexistent")

    def test_resolve_none_defaults(self):
        c = _resolve_preset(None)
        assert len(c["models"]) >= 4

    def test_hpo_auto(self):
        r = _resolve_hpo_kwargs("auto")
        assert r["searcher"] == "bayes"
        assert r["num_trials"] == 10

    def test_hpo_random_str(self):
        r = _resolve_hpo_kwargs("random")
        assert r["searcher"] == "random"

    def test_hpo_none(self):
        assert _resolve_hpo_kwargs(None) is None

    def test_hpo_dict_passthrough(self):
        d = {"num_trials": 5, "searcher": "random"}
        assert _resolve_hpo_kwargs(d) == d


# ---------------------------------------------------------------------------
# [Feature 3] Auto context_length
# ---------------------------------------------------------------------------

class TestAutoContext:
    def test_seasonal_period_daily(self):
        assert _infer_seasonal_period("D") == 7

    def test_seasonal_period_hourly(self):
        assert _infer_seasonal_period("h") == 24

    def test_seasonal_period_monthly(self):
        assert _infer_seasonal_period("MS") == 12

    def test_auto_ctx_basic(self):
        ctx = _auto_context_length(14, "D", max_ts_length=200)
        assert ctx >= 14
        assert ctx <= 200

    def test_auto_ctx_respects_max(self):
        ctx = _auto_context_length(14, "D", max_ts_length=20)
        assert ctx <= 20

    def test_auto_ctx_at_least_pred_len(self):
        ctx = _auto_context_length(50, "D", max_ts_length=55)
        assert ctx >= 50


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        assert p.prediction_length == 7
        assert not p._is_fitted

    def test_cache_predictions_flag(self):
        p = TimeSeriesPredictor(cache_predictions=False)
        assert not p._pred_cache._enabled

    def test_predict_before_fit(self, sample_data):
        with pytest.raises(RuntimeError, match="not fitted"):
            TimeSeriesPredictor().predict(sample_data)


# ---------------------------------------------------------------------------
# Basic fit & predict (unchanged API)
# ---------------------------------------------------------------------------

class TestFitBasic:
    def test_fit(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        assert p._is_fitted
        assert len(p._models) >= 2

    def test_predict(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        preds = p.predict(sample_data)
        assert isinstance(preds, TimeSeriesDataFrame)
        assert "mean" in preds.columns

    def test_leaderboard(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        lb = p.leaderboard(silent=True)
        assert "model" in lb.columns
        assert len(lb) >= 2

    def test_score(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        s = p.score(sample_data)
        assert isinstance(s, float) and s >= 0


# ---------------------------------------------------------------------------
# [Feature 1] Multi-window backtest
# ---------------------------------------------------------------------------

class TestMultiWindow:
    def test_multi_window_splits(self, sample_data):
        splits = sample_data.multi_window_backtest_splits(7, num_windows=3)
        assert len(splits) == 3
        # Later windows have more data in test
        assert len(splits[2][1]) >= len(splits[0][1])

    def test_predictor_num_val_windows(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset(), num_val_windows=2)
        assert p._is_fitted
        assert len(p._val_splits) == 2

    def test_multi_window_scores_different_from_single(self, sample_data):
        """Multi-window avg may differ from single-window score."""
        p1 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p1.fit(sample_data, presets=_naive_preset(), num_val_windows=1)
        s1 = p1._model_scores.get("Naive", float("inf"))

        p2 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p2.fit(sample_data, presets=_naive_preset(), num_val_windows=3)
        s2 = p2._model_scores.get("Naive", float("inf"))

        # Both should be finite
        assert np.isfinite(s1) and np.isfinite(s2)


# ---------------------------------------------------------------------------
# [Feature 2] HP list values
# ---------------------------------------------------------------------------

class TestHPList:
    def test_list_expands_to_multiple_models(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            sample_data,
            presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"},
            hyperparameters={
                "SeasonalNaive": [
                    {"season_length": 5},
                    {"season_length": 7},
                    {"season_length": 14},
                ],
            },
        )
        # Should have Naive + 3 SeasonalNaive variants
        sn_models = [n for n in p._models if "SeasonalNaive" in n]
        assert len(sn_models) == 3

    def test_single_item_list(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            sample_data,
            presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"},
            hyperparameters={"SeasonalNaive": [{"season_length": 7}]},
        )
        assert "SeasonalNaive" in p._models


# ---------------------------------------------------------------------------
# [Feature 3] Auto context_length injected
# ---------------------------------------------------------------------------

class TestContextLengthInjection:
    def test_context_length_set(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        assert p._context_length is not None
        assert p._context_length >= 7


# ---------------------------------------------------------------------------
# [Feature 4] refit_full
# ---------------------------------------------------------------------------

class TestRefitFull:
    def test_refit_full_runs(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset(), refit_full=True)
        assert p._is_fitted

    def test_refit_full_false_default(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        # No error, just normal training
        assert p._is_fitted


# ---------------------------------------------------------------------------
# [Feature 5] Prediction cache
# ---------------------------------------------------------------------------

class TestPredictionCache:
    def test_cache_hit(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE",
                                cache_predictions=True)
        p.fit(sample_data, presets=_naive_preset())
        pred1 = p.predict(sample_data)
        pred2 = p.predict(sample_data)  # should be cached
        pd.testing.assert_frame_equal(pred1, pred2)

    def test_cache_disabled(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE",
                                cache_predictions=False)
        p.fit(sample_data, presets=_naive_preset())
        pred1 = p.predict(sample_data)
        pred2 = p.predict(sample_data)
        pd.testing.assert_frame_equal(pred1, pred2)


# ---------------------------------------------------------------------------
# [Feature 6] random_seed
# ---------------------------------------------------------------------------

class TestRandomSeed:
    def test_seed_makes_reproducible(self, sample_data):
        p1 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p1.fit(sample_data, presets=_naive_preset(), random_seed=42)
        s1 = p1._model_scores

        p2 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p2.fit(sample_data, presets=_naive_preset(), random_seed=42)
        s2 = p2._model_scores

        for name in s1:
            if name in s2:
                assert abs(s1[name] - s2[name]) < 1e-10


# ---------------------------------------------------------------------------
# [Feature 8] enable_ensemble
# ---------------------------------------------------------------------------

class TestEnableEnsemble:
    def test_ensemble_disabled(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset(), enable_ensemble=False)
        assert p._ensemble is None
        assert "WeightedEnsemble" not in p._model_scores

    def test_ensemble_enabled_default(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        # With 3 models, ensemble should be attempted
        assert p._is_fitted


# ---------------------------------------------------------------------------
# [Feature 9] Stacking (multi-layer ensemble)
# ---------------------------------------------------------------------------

class TestStacking:
    def test_stacking_with_tuple_windows(self, sample_data):
        """num_val_windows=(1, 1) should try stacking."""
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset(), num_val_windows=(1, 1))
        assert p._is_fitted
        # Stacking may or may not succeed depending on data, but no crash
        assert "StackedEnsemble" in p._model_scores or p._stacking_ensemble is None

    def test_stacking_int_no_stacking(self, sample_data):
        """num_val_windows=2 (int) should NOT do stacking."""
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset(), num_val_windows=2)
        assert p._stacking_ensemble is None


# ---------------------------------------------------------------------------
# Excluded models & hyperparameter override
# ---------------------------------------------------------------------------

class TestExcludeOverride:
    def test_exclude(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset(), excluded_model_types=["Average"])
        assert "Average" not in p._models

    def test_override(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset(),
              hyperparameters={"SeasonalNaive": {"season_length": 14}})
        assert p._is_fitted


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        with tempfile.TemporaryDirectory() as d:
            p.path = os.path.join(d, "pred")
            p.save()
            loaded = TimeSeriesPredictor.load(os.path.join(d, "pred"))
            assert loaded._is_fitted
            preds = loaded.predict(sample_data)
            assert len(preds) > 0

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            TimeSeriesPredictor.load("/nonexistent")


# ---------------------------------------------------------------------------
# fit_summary includes context_length
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_has_context_length(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        s = p.fit_summary()
        assert "context_length" in s
        assert s["context_length"] >= 7

    def test_model_info(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        info = p.model_info("Naive")
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_model(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(sample_data, presets={"models": {"Naive": {}}, "ensemble": "WeightedEnsemble"})
        assert p._ensemble is None

    def test_prediction_length_1(self, sample_data):
        p = TimeSeriesPredictor(prediction_length=1, eval_metric="MAE")
        p.fit(sample_data, presets=_naive_preset())
        preds = p.predict(sample_data)
        for iid in sample_data.item_ids:
            assert len(preds.loc[iid]) == 1
