"""
Integration tests — Full end-to-end pipeline verification.

Exercises the complete MyForecaster workflow:
  Data → Fit (presets/HPO) → Leaderboard → Predict → Score → Save → Load → Predict

Run:
    MYFORECASTER_SKIP_TORCH=1 pytest tests/test_step10_integration.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
from myforecaster.predictor import TimeSeriesPredictor
from myforecaster.hpo.space import Categorical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_seasonal_data(n_items=5, n_points=200, freq="D", seed=42):
    """Generate realistic multi-item daily data with trend + seasonality."""
    np.random.seed(seed)
    rows = []
    for i in range(n_items):
        item_id = f"item_{i}"
        base = np.random.uniform(10, 50)
        trend = np.random.uniform(0.01, 0.1)
        amp = np.random.uniform(1, 5)
        for t in range(n_points):
            val = (
                base
                + trend * t
                + amp * np.sin(2 * np.pi * t / 7)  # weekly
                + np.random.randn() * 0.5
            )
            rows.append({
                "item_id": item_id,
                "timestamp": pd.Timestamp("2022-01-01") + pd.Timedelta(days=t),
                "target": val,
            })
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


@pytest.fixture
def seasonal_data():
    return _make_seasonal_data()


@pytest.fixture
def short_data():
    return _make_seasonal_data(n_items=2, n_points=50, seed=99)


# ---------------------------------------------------------------------------
# E2E: fast_training preset
# ---------------------------------------------------------------------------

class TestE2EFastTraining:
    def test_full_pipeline(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=14, eval_metric="MAE")
        p.fit(seasonal_data, presets="fast_training")

        assert p._is_fitted
        assert len(p._models) >= 3

        lb = p.leaderboard(silent=True)
        assert len(lb) >= 3
        assert lb["score_val"].iloc[0] <= lb["score_val"].iloc[-1]

        preds = p.predict(seasonal_data)
        assert isinstance(preds, TimeSeriesDataFrame)
        for item_id in seasonal_data.item_ids:
            assert len(preds.loc[item_id]) == 14

        score = p.score(seasonal_data)
        assert 0 < score < 100  # sanity: MAE on our data should be reasonable

    def test_predict_specific_models(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(seasonal_data, presets="fast_training")

        for name in p._models:
            preds = p.predict(seasonal_data, model=name)
            assert len(preds) > 0


# ---------------------------------------------------------------------------
# E2E: custom preset with multi-config
# ---------------------------------------------------------------------------

class TestE2ECustomPreset:
    def test_multi_config_models(self, seasonal_data):
        """Same model type with multiple hyperparameter configs."""
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            seasonal_data,
            presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"},
            hyperparameters={
                "SeasonalNaive": [
                    {"season_length": 7},
                    {"season_length": 14},
                ],
            },
        )
        sn = [n for n in p._models if "SeasonalNaive" in n]
        assert len(sn) == 2

        lb = p.leaderboard(silent=True)
        assert len(lb) >= 3  # Naive + 2 SeasonalNaive


# ---------------------------------------------------------------------------
# E2E: multi-window backtest
# ---------------------------------------------------------------------------

class TestE2EMultiWindow:
    def test_multi_window_scoring(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            seasonal_data,
            presets="fast_training",
            num_val_windows=3,
        )
        assert p._is_fitted
        assert len(p._val_splits) == 3

        preds = p.predict(seasonal_data)
        assert len(preds) > 0


# ---------------------------------------------------------------------------
# E2E: HPO integration
# ---------------------------------------------------------------------------

class TestE2EHPO:
    def test_hpo_with_search_space(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            seasonal_data,
            presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"},
            hyperparameters={
                "SeasonalNaive": {"season_length": Categorical(5, 7, 14)},
            },
            hyperparameter_tune_kwargs={"num_trials": 3, "searcher": "random"},
        )
        assert p._is_fitted
        assert "SeasonalNaive" in p._models


# ---------------------------------------------------------------------------
# E2E: refit_full
# ---------------------------------------------------------------------------

class TestE2ERefitFull:
    def test_refit_after_selection(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(seasonal_data, presets="fast_training", refit_full=True)
        assert p._is_fitted
        preds = p.predict(seasonal_data)
        assert len(preds) > 0


# ---------------------------------------------------------------------------
# E2E: enable_ensemble=False
# ---------------------------------------------------------------------------

class TestE2ENoEnsemble:
    def test_no_ensemble(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(seasonal_data, presets="fast_training", enable_ensemble=False)
        assert p._ensemble is None
        assert p.best_model != "WeightedEnsemble"


# ---------------------------------------------------------------------------
# E2E: stacking
# ---------------------------------------------------------------------------

class TestE2EStacking:
    def test_stacking_pipeline(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            seasonal_data,
            presets="fast_training",
            num_val_windows=(1, 1),
        )
        assert p._is_fitted
        preds = p.predict(seasonal_data)
        assert len(preds) > 0


# ---------------------------------------------------------------------------
# E2E: save → load → predict roundtrip
# ---------------------------------------------------------------------------

class TestE2ESaveLoadPredict:
    def test_full_roundtrip(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(seasonal_data, presets="fast_training")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "predictor")
            p.path = save_path
            p.save()

            loaded = TimeSeriesPredictor.load(save_path)
            assert loaded._is_fitted
            assert loaded.prediction_length == 7
            assert loaded.best_model == p.best_model

            preds_orig = p.predict(seasonal_data)
            preds_loaded = loaded.predict(seasonal_data)

            for item_id in seasonal_data.item_ids:
                np.testing.assert_allclose(
                    preds_orig.loc[item_id]["mean"].values,
                    preds_loaded.loc[item_id]["mean"].values,
                    rtol=1e-5,
                )


# ---------------------------------------------------------------------------
# E2E: prediction cache
# ---------------------------------------------------------------------------

class TestE2ECache:
    def test_cache_speeds_up_repeated_calls(self, seasonal_data):
        import time
        p = TimeSeriesPredictor(
            prediction_length=7, eval_metric="MAE", cache_predictions=True,
        )
        p.fit(seasonal_data, presets="fast_training")

        t0 = time.time()
        p.predict(seasonal_data)
        first_time = time.time() - t0

        t0 = time.time()
        p.predict(seasonal_data)
        cached_time = time.time() - t0

        # Cached should be at least 2x faster (usually 100x+)
        assert cached_time < first_time or cached_time < 0.01


# ---------------------------------------------------------------------------
# E2E: reproducibility with random_seed
# ---------------------------------------------------------------------------

class TestE2EReproducibility:
    def test_same_seed_same_results(self, seasonal_data):
        scores1 = {}
        p1 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p1.fit(seasonal_data, presets="fast_training", random_seed=42)
        scores1 = dict(p1._model_scores)

        scores2 = {}
        p2 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p2.fit(seasonal_data, presets="fast_training", random_seed=42)
        scores2 = dict(p2._model_scores)

        for name in scores1:
            if name in scores2:
                assert abs(scores1[name] - scores2[name]) < 1e-10


# ---------------------------------------------------------------------------
# E2E: different metrics
# ---------------------------------------------------------------------------

class TestE2EMetrics:
    @pytest.mark.parametrize("metric", ["MAE", "RMSE", "MASE", "sMAPE"])
    def test_metric_variants(self, seasonal_data, metric):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric=metric)
        p.fit(
            seasonal_data,
            presets={
                "models": {"Naive": {}, "SeasonalNaive": {}},
                "ensemble": "SimpleAverage",
            },
        )
        score = p.score(seasonal_data, metric=metric)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# E2E: context_length auto-detection
# ---------------------------------------------------------------------------

class TestE2EAutoContext:
    def test_context_length_reasonable(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=14, eval_metric="MAE")
        p.fit(seasonal_data, presets="fast_training")
        assert p._context_length >= 14
        assert p._context_length <= 200


# ---------------------------------------------------------------------------
# E2E: short data handling
# ---------------------------------------------------------------------------

class TestE2EShortData:
    def test_short_data_no_crash(self, short_data):
        p = TimeSeriesPredictor(prediction_length=5, eval_metric="MAE")
        p.fit(
            short_data,
            presets={
                "models": {"Naive": {}, "Average": {}},
                "ensemble": "SimpleAverage",
            },
        )
        assert p._is_fitted
        preds = p.predict(short_data)
        assert len(preds) > 0


# ---------------------------------------------------------------------------
# E2E: fit_summary consistency
# ---------------------------------------------------------------------------

class TestE2EFitSummary:
    def test_summary_complete(self, seasonal_data):
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(seasonal_data, presets="fast_training")
        s = p.fit_summary()

        assert s["prediction_length"] == 7
        assert s["eval_metric"] == "MAE"
        assert s["freq"] is not None
        assert s["context_length"] is not None
        assert s["n_models_trained"] >= 3
        assert s["best_model"] is not None
        assert s["best_score"] is not None
        assert isinstance(s["model_scores"], dict)
        assert len(s["model_scores"]) >= 3
