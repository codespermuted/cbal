"""
Tests for data-adaptive "auto" preset and time-budget scheduling.

Run:
    MYFORECASTER_SKIP_TORCH=1 pytest tests/test_auto_preset.py -v
"""

import numpy as np
import pandas as pd
import pytest

from cbal.dataset.ts_dataframe import TimeSeriesDataFrame
from cbal.predictor import (
    TimeSeriesPredictor,
    _build_auto_preset,
    _profile_data,
    _schedule_models_by_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_items=5, n_points=200, freq="D", seed=42, covariates=None):
    np.random.seed(seed)
    rows = []
    for i in range(n_items):
        item_id = f"item_{i}"
        for t in range(n_points):
            row = {
                "item_id": item_id,
                "timestamp": pd.Timestamp("2022-01-01") + pd.Timedelta(days=t),
                "target": 10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 7) + np.random.randn() * 0.3,
            }
            if covariates:
                for cov_name in covariates:
                    row[cov_name] = np.random.randn()
            rows.append(row)
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# _profile_data tests
# ---------------------------------------------------------------------------

class TestProfileData:
    def test_tiny_data(self):
        data = _make_data(n_items=3, n_points=30)
        profile = _profile_data(data, prediction_length=7)
        assert profile["size_category"] == "tiny"
        assert profile["feature_category"] == "univariate"
        assert profile["num_items"] == 3
        assert profile["min_length"] == 30

    def test_small_data(self):
        data = _make_data(n_items=10, n_points=100)
        profile = _profile_data(data, prediction_length=7)
        assert profile["size_category"] == "small"

    def test_medium_data(self):
        data = _make_data(n_items=50, n_points=200)
        profile = _profile_data(data, prediction_length=7)
        assert profile["size_category"] == "medium"

    def test_feature_categories(self):
        # univariate
        data = _make_data(n_items=5, n_points=100)
        p = _profile_data(data, prediction_length=7)
        assert p["feature_category"] == "univariate"

        # low_feature (known covariates)
        p2 = _profile_data(data, prediction_length=7, known_covariates_names=["promo", "holiday"])
        assert p2["feature_category"] == "low_feature"

    def test_rich_feature_with_covariates(self):
        data = _make_data(n_items=5, n_points=100, covariates=["f1", "f2", "f3"])
        data.past_covariates_names = ["f1", "f2", "f3"]
        p = _profile_data(data, prediction_length=7, known_covariates_names=["k1", "k2", "k3"])
        assert p["feature_category"] == "rich_feature"
        assert p["num_features"] == 6


# ---------------------------------------------------------------------------
# _build_auto_preset tests
# ---------------------------------------------------------------------------

class TestBuildAutoPreset:
    def test_tiny_data_stats_only(self):
        profile = {
            "num_items": 3, "min_length": 30, "median_length": 30,
            "max_length": 30, "total_rows": 90,
            "num_known_covariates": 0, "num_past_covariates": 0,
            "has_static_features": False, "num_static_features": 0,
            "num_features": 0, "size_category": "tiny",
            "feature_category": "univariate",
        }
        config = _build_auto_preset(profile, prediction_length=7)
        model_names = set(config["models"].keys())
        # Should only have tier 0-1 (naive + stats)
        assert "Naive" in model_names or "SeasonalNaive" in model_names
        # No DL models for tiny data
        dl_models = {"DLinear", "PatchTST", "DeepAR", "TFT", "N-HiTS"}
        assert not model_names.intersection(dl_models)

    def test_medium_data_includes_dl(self):
        profile = {
            "num_items": 50, "min_length": 200, "median_length": 200,
            "max_length": 200, "total_rows": 10000,
            "num_known_covariates": 0, "num_past_covariates": 0,
            "has_static_features": False, "num_static_features": 0,
            "num_features": 0, "size_category": "medium",
            "feature_category": "univariate",
        }
        config = _build_auto_preset(profile, prediction_length=7)
        model_names = set(config["models"].keys())
        # Medium univariate → up to tier 3 (lightweight DL)
        assert "DLinear" in model_names or "PatchTST" in model_names

    def test_rich_features_includes_covariate_aware(self):
        profile = {
            "num_items": 50, "min_length": 200, "median_length": 200,
            "max_length": 200, "total_rows": 10000,
            "num_known_covariates": 3, "num_past_covariates": 3,
            "has_static_features": True, "num_static_features": 2,
            "num_features": 8, "size_category": "medium",
            "feature_category": "rich_feature",
        }
        config = _build_auto_preset(profile, prediction_length=7)
        model_names = set(config["models"].keys())
        # Rich features → TFT and DeepAR should be included
        assert "TFT" in model_names or "DeepAR" in model_names

    def test_ensemble_type(self):
        profile = {
            "num_items": 50, "min_length": 200, "median_length": 200,
            "max_length": 200, "total_rows": 10000,
            "num_known_covariates": 0, "num_past_covariates": 0,
            "has_static_features": False, "num_static_features": 0,
            "num_features": 0, "size_category": "medium",
            "feature_category": "univariate",
        }
        config = _build_auto_preset(profile, prediction_length=7)
        assert config["ensemble"] in ("WeightedEnsemble", "SimpleAverage")


# ---------------------------------------------------------------------------
# _schedule_models_by_budget tests
# ---------------------------------------------------------------------------

class TestScheduleByBudget:
    def _medium_profile(self):
        return {
            "num_items": 50, "min_length": 200, "median_length": 200,
            "max_length": 200, "total_rows": 10000,
            "num_known_covariates": 0, "num_past_covariates": 0,
            "has_static_features": False, "num_static_features": 0,
            "num_features": 0, "size_category": "medium",
            "feature_category": "univariate",
        }

    def test_generous_budget_keeps_all(self):
        config = {
            "models": {
                "Naive": {},
                "SeasonalNaive": {},
                "AutoETS": {},
            },
            "ensemble": "SimpleAverage",
            "time_limit_per_model": 60,
        }
        profile = self._medium_profile()
        result = _schedule_models_by_budget(config, time_limit=600, profile=profile)
        assert len(result["models"]) == 3

    def test_tight_budget_drops_expensive(self):
        config = {
            "models": {
                "Naive": {},
                "SeasonalNaive": {},
                "AutoETS": {},
                "DeepAR": {"max_epochs": 100},
                "PatchTST": {"max_epochs": 100},
                "TFT": {"max_epochs": 100},
                "iTransformer": {"max_epochs": 100},
                "TimeMixer": {"max_epochs": 100},
            },
            "ensemble": "WeightedEnsemble",
            "time_limit_per_model": 300,
        }
        profile = self._medium_profile()
        # Very tight budget — should drop some expensive models
        result = _schedule_models_by_budget(config, time_limit=30, profile=profile)
        assert len(result["models"]) < 8
        # Baseline models should survive (highest priority)
        assert "Naive" in result["models"] or "SeasonalNaive" in result["models"]

    def test_per_model_limit_set(self):
        config = {
            "models": {"Naive": {}, "AutoETS": {}},
            "ensemble": "SimpleAverage",
            "time_limit_per_model": 60,
        }
        result = _schedule_models_by_budget(
            config, time_limit=120, profile=self._medium_profile()
        )
        assert result["time_limit_per_model"] > 0


# ---------------------------------------------------------------------------
# End-to-end: "auto" preset with real predictor
# ---------------------------------------------------------------------------

class TestAutoPresetE2E:
    def test_auto_preset_tiny(self):
        """Auto preset on tiny data: fits stats-only, produces predictions."""
        data = _make_data(n_items=3, n_points=60)
        predictor = TimeSeriesPredictor(prediction_length=7, eval_metric="MASE")
        predictor.fit(data, presets="auto")
        assert predictor._is_fitted
        preds = predictor.predict(data)
        assert len(preds) > 0

    def test_auto_preset_small(self):
        """Auto preset on small data: includes tabular models."""
        data = _make_data(n_items=10, n_points=100)
        predictor = TimeSeriesPredictor(prediction_length=7, eval_metric="MASE")
        predictor.fit(data, presets="auto")
        assert predictor._is_fitted

    def test_time_limit_respected(self):
        """With time_limit, training finishes within budget."""
        import time
        data = _make_data(n_items=5, n_points=100)
        predictor = TimeSeriesPredictor(prediction_length=7, eval_metric="MASE")
        t0 = time.time()
        predictor.fit(data, presets="medium_quality", time_limit=30)
        elapsed = time.time() - t0
        assert predictor._is_fitted
        # Should finish well within 2x budget (generous for test overhead)
        assert elapsed < 60

    def test_auto_with_time_limit(self):
        """Combine auto preset + time_limit."""
        data = _make_data(n_items=5, n_points=80)
        predictor = TimeSeriesPredictor(prediction_length=7, eval_metric="MASE")
        predictor.fit(data, presets="auto", time_limit=60)
        assert predictor._is_fitted
        preds = predictor.predict(data)
        assert len(preds) > 0

    def test_auto_preset_in_resolve(self):
        """Verify _resolve_preset('auto') returns marker dict."""
        from cbal.predictor import _resolve_preset
        config = _resolve_preset("auto")
        assert config.get("_auto") is True
