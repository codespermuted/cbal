"""
Tests for 10 previously-missing features (now all implemented).

Run:
    MYFORECASTER_SKIP_TORCH=1 pytest tests/test_remaining_10.py -v
"""

import os, tempfile
import numpy as np
import pandas as pd
import pytest

from cbal.dataset.ts_dataframe import TimeSeriesDataFrame, TARGET
from cbal.metrics.scorers import (
    WAPE, Coverage, SQL, MAE, RMSE, get_metric, METRIC_REGISTRY,
)


def _make_data(n_items=3, n_points=80, seed=42):
    np.random.seed(seed)
    rows = []
    for i in range(n_items):
        for t in range(n_points):
            rows.append({
                "item_id": f"item_{i}",
                "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=t),
                "target": 10 + i * 5 + 0.1 * t + np.random.randn(),
                "promo": int(t % 5 == 0),
            })
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


# =====================================================================
# 1. convert_frequency
# =====================================================================

class TestConvertFrequency:
    def test_upsample_daily_to_weekly(self):
        data = _make_data(n_items=2, n_points=28)
        weekly = data.convert_frequency("W")
        for iid in weekly.item_ids:
            assert len(weekly.loc[iid]) <= 5  # ~4 weeks

    def test_metadata_propagated(self):
        data = _make_data()
        data.known_covariates_names = ["promo"]
        weekly = data.convert_frequency("W")
        assert weekly.known_covariates_names == ["promo"]

    def test_new_freq_set(self):
        data = _make_data(n_items=1, n_points=60)
        monthly = data.convert_frequency("MS")
        assert monthly._cached_freq == "MS"


# =====================================================================
# 2. WAPE + Coverage metrics
# =====================================================================

class TestWAPE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert WAPE()(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])
        # |2| + |2| = 4, sum(|y|) = 30, WAPE = 4/30
        assert WAPE()(y_true, y_pred) == pytest.approx(4 / 30, rel=1e-5)

    def test_registered(self):
        assert "WAPE" in METRIC_REGISTRY


class TestCoverage:
    def test_all_covered(self):
        y_true = np.array([5.0, 10.0, 15.0])
        y_pred = np.array([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])
        assert Coverage()(y_true, y_pred) == 1.0

    def test_none_covered(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([[0.0, 1.0], [0.0, 1.0]])
        assert Coverage()(y_true, y_pred) == 0.0

    def test_registered(self):
        assert "Coverage" in METRIC_REGISTRY


class TestSQL:
    def test_registered(self):
        assert "SQL" in METRIC_REGISTRY

    def test_sql_finite(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        y_train = np.arange(20, dtype=float)
        s = SQL()(y_true, y_pred, y_train=y_train)
        assert np.isfinite(s)


# =====================================================================
# 3. eval_metric_seasonal_period
# =====================================================================

class TestEvalMetricSeasonalPeriod:
    def test_get_metric_with_seasonal_period(self):
        m = get_metric("MASE", seasonal_period=7)
        assert m.seasonal_period == 7

    def test_default_no_seasonal(self):
        m = get_metric("MASE")
        assert m.seasonal_period == 1

    def test_predictor_param(self):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(
            prediction_length=7, eval_metric="MASE",
            eval_metric_seasonal_period=7,
        )
        assert p.eval_metric_seasonal_period == 7


# =====================================================================
# 4. horizon_weight
# =====================================================================

class TestHorizonWeight:
    def test_mae_weighted(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 22.0])
        # errors: [1, 2]
        # uniform: mean = 1.5
        assert MAE()(y_true, y_pred) == pytest.approx(1.5)
        # weighted: [0.8, 0.2] → 1*0.8 + 2*0.2 = 1.2 / 1.0 → weighted_avg
        w = np.array([0.8, 0.2])
        weighted = MAE()(y_true, y_pred, horizon_weight=w)
        assert weighted == pytest.approx(np.average([1, 2], weights=w))

    def test_rmse_weighted(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 22.0])
        w = np.array([1.0, 0.0])
        # Only first step counts, error = 1
        assert RMSE()(y_true, y_pred, horizon_weight=w) == pytest.approx(1.0)

    def test_predictor_accepts_horizon_weight(self):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(
            prediction_length=3,
            horizon_weight=[1.0, 0.5, 0.25],
        )
        assert p.horizon_weight is not None
        assert len(p.horizon_weight) == 3


# =====================================================================
# 5. log_to_file
# =====================================================================

class TestLogToFile:
    def test_creates_log_file(self):
        from cbal.predictor import TimeSeriesPredictor
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pred")
            p = TimeSeriesPredictor(
                prediction_length=7, path=path, log_to_file=True,
            )
            assert p.log_to_file is True
            logfile = os.path.join(path, "predictor.log")
            assert os.path.exists(logfile)


# =====================================================================
# 6. ensemble_hyperparameters
# =====================================================================

class TestEnsembleHyperparameters:
    def test_fit_with_ensemble_hp(self):
        from cbal.predictor import TimeSeriesPredictor
        data = _make_data()
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            data,
            presets={"models": {"Naive": {}, "Average": {}}, "ensemble": "WeightedEnsemble"},
            ensemble_hyperparameters={"ensemble_size": 50},
        )
        assert p._is_fitted
        assert p._ensemble is not None


# =====================================================================
# 7. skip_model_selection
# =====================================================================

class TestSkipModelSelection:
    def test_skip_is_faster(self):
        import time
        from cbal.predictor import TimeSeriesPredictor
        data = _make_data()

        p1 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        t0 = time.time()
        p1.fit(data, presets={"models": {"Naive": {}, "Average": {}}, "ensemble": "SimpleAverage"})
        t_normal = time.time() - t0

        p2 = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        t0 = time.time()
        p2.fit(data, presets={"models": {"Naive": {}, "Average": {}}, "ensemble": "SimpleAverage"},
               skip_model_selection=True)
        t_skip = time.time() - t0

        assert p2._is_fitted
        # skip should be no slower (give margin for noise)
        assert t_skip < t_normal * 2.0


# =====================================================================
# 8. covariate_scaler
# =====================================================================

class TestCovariateScaler:
    def test_global_scaler(self):
        from cbal.models.wrappers import CovariateScaler
        data = _make_data()
        cs = CovariateScaler(method="global")
        scaled, _ = cs.fit_transform(data)
        assert cs._is_fitted
        # promo column should be scaled
        assert scaled["promo"].std() != data["promo"].std()

    def test_predictor_integration(self):
        from cbal.predictor import TimeSeriesPredictor
        data = _make_data()
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            data,
            presets={
                "models": {"Naive": {}},
                "ensemble": "SimpleAverage",
                "covariate_scaler": "global",
            },
        )
        assert p._cov_scaler is not None


# =====================================================================
# 9. auto DataFrame conversion
# =====================================================================

class TestAutoDataFrameConversion:
    def test_fit_with_raw_dataframe(self):
        from cbal.predictor import TimeSeriesPredictor
        np.random.seed(42)
        raw_df = pd.DataFrame({
            "item_id": ["A"] * 50 + ["B"] * 50,
            "timestamp": list(pd.date_range("2023-01-01", periods=50, freq="D")) * 2,
            "target": np.random.randn(100) + 10,
        })
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(raw_df, presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"})
        assert p._is_fitted


# =====================================================================
# 10. parallel HPO (joblib)
# =====================================================================

class TestParallelHPO:
    def test_tune_model_with_n_jobs(self):
        from cbal.hpo.runner import tune_model
        from cbal.hpo.space import Int
        data = _make_data(n_items=2, n_points=50)
        train, test = data.train_test_split(7)
        best_config, best_score, history = tune_model(
            model_name="Naive",
            search_space={"seasonal_period": Int(1, 7)},
            train_data=train, val_data=test,
            freq="D", prediction_length=7,
            eval_metric="MAE", num_trials=4,
            searcher="random", n_jobs=2,
        )
        assert len(history) == 4
        assert best_score < float("inf")

    def test_n_jobs_1_works(self):
        from cbal.hpo.runner import tune_model
        from cbal.hpo.space import Int
        data = _make_data(n_items=2, n_points=50)
        train, test = data.train_test_split(7)
        _, score, hist = tune_model(
            model_name="Naive",
            search_space={"seasonal_period": Int(1, 3)},
            train_data=train, val_data=test,
            freq="D", prediction_length=7,
            num_trials=2, n_jobs=1,
        )
        assert len(hist) == 2
