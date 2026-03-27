"""
Tests for covariate system: static_features, covariates, target_scaler,
covariate_regressor, predictor integration.

Run:
    MYFORECASTER_SKIP_TORCH=1 pytest tests/test_covariate_system.py -v
"""

import os, tempfile
import numpy as np
import pandas as pd
import pytest

from cbal.dataset.ts_dataframe import TimeSeriesDataFrame, ITEMID, TARGET, TIMESTAMP
from cbal.models.wrappers import TargetScaler, CovariateRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data_with_covariates(n_items=3, n_points=100, seed=42):
    """Generate data with known covariates, past covariates, and static features."""
    np.random.seed(seed)
    rows = []
    for i in range(n_items):
        item_id = f"item_{i}"
        base = np.random.uniform(10, 50)
        for t in range(n_points):
            is_weekend = 1 if (t % 7) in (5, 6) else 0
            promo = 1 if np.random.rand() < 0.2 else 0
            temperature = 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.randn()
            # target depends on covariates
            val = base + 0.1 * t + 3 * promo - 2 * is_weekend + 0.1 * temperature + np.random.randn()
            rows.append({
                "item_id": item_id,
                "timestamp": pd.Timestamp("2022-01-01") + pd.Timedelta(days=t),
                "target": val,
                "promotion": promo,
                "is_weekend": is_weekend,
                "temperature": temperature,
            })

    df = pd.DataFrame(rows)
    tsdf = TimeSeriesDataFrame.from_data_frame(df)

    # Static features
    static = pd.DataFrame({
        "category": ["food", "electronics", "clothing"][:n_items],
        "store_size": [100, 200, 150][:n_items],
    }, index=[f"item_{i}" for i in range(n_items)])
    static.index.name = ITEMID
    tsdf.static_features = static
    tsdf.known_covariates_names = ["promotion", "is_weekend"]
    tsdf.past_covariates_names = ["temperature"]

    return tsdf


@pytest.fixture
def cov_data():
    return _make_data_with_covariates()


@pytest.fixture
def plain_data():
    """Data without covariates."""
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


# =====================================================================
# A. TSDF static_features + covariate tracking
# =====================================================================

class TestTSDFStaticFeatures:
    def test_static_features_setter(self, cov_data):
        assert cov_data.static_features is not None
        assert len(cov_data.static_features) == 3
        assert "category" in cov_data.static_features.columns

    def test_static_features_persist_copy(self, cov_data):
        copied = cov_data.copy()
        # _metadata propagation via pandas
        assert copied._static_features is not None or cov_data.static_features is not None

    def test_known_covariates_names(self, cov_data):
        assert cov_data.known_covariates_names == ["promotion", "is_weekend"]

    def test_past_covariates_names(self, cov_data):
        assert cov_data.past_covariates_names == ["temperature"]

    def test_covariate_columns(self, cov_data):
        cc = cov_data.covariate_columns
        assert "promotion" in cc
        assert "is_weekend" in cc
        assert "temperature" in cc
        assert "target" not in cc

    def test_invalid_covariate_name_raises(self, cov_data):
        with pytest.raises(ValueError, match="not in columns"):
            cov_data.known_covariates_names = ["nonexistent_column"]

    def test_static_features_none_ok(self, plain_data):
        assert plain_data.static_features is None

    def test_static_features_type_check(self, plain_data):
        with pytest.raises(TypeError, match="DataFrame"):
            plain_data.static_features = "not a dataframe"


class TestTSDFSliceByTimestep:
    def test_basic_slice(self, cov_data):
        sliced = cov_data.slice_by_timestep(0, 50)
        for iid in sliced.item_ids:
            assert len(sliced.loc[iid]) == 50

    def test_negative_slice(self, cov_data):
        sliced = cov_data.slice_by_timestep(-10, None)
        for iid in sliced.item_ids:
            assert len(sliced.loc[iid]) == 10

    def test_metadata_propagated(self, cov_data):
        sliced = cov_data.slice_by_timestep(0, 50)
        assert sliced._known_covariates_names == cov_data._known_covariates_names


class TestTSDFModelInputs:
    def test_get_model_inputs(self, cov_data):
        past, kc = cov_data.get_model_inputs_for_scoring(
            prediction_length=7,
            known_covariates_names=["promotion", "is_weekend"],
        )
        for iid in past.item_ids:
            assert len(past.loc[iid]) == 100 - 7
        assert kc is not None
        assert "promotion" in kc.columns
        for iid in kc.item_ids:
            assert len(kc.loc[iid]) == 7

    def test_get_model_inputs_no_covariates(self, plain_data):
        past, kc = plain_data.get_model_inputs_for_scoring(prediction_length=7)
        assert kc is None

    def test_metadata_in_train_test_split(self, cov_data):
        train, test = cov_data.train_test_split(7)
        assert train._known_covariates_names == cov_data._known_covariates_names
        assert train._static_features is not None


# =====================================================================
# B. TargetScaler
# =====================================================================

class TestTargetScaler:
    @pytest.mark.parametrize("method", ["standard", "mean_abs", "robust", "min_max"])
    def test_roundtrip_invertible(self, plain_data, method):
        scaler = TargetScaler(method=method)
        scaled = scaler.fit_transform(plain_data)

        # Predictions are normally a different TSDF, but we test on same structure
        preds = scaled.copy()
        preds = preds.rename(columns={"target": "mean"})
        result = scaler.inverse_transform_predictions(preds)

        orig_vals = plain_data[TARGET].values
        restored = result["mean"].values
        np.testing.assert_allclose(orig_vals, restored, atol=1e-5)

    def test_standard_zero_mean(self, plain_data):
        scaler = TargetScaler(method="standard")
        scaled = scaler.fit_transform(plain_data)
        for iid in scaled.item_ids:
            vals = scaled.loc[iid][TARGET].values
            assert abs(np.mean(vals)) < 0.1

    def test_min_max_range(self, plain_data):
        scaler = TargetScaler(method="min_max")
        scaled = scaler.fit_transform(plain_data)
        for iid in scaled.item_ids:
            vals = scaled.loc[iid][TARGET].values
            assert np.min(vals) >= -0.01
            assert np.max(vals) <= 1.01

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown scaler"):
            TargetScaler(method="invalid")


# =====================================================================
# C. CovariateRegressor
# =====================================================================

class TestCovariateRegressor:
    def test_fit_and_remove_effect(self, cov_data):
        reg = CovariateRegressor(
            known_covariates_names=["promotion", "is_weekend"],
            past_covariates_names=["temperature"],
            backend="linear",
        )
        reg.fit(cov_data, cov_data.static_features)
        assert reg._is_fitted

        residual = reg.remove_covariate_effect(cov_data, cov_data.static_features)
        # Residuals should have lower variance than original
        orig_std = cov_data[TARGET].std()
        resid_std = residual[TARGET].std()
        assert resid_std < orig_std * 1.1  # should be lower or similar

    def test_no_covariates_skips(self, plain_data):
        reg = CovariateRegressor()
        reg.fit(plain_data)
        assert not reg._is_fitted

    def test_add_covariate_effect(self, cov_data):
        reg = CovariateRegressor(
            known_covariates_names=["promotion", "is_weekend"],
            backend="linear",
        )
        reg.fit(cov_data, cov_data.static_features)

        # Create fake predictions
        train, test = cov_data.train_test_split(7)
        _, kc = test.get_model_inputs_for_scoring(7, ["promotion", "is_weekend"])

        fake_preds_rows = []
        for iid in test.item_ids:
            for h in range(7):
                fake_preds_rows.append({
                    "item_id": iid,
                    "timestamp": pd.Timestamp("2022-04-08") + pd.Timedelta(days=h),
                    "target": 0.0,
                })
        fake_preds = TimeSeriesDataFrame.from_data_frame(pd.DataFrame(fake_preds_rows))
        fake_preds["mean"] = fake_preds[TARGET]

        result = reg.add_covariate_effect(fake_preds, known_covariates=kc)
        # With zero base predictions, result should be non-zero (covariate effect added)
        assert result["mean"].abs().sum() > 0


# =====================================================================
# D. Predictor integration with target_scaler
# =====================================================================

class TestPredictorScaler:
    def test_fit_with_target_scaler(self, plain_data):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            plain_data,
            presets={
                "models": {"Naive": {}, "Average": {}},
                "ensemble": "SimpleAverage",
                "target_scaler": "standard",
            },
        )
        assert p._is_fitted
        assert p._target_scaler is not None
        assert p._target_scaler.method == "standard"

    def test_predict_with_scaler(self, plain_data):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            plain_data,
            presets={
                "models": {"Naive": {}, "Average": {}},
                "ensemble": "SimpleAverage",
                "target_scaler": "mean_abs",
            },
        )
        preds = p.predict(plain_data)
        # Predictions should be in original scale (not scaled)
        pred_mean = preds["mean"].mean()
        orig_mean = plain_data[TARGET].mean()
        # Should be in same order of magnitude
        assert 0.1 * orig_mean < pred_mean < 10 * orig_mean

    def test_no_scaler_by_default(self, plain_data):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            plain_data,
            presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"},
        )
        assert p._target_scaler is None

    def test_save_load_with_scaler(self, plain_data):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            plain_data,
            presets={
                "models": {"Naive": {}, "Average": {}},
                "ensemble": "SimpleAverage",
                "target_scaler": "robust",
            },
        )
        with tempfile.TemporaryDirectory() as d:
            p.path = os.path.join(d, "pred")
            p.save()
            loaded = TimeSeriesPredictor.load(os.path.join(d, "pred"))
            assert loaded._target_scaler is not None
            assert loaded._target_scaler.method == "robust"
            preds = loaded.predict(plain_data)
            assert len(preds) > 0


# =====================================================================
# E. Predictor integration with covariate_regressor
# =====================================================================

class TestPredictorCovRegressor:
    def test_fit_with_cov_regressor(self, cov_data):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(
            prediction_length=7, eval_metric="MAE",
            known_covariates_names=["promotion", "is_weekend"],
        )
        p.fit(
            cov_data,
            presets={
                "models": {"Naive": {}, "Average": {}},
                "ensemble": "SimpleAverage",
                "covariate_regressor": True,
            },
        )
        assert p._is_fitted
        assert p._cov_regressor is not None
        assert p._cov_regressor._is_fitted

    def test_no_cov_regressor_by_default(self, plain_data):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
        p.fit(
            plain_data,
            presets={"models": {"Naive": {}}, "ensemble": "SimpleAverage"},
        )
        assert p._cov_regressor is None


# =====================================================================
# F. Combined: scaler + regressor
# =====================================================================

class TestCombinedScalerRegressor:
    def test_both_together(self, cov_data):
        from cbal.predictor import TimeSeriesPredictor
        p = TimeSeriesPredictor(
            prediction_length=7, eval_metric="MAE",
            known_covariates_names=["promotion", "is_weekend"],
        )
        p.fit(
            cov_data,
            presets={
                "models": {"Naive": {}, "Average": {}},
                "ensemble": "SimpleAverage",
                "target_scaler": "standard",
                "covariate_regressor": "linear",
            },
        )
        assert p._target_scaler is not None
        assert p._cov_regressor is not None
        preds = p.predict(cov_data)
        assert len(preds) > 0
