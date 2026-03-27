"""Step 5-d Phase 4: Foundation model wrapper verification tests.

These tests verify the wrapper structure WITHOUT loading actual models
(which require large downloads and GPU). Heavy-weight integration tests
should be run separately on GPU servers.

Run on your server:
    cd cbal-project
    pytest tests/test_step5d_phase4_foundation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from cbal.dataset import TimeSeriesDataFrame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_tsdf():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rows = []
    for item_id in ["A", "B"]:
        for i, d in enumerate(dates):
            val = 50 + i * 0.1 + 5 * np.sin(2 * np.pi * i / 7) + np.random.randn()
            rows.append({"item_id": item_id, "timestamp": d, "target": val})
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


@pytest.fixture
def pred_length():
    return 7


@pytest.fixture
def train_test(daily_tsdf, pred_length):
    return daily_tsdf.train_test_split(pred_length)


# ===========================================================================
# Registration tests (always run — no heavy deps needed)
# ===========================================================================
class TestRegistration:
    def test_foundation_models_importable(self):
        """Foundation module should be importable without heavy deps."""
        # This import triggers register_model calls
        from cbal.models import foundation  # noqa: F401

    def test_chronos2_registered(self):
        from cbal.models import foundation  # noqa: F401
        from cbal.models import MODEL_REGISTRY
        assert "Chronos-2" in MODEL_REGISTRY

    def test_timesfm_registered(self):
        from cbal.models import foundation  # noqa: F401
        from cbal.models import MODEL_REGISTRY
        assert "TimesFM" in MODEL_REGISTRY

    def test_moirai_registered(self):
        from cbal.models import foundation  # noqa: F401
        from cbal.models import MODEL_REGISTRY
        assert "Moirai" in MODEL_REGISTRY

    def test_ttm_registered(self):
        from cbal.models import foundation  # noqa: F401
        from cbal.models import MODEL_REGISTRY
        assert "TTM" in MODEL_REGISTRY

    def test_toto_registered(self):
        from cbal.models import foundation  # noqa: F401
        from cbal.models import MODEL_REGISTRY
        assert "Toto" in MODEL_REGISTRY


# ===========================================================================
# Structure tests (verify wrapper API without loading models)
# ===========================================================================
class TestWrapperStructure:
    def test_chronos2_instantiation(self, pred_length):
        from cbal.models.foundation import Chronos2Model
        m = Chronos2Model(freq="D", prediction_length=pred_length)
        assert m.prediction_length == pred_length
        assert not m._is_fitted

    def test_timesfm_instantiation(self, pred_length):
        from cbal.models.foundation import TimesFMModel
        m = TimesFMModel(freq="D", prediction_length=pred_length)
        assert m.prediction_length == pred_length

    def test_moirai_instantiation(self, pred_length):
        from cbal.models.foundation import MoiraiModel
        m = MoiraiModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"model_size": "small", "model_version": "moirai2"},
        )
        assert m.get_hyperparameter("model_size") == "small"

    def test_ttm_instantiation(self, pred_length):
        from cbal.models.foundation import TTMModel
        m = TTMModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"context_length": 512},
        )
        assert m.get_hyperparameter("context_length") == 512

    def test_toto_instantiation(self, pred_length):
        from cbal.models.foundation import TotoModel
        m = TotoModel(freq="D", prediction_length=pred_length)
        assert m.prediction_length == pred_length


class TestFitIsNoOp:
    """Foundation models' fit() should just store metadata, not train."""

    def test_chronos2_fit(self, train_test, pred_length):
        from cbal.models.foundation import Chronos2Model
        train, _ = train_test
        m = Chronos2Model(freq="D", prediction_length=pred_length)
        m.fit(train)
        assert m._is_fitted
        assert m.freq == "D"

    def test_timesfm_fit(self, train_test, pred_length):
        from cbal.models.foundation import TimesFMModel
        train, _ = train_test
        m = TimesFMModel(freq="D", prediction_length=pred_length)
        m.fit(train)
        assert m._is_fitted


class TestDeviceDetection:
    def test_device_auto(self, pred_length):
        from cbal.models.foundation import Chronos2Model
        m = Chronos2Model(freq="D", prediction_length=pred_length)
        device = m._get_device()
        assert device in ("cuda", "cpu")

    def test_device_override(self, pred_length):
        from cbal.models.foundation import Chronos2Model
        m = Chronos2Model(
            freq="D", prediction_length=pred_length,
            hyperparameters={"device": "cpu"},
        )
        assert m._get_device() == "cpu"


class TestDefaultHyperparameters:
    def test_chronos2_defaults(self, pred_length):
        from cbal.models.foundation import Chronos2Model
        m = Chronos2Model(freq="D", prediction_length=pred_length)
        assert m.get_hyperparameter("model_id") == "amazon/chronos-2"
        assert m.get_hyperparameter("batch_size") == 32

    def test_timesfm_defaults(self, pred_length):
        from cbal.models.foundation import TimesFMModel
        m = TimesFMModel(freq="D", prediction_length=pred_length)
        assert "timesfm" in m.get_hyperparameter("model_id")

    def test_moirai_defaults(self, pred_length):
        from cbal.models.foundation import MoiraiModel
        m = MoiraiModel(freq="D", prediction_length=pred_length)
        assert m.get_hyperparameter("model_version") == "moirai2"

    def test_ttm_defaults(self, pred_length):
        from cbal.models.foundation import TTMModel
        m = TTMModel(freq="D", prediction_length=pred_length)
        assert "granite" in m.get_hyperparameter("model_id")


# ===========================================================================
# Heavy integration tests (skip if deps not installed)
# ===========================================================================
class TestChronos2Integration:
    """Run only if chronos-forecasting is installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_chronos(self):
        pytest.importorskip("chronos", reason="chronos-forecasting not installed")

    def test_predict(self, train_test, pred_length):
        from cbal.models.foundation import Chronos2Model
        train, _ = train_test
        m = Chronos2Model(
            freq="D", prediction_length=pred_length,
            hyperparameters={"device": "cpu"},
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert len(pred) == 2 * pred_length  # 2 items
        assert "mean" in pred.columns
        assert np.isfinite(pred["mean"].values).all()


class TestTimesFMIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_timesfm(self):
        pytest.importorskip("timesfm", reason="timesfm not installed")

    def test_predict(self, train_test, pred_length):
        from cbal.models.foundation import TimesFMModel
        train, _ = train_test
        m = TimesFMModel(freq="D", prediction_length=pred_length)
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert len(pred) == 2 * pred_length
        assert np.isfinite(pred["mean"].values).all()


class TestTTMIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_tsfm(self):
        pytest.importorskip("tsfm_public", reason="tsfm_public not installed")

    def test_predict(self, train_test, pred_length):
        from cbal.models.foundation import TTMModel
        train, _ = train_test
        m = TTMModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"context_length": 512},
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 2 * pred_length
        assert np.isfinite(pred["mean"].values).all()
