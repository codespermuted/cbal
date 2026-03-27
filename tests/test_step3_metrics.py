"""Step 3: Metrics verification tests."""

import numpy as np
import pytest

from cbal.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    RMSSE,
    WQL,
    QuantileLoss,
    TimeSeriesScorer,
    get_metric,
    sMAPE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def y_true():
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0])


@pytest.fixture
def y_pred():
    return np.array([12.0, 18.0, 33.0, 37.0, 52.0])


@pytest.fixture
def y_train():
    return np.array([5.0, 8.0, 12.0, 15.0, 10.0, 20.0, 18.0, 25.0, 22.0, 30.0])


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------
class TestMAE:
    def test_perfect(self, y_true):
        assert MAE()(y_true, y_true) == 0.0

    def test_known_value(self, y_true, y_pred):
        # |12-10|+|18-20|+|33-30|+|37-40|+|52-50| = 2+2+3+3+2 = 12 / 5 = 2.4
        assert MAE()(y_true, y_pred) == pytest.approx(2.4)

    def test_symmetric(self, y_true, y_pred):
        assert MAE()(y_true, y_pred) == MAE()(y_pred, y_true)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------
class TestRMSE:
    def test_perfect(self, y_true):
        assert RMSE()(y_true, y_true) == 0.0

    def test_known_value(self, y_true, y_pred):
        # MSE = (4+4+9+9+4)/5 = 30/5 = 6.0; RMSE = sqrt(6) ≈ 2.449
        assert RMSE()(y_true, y_pred) == pytest.approx(np.sqrt(6.0), rel=1e-6)

    def test_ge_mae(self, y_true, y_pred):
        """RMSE >= MAE always."""
        assert RMSE()(y_true, y_pred) >= MAE()(y_true, y_pred)


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------
class TestMAPE:
    def test_perfect(self, y_true):
        assert MAPE()(y_true, y_true) == 0.0

    def test_known_value(self, y_true, y_pred):
        # |(12-10)/10|+|(18-20)/20|+|(33-30)/30|+|(37-40)/40|+|(52-50)/50|
        # = 0.2 + 0.1 + 0.1 + 0.075 + 0.04 = 0.515 => * 100 = 10.3%
        assert MAPE()(y_true, y_pred) == pytest.approx(10.3, rel=1e-3)

    def test_zero_in_y_true(self):
        """Should handle zeros by masking them out."""
        y = np.array([0.0, 10.0, 20.0])
        p = np.array([1.0, 10.0, 20.0])
        score = MAPE()(y, p)
        # Only 2 non-zero entries, both perfect => 0
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------
class TestSMAPE:
    def test_perfect(self, y_true):
        assert sMAPE()(y_true, y_true) == 0.0

    def test_symmetric(self, y_true, y_pred):
        """sMAPE is symmetric: sMAPE(y,p) == sMAPE(p,y)."""
        assert sMAPE()(y_true, y_pred) == pytest.approx(sMAPE()(y_pred, y_true))

    def test_bounded(self, y_true, y_pred):
        """sMAPE is bounded in [0, 200]."""
        score = sMAPE()(y_true, y_pred)
        assert 0 <= score <= 200


# ---------------------------------------------------------------------------
# MASE
# ---------------------------------------------------------------------------
class TestMASE:
    def test_requires_y_train(self, y_true, y_pred):
        with pytest.raises(ValueError, match="y_train"):
            MASE()(y_true, y_pred)

    def test_perfect(self, y_true, y_train):
        assert MASE()(y_true, y_true, y_train=y_train) == 0.0

    def test_known_value(self, y_true, y_pred, y_train):
        # naive scale (period=1): mean(|diff|) of y_train
        diffs = np.abs(np.diff(y_train))
        scale = np.mean(diffs)
        mae = 2.4
        expected = mae / scale
        assert MASE()(y_true, y_pred, y_train=y_train) == pytest.approx(expected, rel=1e-6)

    def test_seasonal_period(self, y_true, y_pred):
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7])
        scorer = MASE(seasonal_period=7)
        score = scorer(y_true, y_pred, y_train=y_train)
        # seasonal naive errors = 0 (perfect repeat) → scale fallback to 1.0
        # (AG convention: prevents inf from dominating averages)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# RMSSE
# ---------------------------------------------------------------------------
class TestRMSSE:
    def test_requires_y_train(self, y_true, y_pred):
        with pytest.raises(ValueError, match="y_train"):
            RMSSE()(y_true, y_pred)

    def test_perfect(self, y_true, y_train):
        assert RMSSE()(y_true, y_true, y_train=y_train) == 0.0

    def test_known_value(self, y_true, y_pred, y_train):
        diffs = np.diff(y_train)
        scale_sq = np.mean(diffs ** 2)
        mse = np.mean((y_true - y_pred) ** 2)
        expected = np.sqrt(mse / scale_sq)
        assert RMSSE()(y_true, y_pred, y_train=y_train) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# WQL (quantile)
# ---------------------------------------------------------------------------
class TestWQL:
    def test_point_forecast_fallback(self, y_true, y_pred):
        """1-D y_pred: WQL falls back to MAE."""
        assert WQL()(y_true, y_pred) == pytest.approx(MAE()(y_true, y_pred))

    def test_requires_quantile_levels_for_2d(self, y_true):
        y_pred_q = np.column_stack([y_true - 1, y_true, y_true + 1])
        with pytest.raises(ValueError, match="quantile_levels"):
            WQL()(y_true, y_pred_q)

    def test_perfect_quantiles(self, y_true):
        """Perfect quantile predictions => 0."""
        y_pred_q = np.column_stack([y_true, y_true, y_true])
        assert WQL()(y_true, y_pred_q, quantile_levels=[0.1, 0.5, 0.9]) == 0.0

    def test_known_quantile(self):
        y_true = np.array([10.0, 20.0])
        # All quantile predictions = 15
        y_pred_q = np.array([[15.0, 15.0], [15.0, 15.0]])
        quantiles = [0.1, 0.9]
        score = WQL()(y_true, y_pred_q, quantile_levels=quantiles)
        assert score > 0


# ---------------------------------------------------------------------------
# QuantileLoss
# ---------------------------------------------------------------------------
class TestQuantileLoss:
    def test_invalid_quantile(self):
        with pytest.raises(ValueError, match="quantile"):
            QuantileLoss(quantile=0.0)
        with pytest.raises(ValueError, match="quantile"):
            QuantileLoss(quantile=1.0)

    def test_median_equals_half_mae(self, y_true, y_pred):
        """Pinball loss at 0.5 = 0.5 * MAE."""
        ql = QuantileLoss(quantile=0.5)
        assert ql(y_true, y_pred) == pytest.approx(0.5 * MAE()(y_true, y_pred))

    def test_name_includes_quantile(self):
        ql = QuantileLoss(quantile=0.9)
        assert "0.9" in ql.name


# ---------------------------------------------------------------------------
# get_metric / registry
# ---------------------------------------------------------------------------
class TestGetMetric:
    def test_by_name(self):
        m = get_metric("RMSE")
        assert isinstance(m, RMSE)

    def test_case_insensitive(self):
        m = get_metric("rmse")
        assert isinstance(m, RMSE)

    def test_pass_through_instance(self):
        original = MAE()
        m = get_metric(original)
        assert m is original

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("NONEXISTENT")

    def test_all_registered(self):
        for name in ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "RMSSE", "WQL"]:
            m = get_metric(name)
            assert isinstance(m, TimeSeriesScorer)


# ---------------------------------------------------------------------------
# Scorer interface
# ---------------------------------------------------------------------------
class TestScorerInterface:
    def test_sign_and_optimum(self):
        """All error metrics: sign=-1, optimum=0."""
        for cls in [MAE, RMSE, MAPE, sMAPE, MASE, RMSSE, WQL]:
            s = cls() if cls not in (MASE, RMSSE) else cls()
            assert s.sign == -1
            assert s.optimum == 0.0

    def test_repr(self):
        r = repr(MAE())
        assert "MAE" in r
        assert "lower_is_better" in r

    def test_accepts_pandas_series(self, y_true, y_pred):
        import pandas as pd
        y_t = pd.Series(y_true)
        y_p = pd.Series(y_pred)
        assert MAE()(y_t, y_p) == pytest.approx(2.4)
