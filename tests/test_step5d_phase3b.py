"""Step 5-d Phase 3b: TimeMixer, TimesNet, ModernTCN verification tests.

Run on your server:
    cd cbal-project
    pytest tests/test_step5d_phase3b.py -v
"""

import numpy as np
import pandas as pd
import pytest

from cbal.dataset import TimeSeriesDataFrame

import os, importlib.util
if os.environ.get("MYFORECASTER_SKIP_TORCH", ""):
    pytest.skip("MYFORECASTER_SKIP_TORCH is set", allow_module_level=True)
if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)

import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_tsdf():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    rows = []
    for item_id in ["A", "B", "C"]:
        base = {"A": 100, "B": 200, "C": 50}[item_id]
        for i, d in enumerate(dates):
            val = base + i * 0.2 + 10 * np.sin(2 * np.pi * i / 7) + np.random.randn() * 2
            rows.append({"item_id": item_id, "timestamp": d, "target": val})
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


@pytest.fixture
def pred_length():
    return 7


@pytest.fixture
def train_test(daily_tsdf, pred_length):
    return daily_tsdf.train_test_split(pred_length)


# ===========================================================================
# TimeMixer tests
# ===========================================================================
class TestTimeMixerNetwork:
    def test_output_shape(self):
        from cbal.models.deep_learning.timemixer import TimeMixerNetwork
        net = TimeMixerNetwork(context_length=96, prediction_length=24,
                                d_model=32, n_scales=3, n_layers=2)
        x = torch.randn(4, 96)
        out = net(x)
        assert out.shape == (4, 24)

    def test_multiscale_lengths(self):
        from cbal.models.deep_learning.timemixer import TimeMixerNetwork
        net = TimeMixerNetwork(context_length=96, prediction_length=24, n_scales=4)
        assert net.scale_lengths == [96, 48, 24, 12]

    def test_gradient_flows(self):
        from cbal.models.deep_learning.timemixer import TimeMixerNetwork
        net = TimeMixerNetwork(context_length=64, prediction_length=12,
                                d_model=16, n_scales=3, n_layers=1)
        x = torch.randn(2, 64)
        loss = net(x).mean()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_pdm_decomposition(self):
        """PDM should decompose and mix at multiple scales."""
        from cbal.models.deep_learning.timemixer import PDMBlock
        pdm = PDMBlock(n_channels=1, d_model=16, n_scales=3,
                        scale_lengths=[64, 32, 16])
        x_scales = [torch.randn(2, 64, 16), torch.randn(2, 32, 16), torch.randn(2, 16, 16)]
        out = pdm(x_scales)
        assert len(out) == 3
        assert out[0].shape == (2, 64, 16)
        assert out[2].shape == (2, 16, 16)


class TestTimeMixerModel:
    def test_fit_predict(self, train_test, pred_length):
        from cbal.models.deep_learning.timemixer import TimeMixerModel
        train, _ = train_test
        m = TimeMixerModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 16, "n_scales": 3, "n_layers": 1,
                "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from cbal.models.deep_learning.timemixer import TimeMixerModel
        train, test = train_test
        m = TimeMixerModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 16, "n_scales": 3, "n_layers": 1,
            },
        )
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from cbal.models.deep_learning.timemixer import TimeMixerModel
        from cbal.models import MODEL_REGISTRY
        assert "TimeMixer" in MODEL_REGISTRY


# ===========================================================================
# TimesNet tests
# ===========================================================================
class TestTimesNetNetwork:
    def test_output_shape(self):
        from cbal.models.deep_learning.timesnet import TimesNetNetwork
        net = TimesNetNetwork(context_length=96, prediction_length=24,
                               d_model=32, d_ff=32, n_layers=1, top_k=3)
        x = torch.randn(4, 96)
        out = net(x)
        assert out.shape == (4, 24)

    def test_fft_period_discovery(self):
        """TimesBlock should handle periodic input."""
        from cbal.models.deep_learning.timesnet import TimesBlock
        block = TimesBlock(seq_len=96, d_model=16, d_ff=16, top_k=3)
        x = torch.randn(2, 96, 16)
        out = block(x)
        assert out.shape == (2, 96, 16)
        assert torch.isfinite(out).all()

    def test_inception_block(self):
        from cbal.models.deep_learning.timesnet import InceptionBlock
        inc = InceptionBlock(d_model=16, d_ff=8, num_kernels=3)
        x = torch.randn(2, 16, 8, 12)  # (B, D, num_periods, period_len)
        out = inc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 16

    def test_gradient_flows(self):
        from cbal.models.deep_learning.timesnet import TimesNetNetwork
        net = TimesNetNetwork(context_length=64, prediction_length=12,
                               d_model=16, d_ff=16, n_layers=1, top_k=2)
        x = torch.randn(2, 64)
        loss = net(x).mean()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0


class TestTimesNetModel:
    def test_fit_predict(self, train_test, pred_length):
        from cbal.models.deep_learning.timesnet import TimesNetModel
        train, _ = train_test
        m = TimesNetModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 16, "d_ff": 16, "n_layers": 1,
                "top_k": 3, "num_kernels": 3, "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from cbal.models.deep_learning.timesnet import TimesNetModel
        train, test = train_test
        m = TimesNetModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 16, "d_ff": 16, "n_layers": 1, "top_k": 3,
            },
        )
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from cbal.models.deep_learning.timesnet import TimesNetModel
        from cbal.models import MODEL_REGISTRY
        assert "TimesNet" in MODEL_REGISTRY


# ===========================================================================
# ModernTCN tests
# ===========================================================================
class TestModernTCNNetwork:
    def test_output_shape(self):
        from cbal.models.deep_learning.moderntcn import ModernTCNNetwork
        net = ModernTCNNetwork(context_length=96, prediction_length=24,
                                d_model=32, d_ff=64, n_layers=2,
                                kernel_size=13, patch_len=16, stride=8)
        x = torch.randn(4, 96)
        out = net(x)
        assert out.shape == (4, 24)

    def test_dwconv_is_depthwise(self):
        """DWConv should have groups=d_model (depthwise)."""
        from cbal.models.deep_learning.moderntcn import ModernTCNBlock
        block = ModernTCNBlock(d_model=32, kernel_size=7)
        assert block.dwconv.groups == 32

    def test_large_kernel(self):
        """ModernTCN should support large kernels."""
        from cbal.models.deep_learning.moderntcn import ModernTCNBlock
        block = ModernTCNBlock(d_model=32, kernel_size=51)
        x = torch.randn(2, 32, 100)
        out = block(x)
        assert out.shape == (2, 32, 100)

    def test_gradient_flows(self):
        from cbal.models.deep_learning.moderntcn import ModernTCNNetwork
        net = ModernTCNNetwork(context_length=64, prediction_length=12,
                                d_model=16, d_ff=32, n_layers=1,
                                patch_len=8, stride=4)
        x = torch.randn(2, 64)
        loss = net(x).mean()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0


class TestModernTCNModel:
    def test_fit_predict(self, train_test, pred_length):
        from cbal.models.deep_learning.moderntcn import ModernTCNModel
        train, _ = train_test
        m = ModernTCNModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 32, "d_ff": 64, "n_layers": 2,
                "kernel_size": 13, "patch_len": 8, "stride": 4,
                "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from cbal.models.deep_learning.moderntcn import ModernTCNModel
        train, test = train_test
        m = ModernTCNModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 32, "d_ff": 64, "n_layers": 2,
                "patch_len": 8, "stride": 4,
            },
        )
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from cbal.models.deep_learning.moderntcn import ModernTCNModel
        from cbal.models import MODEL_REGISTRY
        assert "ModernTCN" in MODEL_REGISTRY
