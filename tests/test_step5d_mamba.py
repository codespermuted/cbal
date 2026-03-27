"""Step 5-d Phase 2.5: SSM/Mamba models (S-Mamba + MambaTS) verification tests.

Run on your server:
    cd myforecaster-project
    pytest tests/test_step5d_mamba.py -v
"""

import numpy as np
import pandas as pd
import pytest

from myforecaster.dataset import TimeSeriesDataFrame

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


# ---------------------------------------------------------------------------
# Core Mamba block tests
# ---------------------------------------------------------------------------
class TestSelectiveSSM:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.layers.mamba import SelectiveSSM
        ssm = SelectiveSSM(d_inner=64, d_state=16)
        x = torch.randn(4, 20, 64)
        out = ssm(x)
        assert out.shape == (4, 20, 64)

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.layers.mamba import SelectiveSSM
        ssm = SelectiveSSM(d_inner=32, d_state=8)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out = ssm(x)
        out.sum().backward()
        assert x.grad is not None


class TestMambaBlock:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.layers.mamba import MambaBlock
        block = MambaBlock(d_model=64, d_state=16, expand=2)
        x = torch.randn(4, 20, 64)
        out = block(x)
        assert out.shape == (4, 20, 64)

    def test_without_causal_conv(self):
        from myforecaster.models.deep_learning.layers.mamba import MambaBlock
        block = MambaBlock(d_model=64, d_state=16, use_causal_conv=False)
        x = torch.randn(4, 20, 64)
        out = block(x)
        assert out.shape == (4, 20, 64)

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.layers.mamba import MambaBlock
        block = MambaBlock(d_model=32, d_state=8)
        x = torch.randn(2, 10, 32)
        out = block(x)
        out.sum().backward()
        grad_count = sum(1 for p in block.parameters() if p.grad is not None)
        assert grad_count > 0


class TestBidirectionalMambaBlock:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.layers.mamba import BidirectionalMambaBlock
        block = BidirectionalMambaBlock(d_model=64, d_state=16)
        x = torch.randn(4, 20, 64)
        out = block(x)
        assert out.shape == (4, 20, 64)

    def test_residual_connection(self):
        from myforecaster.models.deep_learning.layers.mamba import BidirectionalMambaBlock
        block = BidirectionalMambaBlock(d_model=32, d_state=8)
        block.eval()
        x = torch.randn(2, 5, 32)
        with torch.no_grad():
            out = block(x)
        # Output should not be zero (residual adds input back)
        assert out.abs().sum() > 0


# ---------------------------------------------------------------------------
# S-Mamba tests
# ---------------------------------------------------------------------------
class TestSMambaNetwork:
    def test_univariate_output_shape(self):
        from myforecaster.models.deep_learning.s_mamba import SMambaNetwork
        net = SMambaNetwork(
            context_length=96, prediction_length=24,
            n_variates=1, d_model=32, d_state=8, n_layers=1, d_ff=64,
        )
        x = torch.randn(4, 96)
        out = net(x)
        assert out.shape == (4, 24)

    def test_multivariate_output_shape(self):
        from myforecaster.models.deep_learning.s_mamba import SMambaNetwork
        net = SMambaNetwork(
            context_length=96, prediction_length=24,
            n_variates=3, d_model=32, d_state=8, n_layers=1, d_ff=64,
        )
        x = torch.randn(4, 96, 3)
        out = net(x)
        assert out.shape == (4, 24, 3)

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.s_mamba import SMambaNetwork
        net = SMambaNetwork(
            context_length=64, prediction_length=12,
            d_model=32, d_state=8, n_layers=1,
        )
        x = torch.randn(2, 64)
        target = torch.randn(2, 12)
        pred = net(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0


class TestSMambaModel:
    def test_fit_predict(self, train_test, pred_length):
        from myforecaster.models.deep_learning.s_mamba import SMambaModel
        train, _ = train_test
        m = SMambaModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 2, "context_length": 30,
                "d_model": 32, "d_state": 8, "n_layers": 1, "d_ff": 64,
                "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert np.isfinite(pred["mean"].values).all()

    def test_score_is_finite(self, train_test, pred_length):
        from myforecaster.models.deep_learning.s_mamba import SMambaModel
        train, test = train_test
        m = SMambaModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 2, "context_length": 30,
                "d_model": 32, "d_state": 8, "n_layers": 1,
            },
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)

    def test_registered(self):
        from myforecaster.models.deep_learning.s_mamba import SMambaModel
        from myforecaster.models import MODEL_REGISTRY
        assert "S-Mamba" in MODEL_REGISTRY


# ---------------------------------------------------------------------------
# MambaTS tests
# ---------------------------------------------------------------------------
class TestTemporalMambaBlock:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.mambats import TemporalMambaBlock
        tmb = TemporalMambaBlock(d_model=64, d_state=16)
        x = torch.randn(4, 20, 64)
        out = tmb(x)
        assert out.shape == (4, 20, 64)


class TestMambaTSNetwork:
    def test_univariate_output_shape(self):
        from myforecaster.models.deep_learning.mambats import MambaTSNetwork
        net = MambaTSNetwork(
            context_length=96, prediction_length=24,
            d_model=32, d_state=8, n_layers=1, d_ff=64,
            patch_len=16, stride=8,
        )
        x = torch.randn(4, 96)
        out = net(x)
        assert out.shape == (4, 24)

    def test_n_patches(self):
        from myforecaster.models.deep_learning.mambats import MambaTSNetwork
        net = MambaTSNetwork(
            context_length=96, prediction_length=24,
            patch_len=16, stride=8,
        )
        assert net.n_patches == 11  # (96 - 16) / 8 + 1

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.mambats import MambaTSNetwork
        net = MambaTSNetwork(
            context_length=64, prediction_length=12,
            d_model=32, d_state=8, n_layers=1,
            patch_len=8, stride=4,
        )
        x = torch.randn(2, 64)
        target = torch.randn(2, 12)
        pred = net(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0


class TestMambaTSModel:
    def test_fit_predict(self, train_test, pred_length):
        from myforecaster.models.deep_learning.mambats import MambaTSModel
        train, _ = train_test
        m = MambaTSModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 2, "context_length": 64,
                "d_model": 32, "d_state": 8, "n_layers": 1, "d_ff": 64,
                "patch_len": 8, "stride": 4,
                "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert np.isfinite(pred["mean"].values).all()

    def test_score_is_finite(self, train_test, pred_length):
        from myforecaster.models.deep_learning.mambats import MambaTSModel
        train, test = train_test
        m = MambaTSModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 2, "context_length": 64,
                "d_model": 32, "d_state": 8, "n_layers": 1,
                "patch_len": 8, "stride": 4,
            },
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)

    def test_registered(self):
        from myforecaster.models.deep_learning.mambats import MambaTSModel
        from myforecaster.models import MODEL_REGISTRY
        assert "MambaTS" in MODEL_REGISTRY


# ---------------------------------------------------------------------------
# VPT / VAST paper-specific tests
# ---------------------------------------------------------------------------
class TestVPT:
    def test_vpt_shuffles_during_training(self):
        """VPT should randomly permute variable order during training."""
        from myforecaster.models.deep_learning.mambats import MambaTSNetwork
        net = MambaTSNetwork(
            context_length=32, prediction_length=8,
            n_variates=4, d_model=16, d_state=4, n_layers=1,
            d_ff=32, patch_len=8, stride=4, use_vpt=True,
        )
        net.train()
        x = torch.randn(2, 32, 4)
        out = net(x)
        assert out.shape == (2, 8, 4)
        # After a training forward, _last_perm should be set
        assert net._last_perm is not None
        assert len(net._last_perm) == 4

    def test_vpt_disabled(self):
        """Without VPT, no permutation should happen."""
        from myforecaster.models.deep_learning.mambats import MambaTSNetwork
        net = MambaTSNetwork(
            context_length=32, prediction_length=8,
            n_variates=3, d_model=16, d_state=4, n_layers=1,
            d_ff=32, patch_len=8, stride=4, use_vpt=False,
        )
        net.train()
        x = torch.randn(2, 32, 3)
        net(x)
        assert net._last_perm is None

    def test_multivariate_output_shape(self):
        from myforecaster.models.deep_learning.mambats import MambaTSNetwork
        net = MambaTSNetwork(
            context_length=64, prediction_length=12,
            n_variates=5, d_model=32, d_state=8, n_layers=1,
            d_ff=64, patch_len=8, stride=4,
        )
        x = torch.randn(2, 64, 5)
        out = net(x)
        assert out.shape == (2, 12, 5)


class TestVAST:
    def test_vast_adjacency_update(self):
        """VAST should accumulate adjacency information."""
        from myforecaster.models.deep_learning.mambats import VAST
        vast = VAST(n_variates=4)
        assert vast.count == 0

        # Simulate some training updates
        vast.update(torch.tensor([0, 2, 1, 3]), loss_value=0.5)
        vast.update(torch.tensor([2, 0, 3, 1]), loss_value=0.3)
        assert vast.count == 2
        # adjacency[0][2] should be updated (from first perm)
        assert vast.adjacency[0, 2] > 0

    def test_vast_solve_order(self):
        """VAST should return a valid permutation."""
        from myforecaster.models.deep_learning.mambats import VAST
        vast = VAST(n_variates=4)
        # Add some data
        for _ in range(20):
            perm = torch.randperm(4)
            vast.update(perm, loss_value=torch.rand(1).item() + 0.1)
        order = vast.solve_scan_order()
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}  # valid permutation

    def test_vast_used_at_inference(self):
        """At eval time with trained VAST, network uses solved order."""
        from myforecaster.models.deep_learning.mambats import MambaTSNetwork
        net = MambaTSNetwork(
            context_length=32, prediction_length=8,
            n_variates=3, d_model=16, d_state=4, n_layers=1,
            d_ff=32, patch_len=8, stride=4, use_vpt=True,
        )
        # Simulate VPT training
        net.train()
        x = torch.randn(2, 32, 3)
        for _ in range(10):
            out = net(x)
            loss = out.mean()
            if net._last_perm is not None:
                net.vast.update(torch.tensor(net._last_perm), loss.item())

        # At eval, VAST should have data
        net.eval()
        assert net.vast.count > 0
        with torch.no_grad():
            out = net(x)
        assert out.shape == (2, 8, 3)

