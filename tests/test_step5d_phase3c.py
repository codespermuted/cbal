"""Step 5-d Phase 3c: MTGNN and CrossGNN verification tests.

Run on your server:
    cd myforecaster-project
    pytest tests/test_step5d_phase3c.py -v
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


# ===========================================================================
# MTGNN tests
# ===========================================================================
class TestGraphLearning:
    def test_adjacency_shape(self):
        from myforecaster.models.deep_learning.mtgnn import GraphLearning
        gl = GraphLearning(n_nodes=5, embed_dim=10)
        adj = gl()
        assert adj.shape == (5, 5)
        # Rows should sum to ~1 (softmax)
        assert (adj.sum(dim=1) - 1.0).abs().max() < 1e-5

    def test_adjacency_is_directed(self):
        """MTGNN learns uni-directed (asymmetric) adjacency."""
        from myforecaster.models.deep_learning.mtgnn import GraphLearning
        gl = GraphLearning(n_nodes=4, embed_dim=8)
        adj = gl()
        # Should NOT be symmetric in general
        # (could be by chance, but very unlikely with random init)
        assert not torch.allclose(adj, adj.T, atol=0.01)


class TestMixhopGraphConv:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.mtgnn import MixhopGraphConv, GraphLearning
        gc = MixhopGraphConv(d_in=16, d_out=16, n_hops=2)
        gl = GraphLearning(n_nodes=5)
        adj = gl()
        x = torch.randn(2, 5, 16)
        out = gc(x, adj)
        assert out.shape == (2, 5, 16)


class TestMTGNNNetwork:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.mtgnn import MTGNNNetwork
        net = MTGNNNetwork(n_nodes=3, context_length=64, prediction_length=12,
                            d_model=16, n_layers=2, embed_dim=8)
        x = torch.randn(2, 64, 3)  # 3 variates
        out = net(x)
        assert out.shape == (2, 12, 3)

    def test_univariate(self):
        from myforecaster.models.deep_learning.mtgnn import MTGNNNetwork
        net = MTGNNNetwork(n_nodes=1, context_length=64, prediction_length=12,
                            d_model=16, n_layers=1)
        x = torch.randn(2, 64, 1)
        out = net(x)
        assert out.shape == (2, 12, 1)

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.mtgnn import MTGNNNetwork
        net = MTGNNNetwork(n_nodes=2, context_length=32, prediction_length=8,
                            d_model=8, n_layers=1, embed_dim=4)
        x = torch.randn(2, 32, 2)
        loss = net(x).mean()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0


class TestMTGNNModel:
    def test_fit_predict(self, train_test, pred_length):
        from myforecaster.models.deep_learning.mtgnn import MTGNNModel
        train, _ = train_test
        m = MTGNNModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 16, "n_layers": 1, "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from myforecaster.models.deep_learning.mtgnn import MTGNNModel
        train, test = train_test
        m = MTGNNModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 16, "n_layers": 1,
            },
        )
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from myforecaster.models.deep_learning.mtgnn import MTGNNModel
        from myforecaster.models import MODEL_REGISTRY
        assert "MTGNN" in MODEL_REGISTRY


# ===========================================================================
# CrossGNN tests
# ===========================================================================
class TestAMSI:
    def test_multiscale_output(self):
        from myforecaster.models.deep_learning.crossgnn import AMSI
        amsi = AMSI(context_length=96, n_scales=4)
        x = torch.randn(2, 96, 16)
        scales = amsi(x)
        assert len(scales) == 4
        assert scales[0].shape == (2, 96, 16)
        assert scales[1].shape == (2, 48, 16)
        assert scales[3].shape == (2, 12, 16)

    def test_learnable_weights(self):
        from myforecaster.models.deep_learning.crossgnn import AMSI
        amsi = AMSI(context_length=64, n_scales=3)
        w = amsi.get_weights()
        assert w.shape == (3,)
        assert (w.sum() - 1.0).abs() < 1e-5


class TestCrossScaleGNN:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.crossgnn import CrossScaleGNN
        gnn = CrossScaleGNN(d_model=16, n_scales=4)
        feats = [torch.randn(2, 16) for _ in range(4)]
        out = gnn(feats)
        assert len(out) == 4
        assert out[0].shape == (2, 16)


class TestCrossVariableGNN:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.crossgnn import CrossVariableGNN
        gnn = CrossVariableGNN(d_model=16, n_vars=5)
        x = torch.randn(2, 5, 16)  # 5 variables
        out = gnn(x)
        assert out.shape == (2, 5, 16)


class TestCrossGNNNetwork:
    def test_univariate_output(self):
        from myforecaster.models.deep_learning.crossgnn import CrossGNNNetwork
        net = CrossGNNNetwork(context_length=64, prediction_length=12,
                               n_vars=1, d_model=16, n_scales=3, n_layers=1)
        x = torch.randn(2, 64)
        out = net(x)
        assert out.shape == (2, 12)

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.crossgnn import CrossGNNNetwork
        net = CrossGNNNetwork(context_length=32, prediction_length=8,
                               d_model=8, n_scales=2, n_layers=1)
        x = torch.randn(2, 32)
        loss = net(x).mean()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0


class TestCrossGNNModel:
    def test_fit_predict(self, train_test, pred_length):
        from myforecaster.models.deep_learning.crossgnn import CrossGNNModel
        train, _ = train_test
        m = CrossGNNModel(
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
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from myforecaster.models.deep_learning.crossgnn import CrossGNNModel
        train, test = train_test
        m = CrossGNNModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 16, "n_scales": 3, "n_layers": 1,
            },
        )
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from myforecaster.models.deep_learning.crossgnn import CrossGNNModel
        from myforecaster.models import MODEL_REGISTRY
        assert "CrossGNN" in MODEL_REGISTRY
