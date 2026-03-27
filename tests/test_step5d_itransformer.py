"""Step 5-d Phase 2: iTransformer verification tests.

Run on your server:
    cd cbal-project
    pytest tests/test_step5d_itransformer.py -v
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


# ---------------------------------------------------------------------------
# InvertedTransformerBlock tests
# ---------------------------------------------------------------------------
class TestInvertedTransformerBlock:
    def test_output_shape(self):
        from cbal.models.deep_learning.itransformer import InvertedTransformerBlock
        block = InvertedTransformerBlock(d_model=64, n_heads=4, d_ff=128)
        x = torch.randn(4, 3, 64)  # B=4, N=3 variates, D=64
        out = block(x)
        assert out.shape == (4, 3, 64)

    def test_single_variate(self):
        from cbal.models.deep_learning.itransformer import InvertedTransformerBlock
        block = InvertedTransformerBlock(d_model=64, n_heads=4, d_ff=128)
        x = torch.randn(4, 1, 64)  # single variate
        out = block(x)
        assert out.shape == (4, 1, 64)

    def test_gradient_flows(self):
        from cbal.models.deep_learning.itransformer import InvertedTransformerBlock
        block = InvertedTransformerBlock(d_model=64, n_heads=4, d_ff=128)
        x = torch.randn(4, 3, 64)
        out = block(x)
        out.sum().backward()
        for p in block.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# iTransformerNetwork tests
# ---------------------------------------------------------------------------
class TestiTransformerNetwork:
    def test_univariate_output_shape(self):
        from cbal.models.deep_learning.itransformer import iTransformerNetwork
        net = iTransformerNetwork(
            context_length=96, prediction_length=24,
            n_variates=1, d_model=64, n_heads=4, n_layers=1, d_ff=128,
        )
        x = torch.randn(4, 96)  # univariate: (B, L)
        out = net(x)
        assert out.shape == (4, 24)

    def test_multivariate_output_shape(self):
        from cbal.models.deep_learning.itransformer import iTransformerNetwork
        net = iTransformerNetwork(
            context_length=96, prediction_length=24,
            n_variates=5, d_model=64, n_heads=4, n_layers=1, d_ff=128,
        )
        x = torch.randn(4, 96, 5)  # multivariate: (B, L, N)
        out = net(x)
        assert out.shape == (4, 24, 5)

    def test_without_revin(self):
        from cbal.models.deep_learning.itransformer import iTransformerNetwork
        net = iTransformerNetwork(
            context_length=96, prediction_length=24,
            n_variates=1, d_model=64, n_heads=4, n_layers=1, revin=False,
        )
        x = torch.randn(4, 96)
        out = net(x)
        assert out.shape == (4, 24)

    def test_deeper_network(self):
        from cbal.models.deep_learning.itransformer import iTransformerNetwork
        net = iTransformerNetwork(
            context_length=96, prediction_length=24,
            n_variates=3, d_model=64, n_heads=4, n_layers=4, d_ff=128,
        )
        x = torch.randn(4, 96, 3)
        out = net(x)
        assert out.shape == (4, 24, 3)

    def test_gradient_flows_full_network(self):
        from cbal.models.deep_learning.itransformer import iTransformerNetwork
        net = iTransformerNetwork(
            context_length=96, prediction_length=24,
            n_variates=1, d_model=64, n_heads=4, n_layers=2,
        )
        x = torch.randn(4, 96)
        target = torch.randn(4, 24)
        pred = net(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0


# ---------------------------------------------------------------------------
# iTransformer Model integration tests
# ---------------------------------------------------------------------------
class TestiTransformerModel:
    def test_fit_predict(self, train_test, pred_length):
        from cbal.models.deep_learning.itransformer import iTransformerModel
        train, _ = train_test
        m = iTransformerModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 64, "n_heads": 4, "n_layers": 1, "d_ff": 128,
                "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns

    def test_prediction_values_are_finite(self, train_test, pred_length):
        from cbal.models.deep_learning.itransformer import iTransformerModel
        train, _ = train_test
        m = iTransformerModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 64, "n_heads": 4, "n_layers": 1,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert np.isfinite(pred["mean"].values).all()

    def test_future_timestamps_correct(self, train_test, pred_length):
        from cbal.models.deep_learning.itransformer import iTransformerModel
        train, _ = train_test
        m = iTransformerModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 2, "context_length": 64,
                "d_model": 64, "n_heads": 4, "n_layers": 1,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        for item_id in train.item_ids:
            last_ts = train.loc[item_id].index.get_level_values("timestamp").max()
            pred_ts = pred.loc[item_id].index.get_level_values("timestamp")
            assert pred_ts.min() > last_ts

    def test_score_is_finite(self, train_test, pred_length):
        from cbal.models.deep_learning.itransformer import iTransformerModel
        train, test = train_test
        m = iTransformerModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 64,
                "d_model": 64, "n_heads": 4, "n_layers": 1,
            },
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)

    def test_registered(self):
        from cbal.models.deep_learning.itransformer import iTransformerModel
        from cbal.models import MODEL_REGISTRY
        assert "iTransformer" in MODEL_REGISTRY
