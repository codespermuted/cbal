"""Step 5-d Phase 3 batch 1: N-HiTS, TSMixer, SegRNN verification tests.

Run on your server:
    cd myforecaster-project
    pytest tests/test_step5d_phase3a.py -v
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


HP = {"max_epochs": 2, "batch_size": 16}


# ===========================================================================
# N-HiTS
# ===========================================================================
class TestNHiTSNetwork:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.nhits import NHiTSNetwork
        net = NHiTSNetwork(context_length=96, prediction_length=24,
                           n_stacks=2, hidden_size=64, n_mlp_layers=2)
        out = net(torch.randn(4, 96))
        assert out.shape == (4, 24)

    def test_hierarchical_pooling(self):
        from myforecaster.models.deep_learning.nhits import NHiTSNetwork
        net = NHiTSNetwork(context_length=96, prediction_length=24,
                           n_stacks=3, pooling_kernels=[1, 4, 8])
        assert len(net.blocks) == 3
        assert net.blocks[0].pooling_kernel == 1
        assert net.blocks[2].pooling_kernel == 8

    def test_backcast_forecast_theta_separation(self):
        """Per paper: backcast and forecast use different n_theta."""
        from myforecaster.models.deep_learning.nhits import NHiTSBlock
        block = NHiTSBlock(context_length=96, prediction_length=24,
                           hidden_size=64, pooling_kernel=4)
        # backcast: context_length // pooling = 96 // 4 = 24
        assert block.n_theta_backcast == 24
        # forecast: prediction_length // pooling = 24 // 4 = 6
        assert block.n_theta_forecast == 6
        # They should be DIFFERENT
        assert block.n_theta_backcast != block.n_theta_forecast

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.nhits import NHiTSNetwork
        net = NHiTSNetwork(context_length=64, prediction_length=12,
                           n_stacks=2, hidden_size=32)
        pred = net(torch.randn(2, 64))
        loss = pred.sum()
        loss.backward()
        assert sum(1 for p in net.parameters() if p.grad is not None) > 0


class TestNHiTSModel:
    def test_fit_predict(self, train_test, pred_length):
        from myforecaster.models.deep_learning.nhits import NHiTSModel
        train, _ = train_test
        m = NHiTSModel(freq="D", prediction_length=pred_length,
                        hyperparameters={**HP, "context_length": 64,
                                         "n_stacks": 2, "hidden_size": 64})
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from myforecaster.models.deep_learning.nhits import NHiTSModel
        train, test = train_test
        m = NHiTSModel(freq="D", prediction_length=pred_length,
                        hyperparameters={**HP, "context_length": 64,
                                         "n_stacks": 2, "hidden_size": 64})
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from myforecaster.models.deep_learning.nhits import NHiTSModel
        from myforecaster.models import MODEL_REGISTRY
        assert "N-HiTS" in MODEL_REGISTRY


# ===========================================================================
# TSMixer
# ===========================================================================
class TestTSMixerNetwork:
    def test_univariate_output_shape(self):
        from myforecaster.models.deep_learning.tsmixer import TSMixerNetwork
        net = TSMixerNetwork(context_length=96, prediction_length=24,
                             n_channels=1, d_ff=32, n_layers=2)
        out = net(torch.randn(4, 96))
        assert out.shape == (4, 24)

    def test_multivariate_output_shape(self):
        from myforecaster.models.deep_learning.tsmixer import TSMixerNetwork
        net = TSMixerNetwork(context_length=96, prediction_length=24,
                             n_channels=5, d_ff=32, n_layers=2, revin=False)
        out = net(torch.randn(4, 96, 5))
        assert out.shape == (4, 24, 5)

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.tsmixer import TSMixerNetwork
        net = TSMixerNetwork(context_length=64, prediction_length=12,
                             d_ff=32, n_layers=2)
        pred = net(torch.randn(2, 64))
        pred.sum().backward()
        assert sum(1 for p in net.parameters() if p.grad is not None) > 0


class TestTSMixerModel:
    def test_fit_predict(self, train_test, pred_length):
        from myforecaster.models.deep_learning.tsmixer import TSMixerModel
        train, _ = train_test
        m = TSMixerModel(freq="D", prediction_length=pred_length,
                          hyperparameters={**HP, "context_length": 64,
                                           "d_ff": 32, "n_layers": 2})
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from myforecaster.models.deep_learning.tsmixer import TSMixerModel
        train, test = train_test
        m = TSMixerModel(freq="D", prediction_length=pred_length,
                          hyperparameters={**HP, "context_length": 64,
                                           "d_ff": 32, "n_layers": 2})
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from myforecaster.models.deep_learning.tsmixer import TSMixerModel
        from myforecaster.models import MODEL_REGISTRY
        assert "TSMixer" in MODEL_REGISTRY


# ===========================================================================
# SegRNN
# ===========================================================================
class TestSegRNNNetwork:
    def test_rmr_output_shape(self):
        from myforecaster.models.deep_learning.segrnn import SegRNNNetwork
        net = SegRNNNetwork(context_length=96, prediction_length=24,
                            seg_len=12, d_model=64, strategy="rmr")
        out = net(torch.randn(4, 96))
        assert out.shape == (4, 24)

    def test_pmr_output_shape(self):
        from myforecaster.models.deep_learning.segrnn import SegRNNNetwork
        net = SegRNNNetwork(context_length=96, prediction_length=24,
                            seg_len=12, d_model=64, strategy="pmr")
        out = net(torch.randn(4, 96))
        assert out.shape == (4, 24)

    def test_different_seg_lengths(self):
        from myforecaster.models.deep_learning.segrnn import SegRNNNetwork
        for seg in [6, 12, 24]:
            for strat in ["rmr", "pmr"]:
                net = SegRNNNetwork(context_length=96, prediction_length=24,
                                    seg_len=seg, d_model=32, strategy=strat)
                out = net(torch.randn(4, 96))
                assert out.shape == (4, 24), f"Failed for seg={seg}, strategy={strat}"

    def test_gradient_flows_rmr(self):
        from myforecaster.models.deep_learning.segrnn import SegRNNNetwork
        net = SegRNNNetwork(context_length=64, prediction_length=12,
                            seg_len=8, d_model=32, strategy="rmr")
        pred = net(torch.randn(2, 64))
        pred.sum().backward()
        assert sum(1 for p in net.parameters() if p.grad is not None) > 0

    def test_gradient_flows_pmr(self):
        from myforecaster.models.deep_learning.segrnn import SegRNNNetwork
        net = SegRNNNetwork(context_length=64, prediction_length=12,
                            seg_len=8, d_model=32, strategy="pmr")
        pred = net(torch.randn(2, 64))
        pred.sum().backward()
        assert sum(1 for p in net.parameters() if p.grad is not None) > 0


class TestSegRNNModel:
    def test_fit_predict_rmr(self, train_test, pred_length):
        from myforecaster.models.deep_learning.segrnn import SegRNNModel
        train, _ = train_test
        m = SegRNNModel(freq="D", prediction_length=pred_length,
                         hyperparameters={**HP, "context_length": 48,
                                          "seg_len": 6, "d_model": 32,
                                          "strategy": "rmr"})
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert np.isfinite(pred["mean"].values).all()

    def test_fit_predict_pmr(self, train_test, pred_length):
        from myforecaster.models.deep_learning.segrnn import SegRNNModel
        train, _ = train_test
        m = SegRNNModel(freq="D", prediction_length=pred_length,
                         hyperparameters={**HP, "context_length": 48,
                                          "seg_len": 6, "d_model": 32,
                                          "strategy": "pmr"})
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert np.isfinite(pred["mean"].values).all()

    def test_score(self, train_test, pred_length):
        from myforecaster.models.deep_learning.segrnn import SegRNNModel
        train, test = train_test
        m = SegRNNModel(freq="D", prediction_length=pred_length,
                         hyperparameters={**HP, "context_length": 48,
                                          "seg_len": 6, "d_model": 32})
        m.fit(train)
        assert np.isfinite(m.score(test, metric="MAE"))

    def test_registered(self):
        from myforecaster.models.deep_learning.segrnn import SegRNNModel
        from myforecaster.models import MODEL_REGISTRY
        assert "SegRNN" in MODEL_REGISTRY
