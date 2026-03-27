"""Step 5-d Phase 2: TFT (Temporal Fusion Transformer) verification tests.

Run on your server:
    cd myforecaster-project
    pytest tests/test_step5d_tft.py -v
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
# Building block tests
# ---------------------------------------------------------------------------
class TestGRN:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.tft import GRN
        grn = GRN(d_in=32, d_hidden=64, d_out=32)
        x = torch.randn(4, 10, 32)
        out = grn(x)
        assert out.shape == (4, 10, 32)

    def test_with_context(self):
        from myforecaster.models.deep_learning.tft import GRN
        grn = GRN(d_in=32, d_hidden=64, d_out=32, d_context=16)
        x = torch.randn(4, 10, 32)
        ctx = torch.randn(4, 10, 16)
        out = grn(x, ctx)
        assert out.shape == (4, 10, 32)

    def test_dimension_change(self):
        from myforecaster.models.deep_learning.tft import GRN
        grn = GRN(d_in=32, d_hidden=64, d_out=48)
        x = torch.randn(4, 10, 32)
        out = grn(x)
        assert out.shape == (4, 10, 48)


class TestVSN:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.tft import VariableSelectionNetwork
        vsn = VariableSelectionNetwork(n_vars=3, d_model=32, d_input_per_var=32)
        x_list = [torch.randn(4, 10, 32) for _ in range(3)]
        out = vsn(x_list)
        assert out.shape == (4, 10, 32)

    def test_with_context(self):
        from myforecaster.models.deep_learning.tft import VariableSelectionNetwork
        vsn = VariableSelectionNetwork(n_vars=2, d_model=32, d_input_per_var=32, d_context=16)
        x_list = [torch.randn(4, 10, 32) for _ in range(2)]
        ctx = torch.randn(4, 16)
        out = vsn(x_list, ctx)
        assert out.shape == (4, 10, 32)


class TestInterpretableAttention:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.tft import InterpretableMultiHeadAttention
        attn = InterpretableMultiHeadAttention(d_model=32, n_heads=4)
        q = torch.randn(4, 7, 32)   # decoder
        k = torch.randn(4, 30, 32)  # encoder
        v = k
        out, attn_weights = attn(q, k, v)
        assert out.shape == (4, 7, 32)
        assert attn_weights.shape == (4, 7, 30)

    def test_attention_sums_to_one(self):
        from myforecaster.models.deep_learning.tft import InterpretableMultiHeadAttention
        attn = InterpretableMultiHeadAttention(d_model=32, n_heads=4)
        attn.eval()  # disable dropout so softmax sums to 1
        q = torch.randn(4, 7, 32)
        k = torch.randn(4, 30, 32)
        with torch.no_grad():
            _, attn_weights = attn(q, k, k)
        sums = attn_weights.sum(dim=-1)
        assert (sums - 1.0).abs().max() < 1e-5


class TestQuantileLoss:
    def test_perfect_prediction(self):
        from myforecaster.models.deep_learning.tft import QuantileLoss
        loss_fn = QuantileLoss([0.1, 0.5, 0.9])
        target = torch.randn(4, 7)
        pred = target.unsqueeze(-1).expand(-1, -1, 3)  # perfect for all quantiles
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-6

    def test_loss_is_positive(self):
        from myforecaster.models.deep_learning.tft import QuantileLoss
        loss_fn = QuantileLoss([0.1, 0.5, 0.9])
        pred = torch.randn(4, 7, 3)
        target = torch.randn(4, 7)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# TFT Network tests
# ---------------------------------------------------------------------------
class TestTFTNetwork:
    def test_output_shape(self):
        from myforecaster.models.deep_learning.tft import TFTNetwork
        net = TFTNetwork(
            context_length=30, prediction_length=7,
            d_model=32, n_heads=4, n_lstm_layers=1, n_quantiles=3,
        )
        B, C, H, F = 4, 30, 7, 5
        out = net(
            past_target=torch.randn(B, C),
            past_time_features=torch.randn(B, C, F),
            future_time_features=torch.randn(B, H, F),
        )
        assert out.shape == (B, H, 3)

    def test_gradient_flows(self):
        from myforecaster.models.deep_learning.tft import TFTNetwork
        net = TFTNetwork(
            context_length=30, prediction_length=7,
            d_model=32, n_heads=4, n_lstm_layers=1,
        )
        B, C, H, F = 4, 30, 7, 5
        out = net(
            past_target=torch.randn(B, C),
            past_time_features=torch.randn(B, C, F),
            future_time_features=torch.randn(B, H, F),
        )
        loss = out.mean()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_different_quantile_count(self):
        from myforecaster.models.deep_learning.tft import TFTNetwork
        net = TFTNetwork(
            context_length=30, prediction_length=7,
            d_model=32, n_heads=4, n_lstm_layers=1, n_quantiles=5,
        )
        B = 4
        out = net(
            past_target=torch.randn(B, 30),
            past_time_features=torch.randn(B, 30, 5),
            future_time_features=torch.randn(B, 7, 5),
        )
        assert out.shape == (B, 7, 5)


# ---------------------------------------------------------------------------
# TFT Model integration tests
# ---------------------------------------------------------------------------
class TestTFTModel:
    def test_fit_predict(self, train_test, pred_length):
        from myforecaster.models.deep_learning.tft import TFTModel
        train, _ = train_test
        m = TFTModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 30,
                "d_model": 32, "n_heads": 4, "n_lstm_layers": 1,
                "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert "0.1" in pred.columns
        assert "0.5" in pred.columns
        assert "0.9" in pred.columns

    def test_quantile_ordering(self, train_test, pred_length):
        from myforecaster.models.deep_learning.tft import TFTModel
        train, _ = train_test
        m = TFTModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 5, "context_length": 30,
                "d_model": 32, "n_heads": 4, "n_lstm_layers": 1,
                "batch_size": 16,
            },
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        # After some training, quantiles should mostly be ordered
        q10 = pred["0.1"].values
        q90 = pred["0.9"].values
        ordering_ratio = (q10 <= q90 + 1.0).mean()
        assert ordering_ratio > 0.8  # allow some tolerance for 3 epochs

    def test_prediction_values_are_finite(self, train_test, pred_length):
        from myforecaster.models.deep_learning.tft import TFTModel
        train, _ = train_test
        m = TFTModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 30,
                "d_model": 32, "n_heads": 4, "n_lstm_layers": 1,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert np.isfinite(pred["mean"].values).all()

    def test_score_is_finite(self, train_test, pred_length):
        from myforecaster.models.deep_learning.tft import TFTModel
        train, test = train_test
        m = TFTModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 30,
                "d_model": 32, "n_heads": 4, "n_lstm_layers": 1,
            },
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)

    def test_registered(self):
        from myforecaster.models.deep_learning.tft import TFTModel
        from myforecaster.models import MODEL_REGISTRY
        assert "TFT" in MODEL_REGISTRY

    def test_custom_quantiles(self, train_test, pred_length):
        from myforecaster.models.deep_learning.tft import TFTModel
        train, _ = train_test
        m = TFTModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 2, "context_length": 30,
                "d_model": 32, "n_heads": 4, "n_lstm_layers": 1,
                "quantile_levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            },
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95])
        assert "0.05" in pred.columns
        assert "0.95" in pred.columns


# ---------------------------------------------------------------------------
# Paper-specific feature verification
# ---------------------------------------------------------------------------
class TestTFTPaperFeatures:
    """Verify TFT implements all features from Lim et al. (2021)."""

    def test_static_covariate_encoder_produces_4_contexts(self):
        from myforecaster.models.deep_learning.tft import StaticCovariateEncoder
        enc = StaticCovariateEncoder(d_static=16, d_model=32)
        static = torch.randn(4, 16)
        c_s, c_e, c_h, c_c = enc(static)
        assert c_s.shape == (4, 32)
        assert c_e.shape == (4, 32)
        assert c_h.shape == (4, 32)
        assert c_c.shape == (4, 32)
        # 4 contexts should be different (different GRNs)
        assert not torch.allclose(c_s, c_e)

    def test_entity_embedding_in_network(self):
        from myforecaster.models.deep_learning.tft import TFTNetwork
        net = TFTNetwork(context_length=20, prediction_length=5,
                         d_model=32, n_heads=4, n_items=5, embedding_dim=8)
        assert hasattr(net, 'entity_embedding')
        assert net.entity_embedding.num_embeddings == 5
        assert net.entity_embedding.embedding_dim == 8

    def test_different_items_produce_different_outputs(self):
        from myforecaster.models.deep_learning.tft import TFTNetwork
        net = TFTNetwork(context_length=20, prediction_length=5,
                         d_model=32, n_heads=4, n_items=5, embedding_dim=8)
        net.eval()
        past = torch.randn(1, 20)
        time_f = torch.randn(1, 20, 5)
        fut_f = torch.randn(1, 5, 5)
        with torch.no_grad():
            out1 = net(past, time_f, fut_f, item_id_index=torch.tensor([0]))
            out2 = net(past, time_f, fut_f, item_id_index=torch.tensor([3]))
        diff = (out1 - out2).abs().mean()
        assert diff > 0.001, "Different items should produce different outputs"

    def test_3_separate_vsns_exist(self):
        from myforecaster.models.deep_learning.tft import TFTNetwork
        net = TFTNetwork(context_length=20, prediction_length=5, d_model=32, n_heads=4)
        assert hasattr(net, 'vsn_past_observed')
        assert hasattr(net, 'vsn_past_known')
        assert hasattr(net, 'vsn_future_known')

    def test_lstm_init_projections_exist(self):
        from myforecaster.models.deep_learning.tft import TFTNetwork
        net = TFTNetwork(context_length=20, prediction_length=5,
                         d_model=32, n_heads=4, n_lstm_layers=2)
        assert hasattr(net, 'h_init_proj')
        assert hasattr(net, 'c_init_proj')
        # Output should be n_layers * d_model
        assert net.h_init_proj.out_features == 2 * 32
