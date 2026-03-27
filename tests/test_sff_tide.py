"""
Tests for SimpleFeedForward and TiDE deep learning models.

Run (requires torch):
    pytest tests/test_sff_tide.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest

# Skip entire module if torch unavailable
torch = pytest.importorskip("torch")

from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
from myforecaster.models.deep_learning.simple_feedforward import (
    SimpleFeedForwardModel, SimpleFeedForwardNetwork,
)
from myforecaster.models.deep_learning.tide import (
    TiDEModel, TiDENetwork,
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
            })
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


# =====================================================================
# SimpleFeedForward — Network
# =====================================================================

class TestSFFNetwork:
    def test_forward_shape(self):
        net = SimpleFeedForwardNetwork(
            context_length=30, prediction_length=7,
            hidden_dims=[32, 32], distribution="student_t",
        )
        x = torch.randn(4, 30)
        out = net(x)
        assert out.shape == (4, 7, 3)  # student_t: 3 params

    def test_normal_distribution(self):
        net = SimpleFeedForwardNetwork(
            context_length=20, prediction_length=5,
            hidden_dims=[16], distribution="normal",
        )
        x = torch.randn(2, 20)
        out = net(x)
        assert out.shape == (2, 5, 2)  # normal: 2 params

    def test_sample_shape(self):
        net = SimpleFeedForwardNetwork(
            context_length=20, prediction_length=5,
            hidden_dims=[16], distribution="student_t",
        )
        x = torch.randn(3, 20)
        params = net(x)
        samples = net.sample(params, num_samples=50)
        assert samples.shape == (3, 50, 5)

    def test_mean_extraction(self):
        net = SimpleFeedForwardNetwork(
            context_length=10, prediction_length=3,
            hidden_dims=[8],
        )
        x = torch.randn(2, 10)
        params = net(x)
        mean = net.mean(params)
        assert mean.shape == (2, 3)


# =====================================================================
# SimpleFeedForward — Model (fit/predict)
# =====================================================================

class TestSFFModel:
    def test_fit_predict(self):
        data = _make_data()
        train, test = data.train_test_split(7)
        model = SimpleFeedForwardModel(
            freq="D", prediction_length=7,
            hyperparameters={
                "hidden_dims": [16, 16], "max_epochs": 3,
                "batch_size": 16, "num_samples": 20,
            },
        )
        model.fit(train, val_data=test)
        preds = model.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert "mean" in preds.columns
        assert len(preds) == 3 * 7  # 3 items × 7 steps

    def test_score(self):
        data = _make_data()
        train, test = data.train_test_split(7)
        model = SimpleFeedForwardModel(
            freq="D", prediction_length=7,
            hyperparameters={"hidden_dims": [16], "max_epochs": 2},
        )
        model.fit(train, val_data=test)
        score = model.score(test)
        assert np.isfinite(score)

    def test_registry(self):
        from myforecaster.predictor import _create_model
        m = _create_model("SimpleFeedForward", "D", 7, {"max_epochs": 1}, "MAE")
        assert m is not None
        assert "SimpleFeedForward" in type(m).__name__


# =====================================================================
# TiDE — Network
# =====================================================================

class TestTiDENetwork:
    def test_forward_shape(self):
        net = TiDENetwork(
            context_length=30, prediction_length=7,
            hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1,
            decoder_output_dim=8, temporal_decoder_hidden=16,
        )
        x = torch.randn(4, 30)
        out = net(x)
        assert out.shape == (4, 7)

    def test_with_feature_projection(self):
        net = TiDENetwork(
            context_length=20, prediction_length=5,
            hidden_dim=32, feature_projection_dim=8,
        )
        x = torch.randn(3, 20)
        out = net(x)
        assert out.shape == (3, 5)

    def test_skip_connection(self):
        """Verify lookback skip is active (output changes if we zero the skip)."""
        net = TiDENetwork(
            context_length=15, prediction_length=3,
            hidden_dim=16, num_encoder_layers=1, num_decoder_layers=1,
        )
        x = torch.randn(2, 15)
        out1 = net(x).detach()

        # Zero out skip weights
        with torch.no_grad():
            net.lookback_skip.weight.zero_()
            net.lookback_skip.bias.zero_()
        out2 = net(x).detach()

        # Outputs should differ (skip was contributing)
        assert not torch.allclose(out1, out2, atol=1e-6)


# =====================================================================
# TiDE — Model (fit/predict)
# =====================================================================

class TestTiDEModel:
    def test_fit_predict(self):
        data = _make_data()
        train, test = data.train_test_split(7)
        model = TiDEModel(
            freq="D", prediction_length=7,
            hyperparameters={
                "hidden_dim": 32, "num_encoder_layers": 1,
                "num_decoder_layers": 1, "max_epochs": 3,
                "batch_size": 16,
            },
        )
        model.fit(train, val_data=test)
        preds = model.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert "mean" in preds.columns
        assert len(preds) == 3 * 7

    def test_score(self):
        data = _make_data()
        train, test = data.train_test_split(7)
        model = TiDEModel(
            freq="D", prediction_length=7,
            hyperparameters={"hidden_dim": 16, "max_epochs": 2},
        )
        model.fit(train, val_data=test)
        score = model.score(test)
        assert np.isfinite(score)

    def test_registry(self):
        from myforecaster.predictor import _create_model
        m = _create_model("TiDE", "D", 7, {"max_epochs": 1}, "MAE")
        assert m is not None
        assert "TiDE" in type(m).__name__

    def test_suffix_stripping(self):
        from myforecaster.predictor import _create_model
        m = _create_model("TiDEModel", "D", 7, {"max_epochs": 1}, "MAE")
        assert m is not None
