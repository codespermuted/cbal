"""Step 5-d Phase 1: Deep Learning models (DLinear + DeepAR) verification tests.

These tests require PyTorch. Run on your server with:
    cd cbal-project
    pip install -e ".[deep,dev]"
    pytest tests/test_step5d_dl_phase1.py -v
"""

import numpy as np
import pandas as pd
import pytest

from cbal.dataset import TimeSeriesDataFrame

# Skip if torch is not available.
# In environments where torch import hangs (broken CUDA), set:
#   MYFORECASTER_SKIP_TORCH=1
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
    """3 items, 200 daily obs, trend + seasonality + noise."""
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
# Dataset tests
# ---------------------------------------------------------------------------
class TestTimeSeriesDataset:
    def test_train_dataset_creation(self, train_test, pred_length):
        from cbal.models.deep_learning.dataset import TimeSeriesDataset
        train, _ = train_test
        ds = TimeSeriesDataset(train, context_length=30, prediction_length=pred_length,
                               freq="D", mode="train")
        assert len(ds) > 0

    def test_train_sample_shapes(self, train_test, pred_length):
        from cbal.models.deep_learning.dataset import TimeSeriesDataset
        train, _ = train_test
        ds = TimeSeriesDataset(train, context_length=30, prediction_length=pred_length,
                               freq="D", mode="train")
        sample = ds[0]
        assert sample["past_target"].shape == (30,)
        assert sample["future_target"].shape == (pred_length,)
        assert sample["past_time_features"].shape[0] == 30
        assert sample["future_time_features"].shape[0] == pred_length

    def test_predict_dataset_creation(self, train_test, pred_length):
        from cbal.models.deep_learning.dataset import TimeSeriesDataset
        train, _ = train_test
        ds = TimeSeriesDataset(train, context_length=30, prediction_length=pred_length,
                               freq="D", mode="predict")
        # One sample per item
        assert len(ds) == 3

    def test_dataloader_works(self, train_test, pred_length):
        from cbal.models.deep_learning.dataset import TimeSeriesDataset
        from torch.utils.data import DataLoader
        train, _ = train_test
        ds = TimeSeriesDataset(train, context_length=30, prediction_length=pred_length,
                               freq="D", mode="train")
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch["past_target"].shape[0] <= 4
        assert batch["past_target"].shape[1] == 30


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------
class TestDistributions:
    def test_gaussian_output(self):
        from cbal.models.deep_learning.layers.distributions import GaussianOutput
        head = GaussianOutput(input_dim=32)
        x = torch.randn(4, 10, 32)
        params = head(x)
        assert len(params) == 2  # mu, sigma
        assert params[0].shape == (4, 10)
        assert params[1].shape == (4, 10)
        assert (params[1] > 0).all()  # sigma positive

    def test_gaussian_loss(self):
        from cbal.models.deep_learning.layers.distributions import GaussianOutput
        head = GaussianOutput(input_dim=16)
        x = torch.randn(4, 5, 16)
        params = head(x)
        target = torch.randn(4, 5)
        loss = head.loss(params, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gaussian_sample(self):
        from cbal.models.deep_learning.layers.distributions import GaussianOutput
        head = GaussianOutput(input_dim=16)
        x = torch.randn(4, 5, 16)
        params = head(x)
        samples = head.sample(params, n_samples=50)
        assert samples.shape == (50, 4, 5)

    def test_gaussian_quantile(self):
        from cbal.models.deep_learning.layers.distributions import GaussianOutput
        head = GaussianOutput(input_dim=16)
        x = torch.randn(4, 5, 16)
        params = head(x)
        q = head.quantile(params, [0.1, 0.5, 0.9])
        assert q.shape == (4, 5, 3)
        # q10 <= q50 <= q90
        assert (q[:, :, 0] <= q[:, :, 1] + 1e-5).all()
        assert (q[:, :, 1] <= q[:, :, 2] + 1e-5).all()

    def test_student_t_output(self):
        from cbal.models.deep_learning.layers.distributions import StudentTOutput
        head = StudentTOutput(input_dim=32)
        x = torch.randn(4, 10, 32)
        params = head(x)
        assert len(params) == 3  # mu, sigma, nu

    def test_get_distribution_output(self):
        from cbal.models.deep_learning.layers.distributions import get_distribution_output
        for name in ["gaussian", "student_t", "negative_binomial"]:
            head = get_distribution_output(name, 16)
            assert head is not None


# ---------------------------------------------------------------------------
# Embeddings tests
# ---------------------------------------------------------------------------
class TestEmbeddings:
    def test_cyclic_date_embedding(self):
        from cbal.models.deep_learning.layers.embeddings import CyclicDateEmbedding
        emb = CyclicDateEmbedding(freq="D")
        assert emb.output_dim == 2 * 4  # 4 fields for daily: dow, dom, month, woy

    def test_cyclic_with_projection(self):
        from cbal.models.deep_learning.layers.embeddings import CyclicDateEmbedding
        emb = CyclicDateEmbedding(freq="h", embed_dim=16)
        assert emb.output_dim == 16

    def test_positional_encoding(self):
        from cbal.models.deep_learning.layers.embeddings import PositionalEncoding
        pe = PositionalEncoding(d_model=32, max_len=100)
        x = torch.randn(4, 50, 32)
        out = pe(x)
        assert out.shape == (4, 50, 32)

    def test_value_embedding(self):
        from cbal.models.deep_learning.layers.embeddings import ValueEmbedding
        ve = ValueEmbedding(input_dim=1, d_model=32)
        x = torch.randn(4, 50, 1)
        out = ve(x)
        assert out.shape == (4, 50, 32)


# ---------------------------------------------------------------------------
# DLinear model tests
# ---------------------------------------------------------------------------
class TestDLinear:
    def test_network_forward(self):
        from cbal.models.deep_learning.dlinear import DLinearNetwork
        net = DLinearNetwork(context_length=30, prediction_length=7, n_channels=1)
        x = torch.randn(4, 30, 1)
        out = net(x)
        assert out.shape == (4, 7, 1)

    def test_fit_predict(self, train_test, pred_length):
        from cbal.models.deep_learning import DLinearModel
        train, _ = train_test
        m = DLinearModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"max_epochs": 3, "context_length": 30, "batch_size": 16},
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length  # 3 items
        assert "mean" in pred.columns

    def test_registered(self):
        from cbal.models import MODEL_REGISTRY
        assert "DLinear" in MODEL_REGISTRY

    def test_future_timestamps(self, train_test, pred_length):
        from cbal.models.deep_learning import DLinearModel
        train, _ = train_test
        m = DLinearModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"max_epochs": 2, "context_length": 30},
        )
        m.fit(train)
        pred = m.predict(train)
        for item_id in train.item_ids:
            last_ts = train.loc[item_id].index.get_level_values("timestamp").max()
            pred_ts = pred.loc[item_id].index.get_level_values("timestamp")
            assert pred_ts.min() > last_ts

    def test_score(self, train_test, pred_length):
        from cbal.models.deep_learning import DLinearModel
        train, test = train_test
        m = DLinearModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={"max_epochs": 3, "context_length": 30},
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# DeepAR model tests
# ---------------------------------------------------------------------------
class TestDeepAR:
    def test_network_forward_train(self):
        from cbal.models.deep_learning.deepar import DeepARNetwork
        net = DeepARNetwork(hidden_size=32, num_layers=1, n_time_features=5,
                            lags=[1, 2, 7], n_items=3, embedding_dim=8)
        B, C, H = 4, 30, 7
        out = net(
            past_target=torch.randn(B, C),
            past_time_features=torch.randn(B, C, 5),
            future_time_features=torch.randn(B, H, 5),
            future_target=torch.randn(B, H),
            item_id_index=torch.tensor([0, 1, 2, 0]),
            past_age=torch.arange(C).float().unsqueeze(0).expand(B, -1),
            future_age=torch.arange(C, C + H).float().unsqueeze(0).expand(B, -1),
        )
        assert "loss" in out
        assert torch.isfinite(out["loss"])
        assert "scale" in out

    def test_item_embedding_different_items(self):
        """Different item IDs should produce different outputs."""
        from cbal.models.deep_learning.deepar import DeepARNetwork
        net = DeepARNetwork(hidden_size=16, num_layers=1, lags=[1],
                            n_items=5, embedding_dim=8)
        net.eval()
        B, C, H = 2, 20, 5
        past = torch.randn(1, C).expand(B, -1)  # same data
        time_f = torch.randn(1, C, 5).expand(B, -1, -1)
        fut_f = torch.randn(1, H, 5).expand(B, -1, -1)
        age = torch.arange(C).float().unsqueeze(0).expand(B, -1)
        fut_age = torch.arange(C, C + H).float().unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            # Same data, different item IDs → should produce different results
            s1 = net.sample_trajectories(
                past, time_f, fut_f, n_samples=10,
                item_id_index=torch.tensor([0, 0]),
                past_age=age, future_age=fut_age,
            )
            s2 = net.sample_trajectories(
                past, time_f, fut_f, n_samples=10,
                item_id_index=torch.tensor([3, 3]),
                past_age=age, future_age=fut_age,
            )
        # Means should differ because embeddings differ
        diff = (s1.mean(0) - s2.mean(0)).abs().mean()
        assert diff > 0.001, "Different items produced identical outputs"

    def test_age_covariate_in_dataset(self):
        """Dataset should include age features."""
        from cbal.models.deep_learning.dataset import TimeSeriesDataset
        from cbal.dataset import TimeSeriesDataFrame
        import pandas as pd
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        rows = [{"item_id": "A", "timestamp": d, "target": float(i)}
                for i, d in enumerate(dates)]
        tsdf = TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))
        ds = TimeSeriesDataset(tsdf, context_length=20, prediction_length=5,
                               freq="D", mode="train")
        sample = ds[0]
        assert "past_age" in sample
        assert "future_age" in sample
        assert "item_id_index" in sample
        assert sample["past_age"].shape == (20,)
        assert sample["future_age"].shape == (5,)
        # Age should be increasing
        assert (sample["past_age"][1:] >= sample["past_age"][:-1]).all()

    def test_auto_scaling(self):
        """Verify that inputs are scaled and outputs rescaled."""
        from cbal.models.deep_learning.deepar import DeepARNetwork
        net = DeepARNetwork(hidden_size=16, num_layers=1, lags=[1, 2])
        net.eval()
        # Create data with very different scales
        B, C, H = 2, 20, 5
        past_big = torch.randn(B, C) * 1000 + 5000
        past_small = torch.randn(B, C) * 0.01
        with torch.no_grad():
            out_big = net.sample_trajectories(past_big, torch.randn(B, C, 5),
                                              torch.randn(B, H, 5), n_samples=5)
            out_small = net.sample_trajectories(past_small, torch.randn(B, C, 5),
                                                torch.randn(B, H, 5), n_samples=5)
        # Big input should produce bigger output (scaling works)
        assert out_big.abs().mean() > out_small.abs().mean() * 10

    def test_trajectory_sampling_produces_different_paths(self):
        """Key fix: each sample trajectory should be DIFFERENT."""
        from cbal.models.deep_learning.deepar import DeepARNetwork
        net = DeepARNetwork(hidden_size=32, num_layers=1, lags=[1, 2])
        net.eval()
        B, C, H = 2, 20, 7
        with torch.no_grad():
            samples = net.sample_trajectories(
                torch.randn(B, C) * 10 + 100,
                torch.randn(B, C, 5),
                torch.randn(B, H, 5),
                n_samples=50,
            )  # (50, B, H)
        assert samples.shape == (50, B, H)
        # Different trajectories should NOT be identical
        std_across_samples = samples.std(dim=0)  # (B, H)
        assert std_across_samples.mean() > 0.01, "All trajectories are identical!"

    def test_lagged_features(self):
        """Verify lag extraction works."""
        from cbal.models.deep_learning.deepar import _extract_lags
        series = torch.arange(10, dtype=torch.float).unsqueeze(0)  # (1, 10)
        lags = _extract_lags(series, [1, 3])  # (1, 10, 2)
        assert lags.shape == (1, 10, 2)
        # lag=1: series[t-1]. At t=1, should be series[0]=0
        assert lags[0, 1, 0].item() == 0.0
        # lag=3: at t=3, should be series[0]=0
        assert lags[0, 3, 1].item() == 0.0
        # lag=1: at t=5, should be series[4]=4
        assert lags[0, 5, 0].item() == 4.0

    def test_freq_lag_detection(self):
        from cbal.models.deep_learning.deepar import _get_lags_for_freq
        daily_lags = _get_lags_for_freq("D")
        assert 7 in daily_lags  # weekly lag for daily data
        hourly_lags = _get_lags_for_freq("h")
        assert 24 in hourly_lags  # daily lag for hourly data

    def test_network_predict_quantiles(self):
        from cbal.models.deep_learning.deepar import DeepARNetwork
        net = DeepARNetwork(hidden_size=32, num_layers=1, n_time_features=5,
                            lags=[1, 2, 7])
        net.eval()
        B, C, H = 4, 30, 7
        with torch.no_grad():
            result = net.predict_quantiles(
                past_target=torch.randn(B, C) * 10 + 100,
                past_time_features=torch.randn(B, C, 5),
                future_time_features=torch.randn(B, H, 5),
                quantile_levels=[0.1, 0.5, 0.9],
                n_samples=50,
            )
        assert result["mean"].shape == (B, H)
        assert 0.1 in result["quantiles"]

    def test_fit_predict(self, train_test, pred_length):
        from cbal.models.deep_learning import DeepARModel
        train, _ = train_test
        m = DeepARModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 30,
                "hidden_size": 16, "num_layers": 1, "batch_size": 16,
                "n_samples": 20,
            },
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        assert len(pred) == 3 * pred_length
        assert "mean" in pred.columns
        assert "0.1" in pred.columns
        assert "0.9" in pred.columns

    def test_quantile_spread(self, train_test, pred_length):
        """After the fix, q10 and q90 should NOT be identical."""
        from cbal.models.deep_learning import DeepARModel
        train, _ = train_test
        m = DeepARModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 5, "context_length": 30,
                "hidden_size": 16, "num_layers": 1,
                "n_samples": 50,
            },
        )
        m.fit(train)
        pred = m.predict(train, quantile_levels=[0.1, 0.5, 0.9])
        spread = (pred["0.9"].values - pred["0.1"].values).mean()
        assert spread > 0.1, f"Quantile spread too small: {spread}"

    def test_registered(self):
        from cbal.models import MODEL_REGISTRY
        assert "DeepAR" in MODEL_REGISTRY

    def test_student_t_distribution(self, train_test, pred_length):
        from cbal.models.deep_learning import DeepARModel
        train, _ = train_test
        m = DeepARModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 2, "context_length": 30,
                "hidden_size": 16, "num_layers": 1,
                "distribution": "student_t", "n_samples": 20,
            },
        )
        m.fit(train)
        pred = m.predict(train)
        assert len(pred) == 3 * pred_length

    def test_score(self, train_test, pred_length):
        from cbal.models.deep_learning import DeepARModel
        train, test = train_test
        m = DeepARModel(
            freq="D", prediction_length=pred_length,
            hyperparameters={
                "max_epochs": 3, "context_length": 30,
                "hidden_size": 16, "num_layers": 1,
            },
        )
        m.fit(train)
        score = m.score(test, metric="MAE")
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Model registry completeness
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_dl_models_in_registry(self):
        # Trigger registration by importing
        from cbal.models.deep_learning import DLinearModel, DeepARModel
        from cbal.models import MODEL_REGISTRY
        assert "DLinear" in MODEL_REGISTRY
        assert "DeepAR" in MODEL_REGISTRY
