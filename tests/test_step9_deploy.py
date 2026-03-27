"""
Tests for Step 9: Save/Load/Deploy — CLI, FastAPI serving, roundtrip.

Run:
    MYFORECASTER_SKIP_TORCH=1 pytest tests/test_step9_deploy.py -v
"""

import csv
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
from myforecaster.predictor import TimeSeriesPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    np.random.seed(42)
    rows = []
    for item in ["A", "B", "C"]:
        base = np.random.rand() * 10 + 5
        for t in range(100):
            rows.append({
                "item_id": item,
                "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=t),
                "target": base + 0.1 * t + np.random.randn() * 0.5,
            })
    return TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))


@pytest.fixture
def fitted_predictor(sample_data):
    """Return a fitted predictor saved to a temp directory."""
    p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE")
    p.fit(
        sample_data,
        presets={
            "models": {"Naive": {}, "SeasonalNaive": {}, "Average": {}},
            "ensemble": "SimpleAverage",
        },
    )
    return p


@pytest.fixture
def saved_predictor_path(fitted_predictor, tmp_path):
    """Save predictor and return path."""
    save_dir = str(tmp_path / "test_predictor")
    fitted_predictor.path = save_dir
    fitted_predictor.save()
    return save_dir


@pytest.fixture
def sample_csv(sample_data, tmp_path):
    """Write sample data to CSV and return path."""
    csv_path = str(tmp_path / "train.csv")
    df = sample_data.reset_index()
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Test Save/Load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_files(self, saved_predictor_path):
        assert os.path.exists(os.path.join(saved_predictor_path, "predictor_state.pkl"))
        assert os.path.isdir(os.path.join(saved_predictor_path, "models"))

    def test_load_is_fitted(self, saved_predictor_path):
        loaded = TimeSeriesPredictor.load(saved_predictor_path)
        assert loaded._is_fitted
        assert loaded.prediction_length == 7
        assert loaded.eval_metric == "MAE"

    def test_load_preserves_scores(self, fitted_predictor, saved_predictor_path):
        loaded = TimeSeriesPredictor.load(saved_predictor_path)
        for name in fitted_predictor._model_scores:
            if name in loaded._model_scores:
                assert abs(
                    fitted_predictor._model_scores[name] - loaded._model_scores[name]
                ) < 1e-6

    def test_load_predict(self, sample_data, saved_predictor_path):
        loaded = TimeSeriesPredictor.load(saved_predictor_path)
        preds = loaded.predict(sample_data)
        assert isinstance(preds, TimeSeriesDataFrame)
        assert "mean" in preds.columns
        for item_id in sample_data.item_ids:
            assert len(preds.loc[item_id]) == 7

    def test_load_leaderboard(self, saved_predictor_path):
        loaded = TimeSeriesPredictor.load(saved_predictor_path)
        lb = loaded.leaderboard(silent=True)
        assert isinstance(lb, pd.DataFrame)
        assert len(lb) >= 2

    def test_load_best_model_matches(self, fitted_predictor, saved_predictor_path):
        loaded = TimeSeriesPredictor.load(saved_predictor_path)
        assert loaded.best_model == fitted_predictor.best_model

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            TimeSeriesPredictor.load("/does/not/exist")

    def test_save_with_context_length(self, saved_predictor_path):
        loaded = TimeSeriesPredictor.load(saved_predictor_path)
        assert loaded._context_length is not None


# ---------------------------------------------------------------------------
# Test CLI (subprocess)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_help(self, capsys):
        """Test that main() prints help with no args."""
        from myforecaster.cli import main
        import sys
        old_argv = sys.argv
        sys.argv = ["myforecaster"]
        try:
            main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert "myforecaster" in captured.out.lower() or "usage" in captured.out.lower()

    def test_cmd_info(self, capsys):
        """Test info command prints dependency table."""
        from myforecaster.cli import _cmd_info
        _cmd_info(None)
        captured = capsys.readouterr()
        assert "MyForecaster" in captured.out
        assert "numpy" in captured.out

    def test_fit_and_predict_functions(self, sample_data, tmp_path):
        """Test _cmd_fit / _cmd_predict via argparse Namespace."""
        import types
        from myforecaster.cli import _cmd_fit, _cmd_predict

        # Write CSV
        csv_path = str(tmp_path / "train.csv")
        sample_data.reset_index().to_csv(csv_path, index=False)
        out_path = str(tmp_path / "cli_pred")

        # Fit
        fit_args = types.SimpleNamespace(
            data=csv_path, prediction_length=7, presets="fast_training",
            eval_metric="MAE", output=out_path, time_limit=60,
            num_val_windows=1, refit_full=False,
            id_column="item_id", timestamp_column="timestamp",
            target_column="target",
        )
        _cmd_fit(fit_args)
        assert os.path.exists(os.path.join(out_path, "predictor_state.pkl"))

        # Predict
        pred_csv = str(tmp_path / "preds.csv")
        pred_args = types.SimpleNamespace(
            predictor=out_path, data=csv_path, output=pred_csv,
            model=None, id_column="item_id",
            timestamp_column="timestamp", target_column="target",
        )
        _cmd_predict(pred_args)
        assert os.path.exists(pred_csv)

    def test_leaderboard_function(self, saved_predictor_path, capsys):
        import types
        from myforecaster.cli import _cmd_leaderboard
        args = types.SimpleNamespace(predictor=saved_predictor_path)
        _cmd_leaderboard(args)
        captured = capsys.readouterr()
        assert "Naive" in captured.out or "model" in captured.out


# ---------------------------------------------------------------------------
# Test FastAPI Serving
# ---------------------------------------------------------------------------

class TestServing:
    def test_create_app(self, saved_predictor_path):
        from myforecaster.serving.app import create_app
        app = create_app(saved_predictor_path)
        assert app is not None
        assert app.title == "MyForecaster API"

    def test_health_endpoint(self, saved_predictor_path):
        from myforecaster.serving.app import create_app
        from httpx import ASGITransport, AsyncClient
        import asyncio

        app = create_app(saved_predictor_path)

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.get("/health")
                assert r.status_code == 200
                data = r.json()
                assert data["status"] == "ok"
                assert data["model_loaded"] is True

        asyncio.run(_test())

    def test_info_endpoint(self, saved_predictor_path):
        from myforecaster.serving.app import create_app
        from httpx import ASGITransport, AsyncClient
        import asyncio

        app = create_app(saved_predictor_path)

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.get("/info")
                assert r.status_code == 200
                data = r.json()
                assert data["prediction_length"] == 7
                assert "best_model" in data
                assert "model_names" in data

        asyncio.run(_test())

    def test_leaderboard_endpoint(self, saved_predictor_path):
        from myforecaster.serving.app import create_app
        from httpx import ASGITransport, AsyncClient
        import asyncio

        app = create_app(saved_predictor_path)

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.get("/leaderboard")
                assert r.status_code == 200
                data = r.json()
                assert isinstance(data, list)
                assert len(data) >= 2
                assert "model" in data[0]
                assert "score_val" in data[0]

        asyncio.run(_test())

    def test_predict_endpoint(self, saved_predictor_path, sample_data):
        from myforecaster.serving.app import create_app
        from httpx import ASGITransport, AsyncClient
        import asyncio, json

        app = create_app(saved_predictor_path)

        # Build request payload with plain Python types
        df = sample_data.reset_index()
        records = json.loads(df.to_json(orient="records", date_format="iso"))

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post("/predict", json={
                    "data": records,
                    "model_name": None,
                    "quantile_levels": [0.1, 0.5, 0.9],
                })
                assert r.status_code == 200, f"Body: {r.text}"
                data = r.json()
                assert "predictions" in data
                assert data["prediction_length"] == 7
                assert data["model_used"] is not None
                assert len(data["predictions"]) > 0

        asyncio.run(_test())

    def test_predict_invalid_data(self, saved_predictor_path):
        from myforecaster.serving.app import create_app
        from httpx import ASGITransport, AsyncClient
        import asyncio

        app = create_app(saved_predictor_path)

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post("/predict", json={
                    "data": [{"invalid": "data"}],
                })
                assert r.status_code in (400, 422)

        asyncio.run(_test())

    def test_score_endpoint(self, saved_predictor_path, sample_data):
        from myforecaster.serving.app import create_app
        from httpx import ASGITransport, AsyncClient
        import asyncio, json

        app = create_app(saved_predictor_path)
        df = sample_data.reset_index()
        records = json.loads(df.to_json(orient="records", date_format="iso"))

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post("/score", json={
                    "data": records,
                    "model_name": "Naive",
                })
                assert r.status_code == 200, f"Body: {r.text}"
                data = r.json()
                assert "score" in data
                assert isinstance(data["score"], float)

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Test serving module import without fastapi
# ---------------------------------------------------------------------------

class TestServingImport:
    def test_serving_init_imports(self):
        import myforecaster.serving
        assert hasattr(myforecaster.serving, "__doc__")

    def test_create_app_exists(self):
        from myforecaster.serving.app import create_app
        assert callable(create_app)


# ---------------------------------------------------------------------------
# End-to-end: fit → save → load → predict → serve
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, sample_data, tmp_path):
        """Complete lifecycle: fit → save → load → predict."""
        save_path = str(tmp_path / "e2e_predictor")

        # 1. Fit
        p = TimeSeriesPredictor(prediction_length=7, eval_metric="MAE", path=save_path)
        p.fit(
            sample_data,
            presets={
                "models": {"Naive": {}, "Average": {}},
                "ensemble": "SimpleAverage",
            },
        )
        assert p._is_fitted

        # 2. Save
        p.save()
        assert os.path.exists(os.path.join(save_path, "predictor_state.pkl"))

        # 3. Load
        loaded = TimeSeriesPredictor.load(save_path)
        assert loaded._is_fitted
        assert loaded.best_model == p.best_model

        # 4. Predict from loaded
        preds = loaded.predict(sample_data)
        assert isinstance(preds, TimeSeriesDataFrame)
        assert len(preds) > 0

        # 5. Score from loaded
        score = loaded.score(sample_data)
        assert isinstance(score, float) and score >= 0

        # 6. Leaderboard from loaded
        lb = loaded.leaderboard(silent=True)
        assert len(lb) >= 2
