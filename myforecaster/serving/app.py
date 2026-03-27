"""
REST API serving for trained TimeSeriesPredictor instances.

Provides a FastAPI app that:
- Loads a saved predictor from disk
- Exposes ``POST /predict`` endpoint for real-time forecasting
- Exposes ``GET /health``, ``GET /info``, ``GET /leaderboard`` for monitoring

Usage::

    # From Python
    from myforecaster.serving.app import create_app
    app = create_app("/path/to/saved/predictor")
    # uvicorn.run(app, host="0.0.0.0", port=8000)

    # From CLI
    myforecaster serve --predictor-path /path/to/saved/predictor --port 8000

Requires: ``pip install fastapi uvicorn``
"""

import logging
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def create_app(predictor_path: str):
    """Create a FastAPI app that serves forecasts from a saved predictor.

    Parameters
    ----------
    predictor_path : str
        Path to a saved TimeSeriesPredictor (via ``predictor.save()``).

    Returns
    -------
    FastAPI app
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI serving requires: pip install fastapi uvicorn"
        )

    import pandas as pd
    from myforecaster.predictor import TimeSeriesPredictor
    from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame

    # Load predictor
    logger.info(f"Loading predictor from {predictor_path}...")
    predictor = TimeSeriesPredictor.load(predictor_path)
    logger.info(f"Predictor loaded: {predictor}")

    app = FastAPI(
        title="MyForecaster API",
        description="Real-time time series forecasting via REST API",
        version="0.1.0",
    )

    # --- Request/Response schemas ---

    class ForecastRequest(BaseModel):
        """Request body for /predict and /score endpoints."""
        data: List[dict]
        model_name: Optional[str] = None
        quantile_levels: Optional[List[float]] = None

    class ForecastResponse(BaseModel):
        predictions: List[dict]
        model_used: str
        prediction_length: int
        elapsed_ms: float

    # --- Endpoints ---

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": predictor._is_fitted}

    @app.get("/info")
    async def info():
        summary = predictor.fit_summary()
        return {
            "prediction_length": summary["prediction_length"],
            "eval_metric": summary["eval_metric"],
            "freq": summary["freq"],
            "n_models": summary["n_models_trained"],
            "best_model": summary["best_model"],
            "best_score": summary["best_score"],
            "model_names": predictor.model_names,
        }

    @app.get("/leaderboard")
    async def leaderboard():
        lb = predictor.leaderboard(silent=True)
        return lb.to_dict(orient="records")

    @app.post("/predict", response_model=ForecastResponse)
    async def predict(request: ForecastRequest):
        t0 = time.time()

        # Parse input data
        try:
            df = pd.DataFrame(request.data)
            tsdf = TimeSeriesDataFrame.from_data_frame(df)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid input data: {e}"
            )

        # Generate forecast
        model_name = request.model_name or predictor.best_model
        try:
            preds = predictor.predict(
                tsdf,
                model=model_name,
                quantile_levels=request.quantile_levels,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {e}"
            )

        # Convert to records
        pred_records = []
        for item_id in preds.item_ids:
            item_pred = preds.loc[item_id]
            for _, row in item_pred.iterrows():
                record = {"item_id": item_id}
                record["timestamp"] = str(row.name) if hasattr(row, "name") else ""
                for col in item_pred.columns:
                    record[col] = float(row[col]) if pd.notna(row[col]) else None
                pred_records.append(record)

        elapsed = (time.time() - t0) * 1000

        return ForecastResponse(
            predictions=pred_records,
            model_used=model_name,
            prediction_length=predictor.prediction_length,
            elapsed_ms=round(elapsed, 2),
        )

    @app.post("/score")
    async def score_endpoint(request: ForecastRequest):
        """Score the predictor on provided data (context + future)."""
        try:
            df = pd.DataFrame(request.data)
            tsdf = TimeSeriesDataFrame.from_data_frame(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

        model_name = request.model_name or predictor.best_model
        try:
            score_val = predictor.score(tsdf, model=model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

        return {"model": model_name, "score": score_val}

    return app


def run_server(predictor_path: str, host: str = "0.0.0.0", port: int = 8000):
    """Start the serving process."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Serving requires: pip install uvicorn")

    app = create_app(predictor_path)
    uvicorn.run(app, host=host, port=port)
