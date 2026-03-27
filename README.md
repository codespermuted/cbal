# MyForecaster

**AutoML Time Series Forecasting Library** — inspired by AutoGluon-TimeSeries, fully independent.

Train 30+ forecasting models, automatically select the best, and deploy — all in a few lines of code.

```python
from myforecaster import TimeSeriesPredictor

predictor = TimeSeriesPredictor(prediction_length=14, eval_metric="MASE")
predictor.fit(train_data, presets="medium_quality")
predictions = predictor.predict(test_data)
predictor.leaderboard()
```

## Features

- **38 models** across 7 categories: Naive, Statistical, Tabular (LightGBM/XGBoost/CatBoost), Deep Learning (17 architectures), Foundation Models (5 pretrained), Ensemble
- **AutoML orchestration**: presets, automatic validation, ensemble selection, hyperparameter optimization
- **Multi-window backtesting** for robust model evaluation
- **Multi-layer ensemble (stacking)** for maximum accuracy
- **REST API serving** via FastAPI
- **CLI** for training, prediction, and deployment
- **Zero torch-at-import**: lazy imports keep the package lightweight

## Installation

```bash
# Core (Naive + Statistical + Tabular)
pip install -e "."

# With statistical models
pip install -e ".[stats]"

# With tree-based models
pip install -e ".[tabular]"

# With deep learning (install torch first!)
pip install torch  # or your CUDA-specific version
pip install -e ".[deep]"

# With serving
pip install -e ".[serving]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare data

```python
import pandas as pd
from myforecaster.dataset import TimeSeriesDataFrame

df = pd.read_csv("my_data.csv")  # columns: item_id, timestamp, target
data = TimeSeriesDataFrame.from_data_frame(df)
```

### 2. Train

```python
from myforecaster import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=14,
    eval_metric="MASE",
)
predictor.fit(data, presets="medium_quality")
```

### 3. Predict and evaluate

```python
predictions = predictor.predict(data)
score = predictor.score(data)
predictor.leaderboard()
```

## Presets

| Preset | Models | Typical Time |
|--------|--------|-------------|
| `fast_training` | Naive, ETS, Theta | Seconds |
| `medium_quality` | + Tabular, DLinear, PatchTST | Minutes |
| `high_quality` | + DeepAR, TFT, N-HiTS, TimeMixer | 10+ min |
| `best_quality` | All 20+ models | 30+ min |

## Model Zoo (38 models)

| Category | Models |
|----------|--------|
| Naive (5) | Naive, SeasonalNaive, Average, SeasonalAverage, Drift |
| Statistical (6) | AutoETS, AutoARIMA, AutoTheta, AutoCES, CrostonSBA, MSTL |
| Tabular (5) | RecursiveTabular, DirectTabular, LightGBM, XGBoost, CatBoost |
| Deep Learning (17) | DLinear, DeepAR, PatchTST, TFT, iTransformer, S-Mamba, MambaTS, N-HiTS, TSMixer, SegRNN, TimeMixer, TimesNet, ModernTCN, MTGNN, CrossGNN, SimpleFeedForward, TiDE |
| Foundation (5) | Chronos-2, TimesFM, Moirai, TTM, Toto |
| Ensemble (2) | WeightedEnsemble, SimpleAverage |

## Advanced Usage

### Hyperparameter override

```python
predictor.fit(
    data,
    presets="high_quality",
    hyperparameters={
        "DeepAR": {"hidden_size": 128, "max_epochs": 100},
        "PatchTST": {"d_model": 256, "n_layers": 4},
    },
    excluded_model_types=["AutoARIMA"],
)
```

### Multiple configs for the same model

```python
predictor.fit(
    data,
    hyperparameters={
        "AutoTheta": [
            {"seasonal_period": 7},
            {"seasonal_period": 12},
            {"seasonal_period": 1},
        ],
    },
)
```

### Hyperparameter optimization

```python
from myforecaster.hpo import Int, Real, Categorical

predictor.fit(
    data,
    hyperparameters={
        "PatchTST": {
            "d_model": Categorical(64, 128, 256),
            "n_layers": Int(1, 4),
            "learning_rate": Real(1e-5, 1e-3, log=True),
        },
    },
    hyperparameter_tune_kwargs="auto",  # Bayesian, 10 trials
)
```

### Multi-window backtesting

```python
predictor.fit(data, presets="high_quality", num_val_windows=3)
```

### Multi-layer ensemble (stacking)

```python
predictor.fit(
    data,
    presets="best_quality",
    num_val_windows=(2, 2),  # L1: 2 windows, L2: 2 windows
)
```

### Refit on full data after selection

```python
predictor.fit(data, presets="high_quality", refit_full=True)
```

### Save and load

```python
predictor.save()
loaded = TimeSeriesPredictor.load("myforecaster_predictor_...")
loaded.predict(new_data)
```

## REST API Serving

```python
from myforecaster.serving.app import create_app

app = create_app("/path/to/saved/predictor")
# uvicorn.run(app, host="0.0.0.0", port=8000)
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/info` | Predictor info |
| GET | `/leaderboard` | Model rankings |
| POST | `/predict` | Generate forecasts |
| POST | `/score` | Evaluate on data |

## CLI

```bash
myforecaster info
myforecaster fit --data train.csv --prediction-length 14 --presets medium_quality --output my_predictor
myforecaster predict --predictor my_predictor --data test.csv --output predictions.csv
myforecaster leaderboard --predictor my_predictor
myforecaster serve --predictor my_predictor --port 8000
```

## Metrics

| Metric | Description |
|--------|-------------|
| `MAE` | Mean Absolute Error |
| `RMSE` | Root Mean Squared Error |
| `MAPE` | Mean Absolute Percentage Error |
| `sMAPE` | Symmetric MAPE |
| `MASE` | Mean Absolute Scaled Error (default) |
| `WAPE` | Weighted Absolute Percentage Error |
| `RMSSE` | Root Mean Squared Scaled Error |
| `WQL` | Weighted Quantile Loss |
| `SQL` | Scaled Quantile Loss |
| `Coverage` | Prediction Interval Coverage |

## Architecture

```
myforecaster/
├── __init__.py              # Lazy imports
├── predictor.py             # TimeSeriesPredictor (AutoML orchestrator)
├── cli.py                   # Command-line interface
├── dataset/
│   └── ts_dataframe.py      # TimeSeriesDataFrame
├── metrics/
│   └── scorers.py           # 10 evaluation metrics
├── hpo/
│   ├── space.py             # Int, Real, Categorical
│   ├── searcher.py          # Random, Bayesian (Optuna)
│   └── runner.py            # HPO orchestration
├── models/
│   ├── abstract_model.py    # Base class
│   ├── ensemble.py          # WeightedEnsemble + SimpleAverage
│   ├── naive/               # 5 naive models
│   ├── statsforecast/       # StatsForecast wrappers
│   ├── tabular/             # LightGBM/XGBoost/CatBoost
│   ├── deep_learning/       # 15 DL architectures
│   └── foundation/          # Chronos-2, TimesFM, Moirai, TTM, Toto
└── serving/
    └── app.py               # FastAPI REST API
```

## Testing

```bash
# Non-DL tests (no torch required)
MYFORECASTER_SKIP_TORCH=1 pytest tests/ -v

# All tests (GPU server)
pytest tests/ -v

# Specific steps
pytest tests/test_step10_integration.py -v  # E2E integration
```

## Covariate System (AutoGluon-aligned)

MyForecaster supports the full covariate pipeline, matching AutoGluon's architecture:

### Static features

```python
static = pd.DataFrame({"category": ["food", "electronics"], "store_size": [100, 200]},
                       index=["item_0", "item_1"])
data.static_features = static
```

### Known and past covariates

```python
# Data with covariate columns
data = TimeSeriesDataFrame.from_data_frame(df)  # df has columns: target, promotion, temperature
data.known_covariates_names = ["promotion"]       # known in advance
data.past_covariates_names = ["temperature"]       # only known up to present
```

### target_scaler (all models)

Scale targets per-item before training, inverse-scale predictions automatically:

```python
predictor.fit(data, presets={
    "models": {"DeepAR": {}, "PatchTST": {}},
    "target_scaler": "standard",    # "standard", "mean_abs", "robust", "min_max"
})
```

### covariate_regressor (any model → covariate-aware)

Train a LightGBM regressor on covariates, subtract effect, then add back to predictions. Makes even univariate models like Naive covariate-aware:

```python
predictor = TimeSeriesPredictor(
    prediction_length=14,
    known_covariates_names=["promotion", "is_weekend"],
)
predictor.fit(data, presets={
    "models": {"Naive": {}, "PatchTST": {}},
    "covariate_regressor": True,       # or "lightgbm" / "linear"
    "target_scaler": "standard",
})
```

## Remaining Differences vs AutoGluon

| Feature | Status | Note |
|---------|--------|------|
| All API features | ✅ Complete | evaluate(), feature_importance(), all fit() params |
| Covariate system | ✅ Complete | static_features, known/past covariates, target_scaler, covariate_regressor, covariate_scaler |
| Metrics | ✅ Complete | MAE, RMSE, MAPE, sMAPE, MASE, RMSSE, WQL, SQL, WAPE, Coverage + horizon_weight |
| WaveNet model | Not planned | Superseded by ModernTCN (same CNN family, better results) |
| Chronos-2 fine-tuning | Not yet | Zero-shot only (fine-tuning requires chronos-forecasting library internals) |

### Intentional design differences (not gaps)

| Item | AutoGluon | MyForecaster | Rationale |
|------|-----------|-------------|-----------|
| Default eval_metric | WQL | MASE | Point-forecast-centric workflows |
| Default quantile_levels | 9 levels | 3 levels | Faster; user can override |
| DL implementation | GluonTS wrapper | Independent from papers | No GluonTS/MXNet dependency |
| Tabular backend | AG-Tabular wrapper | Independent LightGBM/XGB | No AG-Tabular dependency |

## License

Apache 2.0
