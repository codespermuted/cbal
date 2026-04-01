# 수정 중~~~
# C-BAL (Compute BAseLine)

**내 모델링의 위대한 시발(始發)점.**

> 복잡한 파라미터 튜닝에 지쳤을 때 외쳐보세요 — *"C-BAL!"*

C-BAL은 시계열 예측을 위한 AutoML 라이브러리입니다. 한 줄의 코드로 20개 이상의 모델을 자동으로 학습하고, 최적의 앙상블을 구성합니다.

---

## Why C-BAL?

- **One-liner AutoML**: `fit()` 한 번이면 통계 모델부터 트랜스포머까지 전부 학습
- **20+ Models**: Naive ~ PatchTST, N-HiTS, DeepAR, TFT, Chronos-2 까지
- **Smart Ensemble**: Greedy forward selection으로 최적 모델 조합
- **Zero Config**: 데이터만 넣으면 프리셋이 알아서 최적화

## Quick Start

```python
from cbal import TimeSeriesPredictor

predictor = TimeSeriesPredictor(prediction_length=96, eval_metric="MAE")
predictor.fit(train_data, presets="medium_quality")
predictions = predictor.predict(test_data)
predictor.leaderboard()
```

## Installation

```bash
# Basic (stats + tabular)
pip install cbal[stats,tabular]

# With deep learning
pip install torch  # install PyTorch first
pip install cbal[stats,tabular,deep]

# Everything
pip install cbal[all]

# From source
git clone https://github.com/codespermuted/cbal.git
cd cbal
pip install -e ".[stats,tabular,deep]"
```

## Model Zoo

| Category | Models |
|----------|--------|
| **Naive** | Naive, SeasonalNaive, Average, Drift |
| **Statistical** | AutoETS, AutoARIMA, AutoTheta, AutoCES, MSTL |
| **Tabular ML** | RecursiveTabular (LightGBM/XGBoost/CatBoost), DirectTabular |
| **Deep Learning** | PatchTST, N-HiTS, DLinear, DeepAR, TFT, iTransformer, TimeMixer, TSMixer, SegRNN, TimesNet, ModernTCN, MTGNN, CrossGNN, S-Mamba, MambaTS, TiDE |
| **Foundation** | Chronos-2, TimesFM, Moirai, TTM, Toto |

## Presets

| Preset | Models | Use Case |
|--------|--------|----------|
| `light` | Stats only | Quick baseline in seconds |
| `fast_training` | Stats + simple | Real-time scoring |
| `medium_quality` | Stats + Tabular + DL + Chronos-2 | **Recommended default** |
| `high_quality` | Full model zoo | Production |
| `best_quality` | Everything + quantile variants | Maximum accuracy |
| `auto` | Data-adaptive | Smart selection |

## Features

- **Multi-window backtesting** for robust model evaluation
- **Per-item scaling** (standard, mean_abs, robust, min_max)
- **Seasonal differencing** for non-stationary data
- **Weighted ensemble** with Caruana et al. greedy selection
- **Multi-GPU support** (round-robin model distribution)
- **Early stopping** for LightGBM and all DL models
- **Conformal prediction** for calibrated uncertainty
- **HPO** with Bayesian or random search
- **FastAPI serving** for production deployment

## License

Apache 2.0

## Author

**Jaehong Yu** ([@codespermuted](https://github.com/codespermuted))
