"""
TimeSeriesPredictor — AutoML orchestrator for time series forecasting.

Aligned with AutoGluon-TimeSeries architecture including:
- Multi-window backtesting (num_val_windows)
- Hyperparameter list values (same model, multiple configs)
- Automatic context_length determination
- refit_full (retrain on full data after model selection)
- Prediction caching
- random_seed for reproducibility
- HPO "auto" shortcut
- enable_ensemble parameter
- Multi-layer ensemble (stacking)

Usage::

    from cbal import TimeSeriesPredictor

    predictor = TimeSeriesPredictor(prediction_length=14, eval_metric="MASE")
    predictor.fit(train_data, presets="medium_quality")
    predictions = predictor.predict(test_data)
    leaderboard = predictor.leaderboard()
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cbal.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame
from cbal.metrics.scorers import TimeSeriesScorer, get_metric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frequency → seasonal period mapping (for auto context_length)
# ---------------------------------------------------------------------------
_FREQ_TO_SEASON = {
    "MS": 12, "ME": 12, "QS": 4, "QE": 4, "YS": 1, "YE": 1,
    "min": 60, "h": 24, "T": 60, "S": 60, "D": 7, "B": 5,
    "W": 52, "M": 12, "Q": 4, "Y": 1, "A": 1,
}


def _infer_seasonal_period(freq: str | None) -> int:
    """Best-effort seasonal period from frequency string."""
    if freq is None:
        return 1
    for key, period in _FREQ_TO_SEASON.items():
        if key in freq:
            return period
    return 1


def _auto_context_length(
    prediction_length: int,
    freq: str | None,
    max_ts_length: int | None = None,
) -> int:
    """Determine context_length automatically (AutoGluon-style).

    Improved heuristic:
    - At least 2 full seasonal cycles (captures repeating patterns)
    - At least 3x prediction_length (gives model enough history)
    - Capped at available data minus prediction_length

    AutoGluon uses max(prediction_length * 2, seasonal_period * 2) as
    a minimum; we use 3x prediction_length for slightly longer context
    which helps DL models like PatchTST.
    """
    # AG-style: min(512, max(10, 2 * prediction_length))
    sp = _infer_seasonal_period(freq)
    ctx = min(512, max(10, 2 * prediction_length, 2 * sp))
    if max_ts_length is not None:
        ctx = min(ctx, max_ts_length - prediction_length)
    return max(ctx, prediction_length)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESETS = {
    # ── 가벼운 시작: 통계 모델만, 수 초 이내 ──
    "light": {
        "models": {
            "Naive": {},
            "SeasonalNaive": {},
            "AutoETS": {},
        },
        "ensemble": "SimpleAverage",
        "time_limit_per_model": 15,
    },
    # ── 빠른 훈련: 통계 + 간단 모델, 1분 이내 ──
    "fast_training": {
        "models": {
            "Naive": {},
            "SeasonalNaive": {},
            "Average": {},
            "Drift": {},
            "AutoETS": {},
            "AutoTheta": {},
        },
        "ensemble": "SimpleAverage",
        "time_limit_per_model": 30,
    },
    # ── 적당한 성능: 통계 + Tabular + 가벼운 DL ──
    "medium_quality": {
        "models": {
            "Naive": {},
            "SeasonalNaive": {},
            "AutoETS": {},
            "AutoTheta": {},
            "RecursiveTabular": {"backend": "LightGBM"},
            "DirectTabular": {"backend": "LightGBM"},
            "DLinear": {"max_epochs": 50, "context_length": None,
                        "learning_rate": 1e-3, "patience": 10},
            "PatchTST": {"max_epochs": 50, "d_model": 128, "n_layers": 3,
                         "learning_rate": 1e-3, "patience": 20,
                         "batch_size": 64},
            "N-HiTS": {"max_epochs": 50, "hidden_size": 256,
                       "learning_rate": 1e-3, "patience": 20},
            "DeepAR": {"max_epochs": 50, "hidden_size": 40,
                       "learning_rate": 1e-3, "patience": 20,
                       "batch_size": 64},
            "TFT": {"max_epochs": 50, "d_model": 64, "patience": 20,
                     "batch_size": 64, "learning_rate": 1e-3},
            # Quantile-trained variant for better PI and ensemble diversity
            "DLinear_Q": {"_base_model_name": "DLinear", "max_epochs": 50,
                          "loss_type": "quantile", "learning_rate": 1e-3},
            "PatchTST_Q": {"_base_model_name": "PatchTST", "max_epochs": 50,
                           "d_model": 128, "n_layers": 3, "learning_rate": 1e-3,
                           "loss_type": "quantile", "batch_size": 64},
            # Foundation model — zero-shot, strong on diverse datasets
            "Chronos-2": {},
        },
        "ensemble": "WeightedEnsemble",
        "time_limit_per_model": 180,
    },
    # ── 좋은 성능: 통계 + Tabular + 다양한 DL ──
    "high_quality": {
        "models": {
            "Naive": {},
            "SeasonalNaive": {},
            "AutoETS": {},
            "AutoTheta": {},
            "AutoARIMA": {},
            "RecursiveTabular": {"backend": "LightGBM"},
            "DirectTabular": {"backend": "LightGBM"},
            "DLinear": {"max_epochs": 100, "learning_rate": 1e-3, "patience": 15},
            "DeepAR": {"max_epochs": 100, "hidden_size": 64, "learning_rate": 1e-3,
                       "patience": 20},
            "PatchTST": {"max_epochs": 100, "d_model": 128, "n_layers": 3,
                         "learning_rate": 1e-4, "patience": 15, "batch_size": 64},
            "TFT": {"max_epochs": 100, "d_model": 64, "patience": 15},
            "N-HiTS": {"max_epochs": 100, "hidden_size": 256, "n_stacks": 3,
                       "learning_rate": 5e-4, "patience": 15},
            "TimeMixer": {"max_epochs": 100, "d_model": 64, "n_scales": 4,
                          "patience": 15},
            # Quantile-trained variants for ensemble diversity
            "PatchTST_Q": {"_base_model_name": "PatchTST", "max_epochs": 100,
                           "d_model": 128, "n_layers": 3, "learning_rate": 1e-4,
                           "loss_type": "quantile", "patience": 15},
            "N-HiTS_Q": {"_base_model_name": "N-HiTS", "max_epochs": 100,
                          "hidden_size": 256, "learning_rate": 5e-4,
                          "loss_type": "quantile", "patience": 15},
        },
        "ensemble": "WeightedEnsemble",
        "time_limit_per_model": 300,
    },
    # ── 최고의 성능: 모든 모델 풀 가동 + quantile 다양성 ──
    "best_quality": {
        "models": {
            "Naive": {},
            "SeasonalNaive": {},
            "Average": {},
            "Drift": {},
            "AutoETS": {},
            "AutoTheta": {},
            "AutoARIMA": {},
            "AutoCES": {},
            "RecursiveTabular": {"backend": "LightGBM"},
            "DirectTabular": {"backend": "LightGBM"},
            "DLinear": {"max_epochs": 100},
            "DeepAR": {"max_epochs": 100, "hidden_size": 64, "num_layers": 2,
                        "learning_rate": 1e-3},
            "PatchTST": {"max_epochs": 50, "d_model": 128, "n_layers": 3,
                         "learning_rate": 1e-4},
            "TFT": {"max_epochs": 50, "d_model": 64, "n_lstm_layers": 2},
            "iTransformer": {"max_epochs": 50, "d_model": 256, "learning_rate": 1e-4},
            "N-HiTS": {"max_epochs": 50, "hidden_size": 256, "n_stacks": 3,
                       "learning_rate": 5e-4},
            "TSMixer": {"max_epochs": 50, "d_ff": 64, "n_layers": 4},
            "TimeMixer": {"max_epochs": 50, "d_model": 64, "n_scales": 4},
            "SegRNN": {"max_epochs": 50, "d_model": 128},
            "ModernTCN": {"max_epochs": 50, "d_model": 128},
            # Quantile-trained variants for richer ensemble
            "DLinear_Q": {"_base_model_name": "DLinear", "max_epochs": 100,
                          "loss_type": "quantile"},
            "PatchTST_Q": {"_base_model_name": "PatchTST", "max_epochs": 50,
                           "d_model": 128, "n_layers": 3, "learning_rate": 1e-4,
                           "loss_type": "quantile"},
            "N-HiTS_Q": {"_base_model_name": "N-HiTS", "max_epochs": 50,
                          "hidden_size": 256, "learning_rate": 5e-4,
                          "loss_type": "quantile"},
            "Chronos-2": {},  # Foundation model (zero-shot)
        },
        "ensemble": "WeightedEnsemble",
        "time_limit_per_model": 600,
    },
}

# Aliases for user convenience
_PRESETS["good_quality"] = _PRESETS["high_quality"]


# ---------------------------------------------------------------------------
# Model priority tiers — used by "auto" preset and time-budget scheduling.
# Lower tier number = higher priority (trained first, dropped last).
# Within a tier, models are ordered by estimated cost (cheapest first).
# ---------------------------------------------------------------------------

_MODEL_PRIORITY = [
    # Tier 0 — baseline (near-instant)
    {"tier": 0, "name": "Naive",         "cost": "negligible", "needs": None},
    {"tier": 0, "name": "SeasonalNaive", "cost": "negligible", "needs": None},
    # Tier 1 — fast statistical
    {"tier": 1, "name": "AutoETS",       "cost": "low",  "needs": "stats"},
    {"tier": 1, "name": "AutoTheta",     "cost": "low",  "needs": "stats"},
    {"tier": 1, "name": "AutoARIMA",     "cost": "low",  "needs": "stats"},
    {"tier": 1, "name": "AutoCES",       "cost": "low",  "needs": "stats"},
    # Tier 2 — tabular (good accuracy-to-cost ratio)
    {"tier": 2, "name": "RecursiveTabular", "cost": "medium", "needs": "tabular",
     "hp": {"backend": "LightGBM"}},
    {"tier": 2, "name": "DirectTabular",    "cost": "medium", "needs": "tabular",
     "hp": {"backend": "LightGBM"}},
    # Tier 3 — lightweight DL
    {"tier": 3, "name": "DLinear",   "cost": "medium", "needs": "torch",
     "hp": {"max_epochs": 30}},
    {"tier": 3, "name": "PatchTST",  "cost": "medium", "needs": "torch",
     "hp": {"max_epochs": 25, "d_model": 64, "n_layers": 2}},
    {"tier": 3, "name": "N-HiTS",    "cost": "medium", "needs": "torch",
     "hp": {"max_epochs": 25, "hidden_size": 128}},
    # Tier 4 — heavier DL
    {"tier": 4, "name": "DeepAR",       "cost": "high", "needs": "torch",
     "hp": {"max_epochs": 50, "hidden_size": 64}},
    {"tier": 4, "name": "TFT",          "cost": "high", "needs": "torch",
     "hp": {"max_epochs": 30, "d_model": 64}},
    {"tier": 4, "name": "TimeMixer",    "cost": "high", "needs": "torch",
     "hp": {"max_epochs": 30, "d_model": 64, "n_scales": 4}},
    {"tier": 4, "name": "iTransformer", "cost": "high", "needs": "torch",
     "hp": {"max_epochs": 30, "d_model": 128}},
    # Tier 5 — expensive DL
    {"tier": 5, "name": "TSMixer",    "cost": "high",   "needs": "torch",
     "hp": {"max_epochs": 40, "d_ff": 64, "n_layers": 4}},
    {"tier": 5, "name": "SegRNN",     "cost": "high",   "needs": "torch",
     "hp": {"max_epochs": 40, "d_model": 128}},
    {"tier": 5, "name": "ModernTCN",  "cost": "high",   "needs": "torch",
     "hp": {"max_epochs": 40, "d_model": 128}},
    {"tier": 5, "name": "TimesNet",   "cost": "high",   "needs": "torch",
     "hp": {"max_epochs": 40, "d_model": 64}},
    {"tier": 5, "name": "MTGNN",      "cost": "high",   "needs": "torch",
     "hp": {"max_epochs": 40, "d_model": 128}},
    {"tier": 5, "name": "CrossGNN",   "cost": "high",   "needs": "torch",
     "hp": {"max_epochs": 40}},
    {"tier": 5, "name": "TiDE",       "cost": "high",   "needs": "torch",
     "hp": {"max_epochs": 40}},
    {"tier": 5, "name": "S-Mamba",    "cost": "very_high", "needs": "ssm",
     "hp": {"max_epochs": 40}},
    {"tier": 5, "name": "MambaTS",    "cost": "very_high", "needs": "ssm",
     "hp": {"max_epochs": 40}},
]

# Estimated cost multipliers (seconds per 1000-row item × model)
_COST_MULTIPLIER = {
    "negligible": 0.01,
    "low": 0.5,
    "medium": 2.0,
    "high": 8.0,
    "very_high": 15.0,
}


def _profile_data(
    data: "TimeSeriesDataFrame",
    prediction_length: int,
    known_covariates_names: list[str] | None = None,
) -> dict:
    """Profile a dataset to inform adaptive preset selection.

    Returns a dict with:
        num_items, min_length, median_length, max_length,
        total_rows, num_known_covariates, num_past_covariates,
        has_static_features, num_features (total external),
        size_category ("tiny" | "small" | "medium" | "large" | "xlarge"),
        feature_category ("univariate" | "low_feature" | "rich_feature").
    """
    lengths = [len(data.loc[iid]) for iid in data.item_ids]
    num_items = data.num_items
    min_len = min(lengths)
    median_len = int(np.median(lengths))
    max_len = max(lengths)
    total_rows = sum(lengths)

    n_known = len(known_covariates_names) if known_covariates_names else 0
    n_past = len(data.past_covariates_names) if hasattr(data, "past_covariates_names") and data.past_covariates_names else 0
    has_static = data.static_features is not None and len(data.static_features.columns) > 0
    n_static = len(data.static_features.columns) if has_static else 0
    num_features = n_known + n_past + n_static

    # Size categories — based on total rows AND median series length.
    # Few items with long series (e.g. 2 items × 90k rows) should still
    # be classified as "large", not "tiny".
    if total_rows < 500 and median_len < 100:
        size_cat = "tiny"
    elif total_rows < 5_000 and median_len < 500:
        size_cat = "small"
    elif total_rows < 100_000 and median_len < 5_000:
        size_cat = "medium"
    elif total_rows < 1_000_000:
        size_cat = "large"
    else:
        size_cat = "xlarge"

    # Feature categories
    if num_features == 0:
        feat_cat = "univariate"
    elif num_features <= 5:
        feat_cat = "low_feature"
    else:
        feat_cat = "rich_feature"

    return {
        "num_items": num_items,
        "min_length": min_len,
        "median_length": median_len,
        "max_length": max_len,
        "total_rows": total_rows,
        "num_known_covariates": n_known,
        "num_past_covariates": n_past,
        "has_static_features": has_static,
        "num_static_features": n_static,
        "num_features": num_features,
        "size_category": size_cat,
        "feature_category": feat_cat,
    }


def _build_auto_preset(profile: dict, prediction_length: int) -> dict:
    """Build a preset config dynamically based on data profile.

    Strategy:
    - tiny data    → stats only (DL will overfit)
    - small data   → stats + tabular (DL only if features justify it)
    - medium data  → stats + tabular + lightweight DL
    - large data   → stats + tabular + diverse DL
    - xlarge data  → focus on scalable models (tabular + efficient DL)

    - rich features → prioritize TFT, Tabular, DeepAR (covariate-aware)
    - univariate   → prioritize PatchTST, N-HiTS, DLinear
    """
    size = profile["size_category"]
    feat = profile["feature_category"]

    # Determine max tier to include
    if size == "tiny":
        max_tier = 1  # stats only
    elif size == "small":
        max_tier = 2  # + tabular
        if feat == "rich_feature":
            max_tier = 3  # + lightweight DL for covariates
    elif size == "medium":
        max_tier = 3  # + lightweight DL
        if feat != "univariate":
            max_tier = 4  # + heavier DL for features
    elif size == "large":
        max_tier = 4
    else:  # xlarge
        max_tier = 4  # skip tier 5 (too expensive at scale)

    models = {}
    for entry in _MODEL_PRIORITY:
        if entry["tier"] > max_tier:
            continue
        hp = dict(entry.get("hp", {}))

        # Feature-aware adjustments
        if feat == "rich_feature":
            if entry["name"] in ("TFT", "DeepAR", "iTransformer"):
                if entry["tier"] <= max_tier + 1:
                    pass
            if entry["name"] in ("Naive", "Average", "Drift") and max_tier >= 2:
                continue

        # For few items with long series: skip slow stats, focus on tabular+DL
        few_long = profile["num_items"] <= 10 and profile["median_length"] > 500
        if few_long and entry["name"] in ("AutoARIMA",) and entry["tier"] >= 1:
            # AutoARIMA is slow and often worse than Tabular on long series
            continue

        # Size-aware epoch adjustments
        if size in ("tiny", "small"):
            if "max_epochs" in hp:
                hp["max_epochs"] = min(hp["max_epochs"], 20)
        elif size == "xlarge":
            if "max_epochs" in hp:
                hp["max_epochs"] = min(hp["max_epochs"], 30)
            hp.setdefault("batch_size", 128)
        elif few_long and "max_epochs" in hp:
            # Boost DL for long single-series data
            hp["max_epochs"] = max(hp["max_epochs"], 50)

        models[entry["name"]] = hp

    # For rich features, ensure covariate-aware models are included
    if feat == "rich_feature" and size not in ("tiny",):
        for name, hp in [
            ("TFT", {"max_epochs": 30, "d_model": 64}),
            ("DeepAR", {"max_epochs": 30, "hidden_size": 64}),
        ]:
            if name not in models:
                models[name] = hp

    # Time limit per model — adaptive
    time_limits = {
        "tiny": 15, "small": 60, "medium": 120, "large": 300, "xlarge": 300,
    }

    config = {
        "models": models,
        "ensemble": "WeightedEnsemble" if len(models) >= 3 else "SimpleAverage",
        "time_limit_per_model": time_limits[size],
    }

    logger.info(
        f"Auto preset: size={size}, features={feat}, "
        f"{len(models)} models (max_tier={max_tier})"
    )
    return config


def _has_gpu() -> bool:
    """Check if CUDA GPU is available (cached)."""
    if not hasattr(_has_gpu, "_cache"):
        try:
            import torch
            _has_gpu._cache = torch.cuda.is_available()
        except ImportError:
            _has_gpu._cache = False
    return _has_gpu._cache


def _estimate_model_time(
    model_name: str,
    cost_level: str,
    profile: dict,
    hp: dict,
) -> float:
    """Rough estimate of training time in seconds for a single model."""
    base = _COST_MULTIPLIER.get(cost_level, 5.0)
    rows_factor = profile["total_rows"] / 1000.0
    epochs = hp.get("max_epochs", 30)
    epoch_factor = epochs / 30.0

    # Statistical / naive: scale by items × series length (they fit per-item)
    if cost_level in ("negligible", "low"):
        median_len_factor = profile["median_length"] / 1000.0
        return base * profile["num_items"] * max(median_len_factor, 0.5)

    est = base * rows_factor * epoch_factor

    # GPU discount: DL models are ~10x faster on GPU
    if _has_gpu() and cost_level in ("medium", "high", "very_high"):
        est *= 0.1

    return est


def _schedule_models_by_budget(
    config: dict,
    time_limit: float,
    profile: dict,
) -> dict:
    """Given a total time budget, prioritize models and drop low-priority
    ones that won't fit within the budget.

    Models are sorted by priority tier (lowest first = highest priority).
    We greedily add models until the estimated cumulative time exceeds
    the budget (keeping a 20% reserve for ensemble + overhead).
    """
    usable_budget = time_limit * 0.80  # 80% for models, 20% for ensemble/overhead

    # Build (name, tier, estimated_time, hp) list
    priority_lookup = {e["name"]: e for e in _MODEL_PRIORITY}
    scheduled = []
    for name, hp in config["models"].items():
        entry = priority_lookup.get(name)
        tier = entry["tier"] if entry else 99
        cost = entry["cost"] if entry else "high"
        est_time = _estimate_model_time(name, cost, profile, hp)
        scheduled.append((name, tier, est_time, hp))

    # Sort by tier (primary), then estimated time (secondary)
    scheduled.sort(key=lambda x: (x[1], x[2]))

    # Greedily select
    selected = {}
    cumulative = 0.0
    skipped = []
    for name, tier, est_time, hp in scheduled:
        if cumulative + est_time <= usable_budget:
            selected[name] = hp
            cumulative += est_time
        else:
            skipped.append(name)

    if skipped:
        logger.info(
            f"Time budget {time_limit:.0f}s: scheduled {len(selected)} models "
            f"(est. {cumulative:.0f}s), skipped {skipped}"
        )
    else:
        logger.info(
            f"Time budget {time_limit:.0f}s: all {len(selected)} models fit "
            f"(est. {cumulative:.0f}s)"
        )

    # Per-model time limit = proportional share of remaining budget
    n = max(len(selected), 1)
    per_model = max(usable_budget / n, 10)

    config["models"] = selected
    config["time_limit_per_model"] = per_model
    return config


def _resolve_preset(presets: str | dict | None) -> dict:
    if presets is None:
        presets = "medium_quality"
    if isinstance(presets, str):
        if presets == "auto":
            # "auto" is resolved later in fit() after data profiling
            return {"_auto": True, "models": {}}
        if presets not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{presets}'. "
                f"Choose from: {list(_PRESETS.keys()) + ['auto']}"
            )
        return copy.deepcopy(_PRESETS[presets])
    return copy.deepcopy(presets)


def _adapt_hyperparameters_to_data(
    config: dict,
    min_ts_length: int,
    num_items: int,
    prediction_length: int,
    freq: str | None,
) -> dict:
    """Adapt model hyperparameters based on data size.

    Adjusts DL epochs, batch size, tree estimators, etc. so that
    models can train successfully regardless of dataset size.
    """
    # ── DL models: adapt context_length, epochs, batch_size ──
    # Cap context_length so training windows can be created
    max_ctx = max(min_ts_length - prediction_length - 1, prediction_length)
    is_small = min_ts_length < 100
    is_tiny = min_ts_length < 50
    is_large = num_items > 500 or min_ts_length > 1000

    # Key insight from AutoGluon: for many short series, per-item stats
    # models dominate. Global DL models struggle with heterogeneous short
    # series. Skip DL if median series length < 5x prediction_length.
    median_len = config.get("_median_length", min_ts_length)
    # Adaptive DL policy based on data characteristics:
    # - Many short series → stats dominate, skip DL
    # - Few long series → DL can excel, boost epochs
    # NOTE: AG does NOT skip DL for many short series — it uses Chronos-2, TFT, etc.
    # Only skip DL if series are truly too short to create even 1 training window
    dl_is_futile = (
        median_len < 3 * prediction_length + 10  # too short for any DL window
    )
    dl_is_promising = (
        median_len > 500 or  # long series
        (num_items <= 10 and median_len > 200)  # few items, enough data
    )

    _DL_MODELS = {
        "DLinear", "DeepAR", "PatchTST", "TFT", "iTransformer",
        "N-HiTS", "TSMixer", "SegRNN", "TimeMixer", "TimesNet",
        "ModernTCN", "S-Mamba", "MambaTS", "MTGNN", "CrossGNN",
        "SimpleFeedForward", "TiDE",
    }

    # DL needs at least 1 training window: context_length + prediction_length <= train_length
    # After val split, train length ≈ min_ts_length - prediction_length
    # So: context_length <= (min_ts_length - prediction_length) - prediction_length - 1
    #   = min_ts_length - 2 * prediction_length - 1
    # We need at least 2 windows for training, so subtract one more prediction_length
    available_after_split = min_ts_length - prediction_length
    max_dl_ctx = max(available_after_split - prediction_length - 2, 2)

    for model_name, hp in config["models"].items():
        base_name = hp.get("_base_model_name", model_name)

        if base_name in _DL_MODELS:
            # Skip DL entirely for many short series (stats dominate)
            if dl_is_futile:
                logger.info(
                    f"Skipping {model_name}: DL unlikely to help "
                    f"(median_len={median_len}, pred_len={prediction_length}, "
                    f"n_items={num_items})"
                )
                hp["_skip"] = True
                continue

            # context_length: never exceed what data allows after val split
            ctx = hp.get("context_length")
            if ctx is None or ctx > max_dl_ctx:
                hp["context_length"] = max(min(max_dl_ctx, 3 * prediction_length), 2)

            # If even minimum context can't produce windows, skip DL model
            if hp["context_length"] + prediction_length > available_after_split:
                logger.info(
                    f"Skipping {model_name}: data too short "
                    f"(need {hp['context_length']+prediction_length}, have {available_after_split})"
                )
                hp["_skip"] = True

            # --- Data-adaptive HP per model (based on research papers) ---
            many_short = num_items > 50 and median_len < 500
            few_long = num_items <= 10 and median_len > 500

            # Epochs and patience
            if is_tiny:
                hp["max_epochs"] = min(hp.get("max_epochs", 100), 20)
                hp["batch_size"] = min(hp.get("batch_size", 64), max(4, num_items))
            elif many_short:
                # Many short series: less overfitting risk, moderate epochs
                hp["max_epochs"] = min(hp.get("max_epochs", 100), 50)
                hp.setdefault("batch_size", 128)
                hp.setdefault("patience", 10)
            elif few_long:
                # Few long series: DL can benefit from more epochs
                hp["max_epochs"] = max(hp.get("max_epochs", 100), 100)
                hp.setdefault("patience", 20)
            elif is_large:
                hp.setdefault("batch_size", 128)

            # Model-specific adaptive HP
            if base_name == "PatchTST":
                # PatchTST paper: d_model scales with complexity
                if many_short:
                    hp.setdefault("d_model", 64)
                    hp.setdefault("n_layers", 2)
                else:
                    hp.setdefault("d_model", 128)
                    hp.setdefault("n_layers", 3)

            elif base_name == "N-HiTS":
                # N-HiTS paper: hidden_size=512 for ETT, 256 for others
                if few_long:
                    hp.setdefault("hidden_size", 512)
                else:
                    hp.setdefault("hidden_size", 256)

            elif base_name == "TFT":
                # TFT paper: d_model=32-64 depending on data size
                if many_short:
                    hp.setdefault("d_model", 32)
                else:
                    hp.setdefault("d_model", 64)

            elif base_name == "DeepAR":
                # DeepAR: AG uses hidden_size=40 as default
                hp.setdefault("hidden_size", 40)

        # ── Tabular models: adapt n_estimators and skip if too short ──
        if base_name in ("RecursiveTabular", "DirectTabular"):
            if is_tiny:
                hp["n_estimators"] = min(hp.get("n_estimators", 200), 50)
                # Tabular needs enough rows for lag features; skip if too short
                if available_after_split < 15:
                    hp["_skip"] = True
            elif is_small:
                hp["n_estimators"] = min(hp.get("n_estimators", 200), 100)

    return config


# ---------------------------------------------------------------------------
# HPO "auto" shortcut
# ---------------------------------------------------------------------------

def _resolve_hpo_kwargs(kwargs) -> dict | None:
    """Resolve hyperparameter_tune_kwargs (string or dict or None)."""
    if kwargs is None:
        return None
    if isinstance(kwargs, str):
        if kwargs == "auto":
            return {"num_trials": 10, "searcher": "bayes", "scheduler": "local"}
        if kwargs == "random":
            return {"num_trials": 10, "searcher": "random", "scheduler": "local"}
        raise ValueError(f"Unknown HPO shortcut '{kwargs}'. Use 'auto', 'random', or a dict.")
    return dict(kwargs)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _create_model(model_name: str, freq: str, prediction_length: int,
                  hyperparameters: dict, eval_metric: str):
    """Create and return an unfitted model instance by registry name.

    Model names can omit or include the ``Model`` suffix — both
    ``"DeepAR"`` and ``"DeepARModel"`` resolve to the same class
    (same convention as AutoGluon).
    """
    # Strip "Model" suffix for lookup (AG compatibility)
    if model_name.endswith("Model") and model_name != "Model":
        model_name = model_name[:-5]

    from cbal.models.naive.models import (
        NaiveModel, SeasonalNaiveModel, AverageModel,
        SeasonalAverageModel, DriftModel,
    )
    _naive_map = {
        "Naive": NaiveModel, "SeasonalNaive": SeasonalNaiveModel,
        "Average": AverageModel, "SeasonalAverage": SeasonalAverageModel,
        "Drift": DriftModel,
    }
    if model_name in _naive_map:
        return _naive_map[model_name](
            freq=freq, prediction_length=prediction_length,
            eval_metric=eval_metric, hyperparameters=hyperparameters,
        )

    _stats_names = {"AutoETS", "AutoARIMA", "AutoTheta", "AutoCES",
                    "CrostonSBA", "MSTL"}
    if model_name in _stats_names:
        try:
            from cbal.models.statsforecast.models import StatsForecastModel
            return StatsForecastModel(
                freq=freq, prediction_length=prediction_length,
                eval_metric=eval_metric,
                hyperparameters={"model_name": model_name, **hyperparameters},
            )
        except ImportError:
            logger.warning(f"Skipping {model_name}: statsforecast not installed")
            return None

    if model_name in ("RecursiveTabular", "DirectTabular"):
        try:
            from cbal.models.tabular.models import (
                RecursiveTabularModel, DirectTabularModel,
            )
            cls = RecursiveTabularModel if model_name == "RecursiveTabular" else DirectTabularModel
            return cls(
                freq=freq, prediction_length=prediction_length,
                eval_metric=eval_metric, hyperparameters=hyperparameters,
            )
        except ImportError:
            logger.warning(f"Skipping {model_name}: tabular deps not installed")
            return None

    _dl_names = {
        "DLinear", "DeepAR", "PatchTST", "TFT", "iTransformer",
        "S-Mamba", "MambaTS", "N-HiTS", "TSMixer", "SegRNN",
        "TimeMixer", "TimesNet", "ModernTCN", "MTGNN", "CrossGNN",
        "SimpleFeedForward", "TiDE",
    }
    if model_name in _dl_names:
        try:
            import importlib
            _name_to_class = {
                "DLinear": "DLinearModel", "DeepAR": "DeepARModel",
                "PatchTST": "PatchTSTModel", "TFT": "TFTModel",
                "iTransformer": "iTransformerModel",
                "S-Mamba": "SMambaModel", "MambaTS": "MambaTSModel",
                "N-HiTS": "NHiTSModel", "TSMixer": "TSMixerModel",
                "SegRNN": "SegRNNModel", "TimeMixer": "TimeMixerModel",
                "TimesNet": "TimesNetModel", "ModernTCN": "ModernTCNModel",
                "MTGNN": "MTGNNModel", "CrossGNN": "CrossGNNModel",
                "SimpleFeedForward": "SimpleFeedForwardModel",
                "TiDE": "TiDEModel",
            }
            class_name = _name_to_class[model_name]
            mod = importlib.import_module("cbal.models.deep_learning")
            ModelClass = getattr(mod, class_name)
            return ModelClass(
                freq=freq, prediction_length=prediction_length,
                eval_metric=eval_metric, hyperparameters=hyperparameters,
            )
        except (ImportError, AttributeError) as e:
            logger.warning(f"Skipping {model_name}: {e}")
            return None

    _foundation_names = {"Chronos-2", "TimesFM", "Moirai", "TTM", "Toto"}
    if model_name in _foundation_names:
        try:
            from cbal.models import foundation
            _fm_map = {
                "Chronos-2": foundation.Chronos2Model,
                "TimesFM": foundation.TimesFMModel,
                "Moirai": foundation.MoiraiModel,
                "TTM": foundation.TTMModel,
                "Toto": foundation.TotoModel,
            }
            return _fm_map[model_name](
                freq=freq, prediction_length=prediction_length,
                eval_metric=eval_metric, hyperparameters=hyperparameters,
            )
        except ImportError as e:
            logger.warning(f"Skipping {model_name}: {e}")
            return None

    logger.warning(f"Unknown model '{model_name}', skipping.")
    return None


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

class _PredictionCache:
    """Simple in-memory LRU-ish cache for predictions."""

    def __init__(self, enabled: bool = True, max_size: int = 128):
        self._enabled = enabled
        self._max_size = max_size
        self._cache: dict[str, TimeSeriesDataFrame] = {}

    @staticmethod
    def _key(model_name: str, data: TimeSeriesDataFrame) -> str:
        raw = f"{model_name}:{len(data)}:{data.index[0]}:{data.index[-1]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, model_name: str, data: TimeSeriesDataFrame):
        if not self._enabled:
            return None
        return self._cache.get(self._key(model_name, data))

    def put(self, model_name: str, data: TimeSeriesDataFrame, preds):
        if not self._enabled:
            return
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[self._key(model_name, data)] = preds

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# TimeSeriesPredictor
# ---------------------------------------------------------------------------

class TimeSeriesPredictor:
    """AutoML orchestrator for time series forecasting.

    Parameters
    ----------
    prediction_length : int
        Number of future time steps to forecast.
    eval_metric : str
        Metric for evaluation (default ``"MASE"``).
        AG uses ``"WQL"`` by default; we default to ``"MASE"`` for
        point-forecast-centric workflows.
    freq : str or None
        If ``None``, inferred from data.
    path : str or None
        Directory for saving.
    quantile_levels : list of float
        Default ``[0.1, 0.5, 0.9]``.
    verbosity : int
        0=silent, 1=basic, 2=detailed, 3=debug.
    cache_predictions : bool
        Cache predict() results (default True).
    """

    def __init__(
        self,
        prediction_length: int = 1,
        eval_metric: str = "MASE",
        freq: str | None = None,
        path: str | None = None,
        quantile_levels: list[float] | None = None,
        verbosity: int = 2,
        cache_predictions: bool = True,
        known_covariates_names: list[str] | None = None,
        target: str = "target",
        eval_metric_seasonal_period: int | None = None,
        horizon_weight: list[float] | np.ndarray | None = None,
        log_to_file: bool = False,
        log_file_path: str | None = None,
        label: str | None = None,
    ):
        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.freq = freq
        self.path = path or f"cbal_predictor_{int(time.time())}"
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        self.verbosity = verbosity
        self.known_covariates_names = known_covariates_names or []
        self.target = target
        self.eval_metric_seasonal_period = eval_metric_seasonal_period
        self.horizon_weight = (
            np.asarray(horizon_weight, dtype=np.float64)
            if horizon_weight is not None else None
        )
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.label = label

        self._models: dict[str, Any] = {}
        self._model_scores: dict[str, float] = {}
        self._model_fit_times: dict[str, float] = {}
        self._best_model_name: str | None = None
        self._ensemble = None
        self._stacking_ensemble = None
        self._is_fitted = False
        self._train_data = None
        self._val_data = None
        self._context_length: int | None = None
        self._pred_cache = _PredictionCache(enabled=cache_predictions)
        self._target_scaler = None
        self._cov_regressor = None
        self._cov_scaler = None  # CovariateScaler instance

        if verbosity >= 3:
            logging.basicConfig(level=logging.DEBUG)
        elif verbosity >= 2:
            logging.basicConfig(level=logging.INFO)

        if log_to_file:
            os.makedirs(self.path, exist_ok=True)
            lf = log_file_path or os.path.join(self.path, "predictor.log")
            os.makedirs(os.path.dirname(lf) if os.path.dirname(lf) else ".", exist_ok=True)
            fh = logging.FileHandler(lf, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            ))
            logging.getLogger("cbal").addHandler(fh)

    # =================================================================
    # fit()
    # =================================================================

    def fit(
        self,
        train_data: "TimeSeriesDataFrame | pd.DataFrame",
        presets: str | dict | None = "medium_quality",
        hyperparameters: dict[str, Any] | None = None,
        val_data: "TimeSeriesDataFrame | pd.DataFrame | None" = None,
        time_limit: float | None = None,
        excluded_model_types: list[str] | None = None,
        hyperparameter_tune_kwargs: str | dict | None = None,
        enable_ensemble: bool = True,
        num_val_windows: int | tuple[int, ...] = 1,
        refit_full: bool = False,
        random_seed: int | None = 123,
        ensemble_hyperparameters: dict | None = None,
        skip_model_selection: bool = False,
        val_step_size: int | None = None,
        refit_every_n_windows: int | None = 1,
        num_bag_folds: int = 0,
        conformal: bool = False,
    ) -> "TimeSeriesPredictor":
        """Train models and build ensemble.

        Parameters
        ----------
        train_data : TimeSeriesDataFrame
        presets : str or dict
        hyperparameters : dict, optional
            ``{ModelName: {param: val}}``.
            Values may be a *list* of dicts to train the same model
            type with multiple configurations (AutoGluon-style)::

                {"Theta": [{"seasonal_period": 7}, {"seasonal_period": 1}]}

        val_data : TimeSeriesDataFrame, optional
        time_limit : float, optional
        excluded_model_types : list, optional
        hyperparameter_tune_kwargs : str | dict | None
            ``"auto"`` → Bayesian (10 trials). Dict for full control.
        enable_ensemble : bool
            Set False to skip ensemble (default True).
        num_val_windows : int or tuple
            Number of backtest windows.  A tuple ``(n_ens1, n_ens2)``
            splits windows between ensemble layers (stacking).
        refit_full : bool
            Re-train best model on full data after selection.
        random_seed : int or None
        ensemble_hyperparameters : dict, optional
            Hyperparameters for ensemble model(s).
        skip_model_selection : bool
            If True, skip validation scoring (faster, no leaderboard).
        num_bag_folds : int
            Number of bagged repeats per model (0=disabled). Each repeat
            trains the same model with a different random seed, increasing
            ensemble diversity. Typical values: 3-5.
        conformal : bool
            If True, fit a conformal calibrator on the last validation
            window after training. This calibrates quantile predictions
            to have guaranteed coverage (e.g., 90% PI actually covers
            90% of the time).
        """
        # Auto-convert raw pd.DataFrame → TimeSeriesDataFrame
        train_data = self._auto_convert(train_data)
        if val_data is not None:
            val_data = self._auto_convert(val_data)

        start_time = time.time()

        # [Feature 6] random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        config = _resolve_preset(presets)

        # Resolve eval_metric with seasonal_period (auto-infer from freq like AG)
        _metric_sp = self.eval_metric_seasonal_period
        if _metric_sp is None and self.freq is not None:
            _metric_sp = _infer_seasonal_period(self.freq)
        self._resolved_metric = get_metric(
            self.eval_metric,
            seasonal_period=_metric_sp,
        )

        # [Feature 7] HPO "auto" shortcut
        hpo_kwargs = _resolve_hpo_kwargs(hyperparameter_tune_kwargs)
        hpo_enabled = hpo_kwargs is not None
        hpo_num_trials = hpo_kwargs.get("num_trials", 10) if hpo_kwargs else 10
        hpo_searcher = hpo_kwargs.get("searcher", "random") if hpo_kwargs else "random"

        if self.freq is None:
            self.freq = train_data.freq

        # [Feature 3] Auto context_length
        min_ts_len = min(
            len(train_data.loc[iid]) for iid in train_data.item_ids
        )
        self._context_length = _auto_context_length(
            self.prediction_length, self.freq, min_ts_len,
        )
        logger.info(f"Auto context_length = {self._context_length}")

        # --- Data profiling (for "auto" preset and time-budget scheduling) ---
        self._data_profile = _profile_data(
            train_data, self.prediction_length, self.known_covariates_names,
        )
        logger.info(
            f"Data profile: {self._data_profile['size_category']} "
            f"({self._data_profile['num_items']} items, "
            f"median_len={self._data_profile['median_length']}, "
            f"features={self._data_profile['feature_category']})"
        )

        # --- Resolve "auto" preset using data profile ---
        if config.get("_auto"):
            config = _build_auto_preset(self._data_profile, self.prediction_length)

        # [Feature 2] Expand list-valued hyperparameters
        if hyperparameters:
            for name, params in hyperparameters.items():
                if isinstance(params, list):
                    # Same model with multiple configs → unique names
                    for i, p in enumerate(params):
                        unique_name = f"{name}_{i+1}" if len(params) > 1 else name
                        config["models"][unique_name] = {
                            "_base_model_name": name, **p,
                        }
                    # Remove original entry only if we expanded to numbered names
                    if len(params) > 1:
                        config["models"].pop(name, None)
                elif name in config["models"]:
                    config["models"][name].update(params)
                else:
                    config["models"][name] = params

        if excluded_model_types:
            for name in excluded_model_types:
                config["models"].pop(name, None)

        # [Feature 10] Bagging: repeat DL/Tabular models with different seeds
        # Each bag fold gets:
        #   1. Different random seed → different weight initialization (DL)
        #   2. Different data subsample (bag_subsample_ratio) → different
        #      training trajectories → genuine ensemble diversity
        if num_bag_folds > 1:
            bagged_models = {}
            _deterministic = {
                "Naive", "SeasonalNaive", "Average", "Drift",
                "SeasonalAverage",
                "AutoETS", "AutoTheta", "AutoARIMA", "AutoCES",
            }
            for name, hp in list(config["models"].items()):
                base_name = hp.get("_base_model_name", name)
                if base_name in _deterministic:
                    bagged_models[name] = hp
                else:
                    for fold in range(num_bag_folds):
                        bag_name = f"{name}_bag{fold+1}"
                        bag_hp = dict(hp)
                        bag_hp["_base_model_name"] = base_name
                        bag_hp["_bag_seed"] = (random_seed or 0) + fold * 1000
                        # Subsample items for diversity (80% of items per fold)
                        bag_hp["_bag_subsample_ratio"] = 0.8
                        bagged_models[bag_name] = bag_hp
            config["models"] = bagged_models
            logger.info(
                f"Bagging enabled: {num_bag_folds} folds → "
                f"{len(config['models'])} models total"
            )

        # [Data-adaptive] Adjust hyperparameters based on data size
        config["_median_length"] = self._data_profile.get("median_length", min_ts_len)
        config = _adapt_hyperparameters_to_data(
            config,
            min_ts_length=min_ts_len,
            num_items=train_data.num_items,
            prediction_length=self.prediction_length,
            freq=self.freq,
        )

        # --- Time-budget scheduling: drop low-priority models if needed ---
        if time_limit is not None:
            config = _schedule_models_by_budget(
                config, time_limit, self._data_profile,
            )

        # --- Validation data ---
        # [Feature 1] Parse num_val_windows for multi-window + stacking
        if isinstance(num_val_windows, tuple):
            total_windows = sum(num_val_windows)
            stacking_split = num_val_windows  # (L1_windows, L2_windows)
        else:
            total_windows = num_val_windows
            stacking_split = None

        if val_data is None:
            min_needed = (total_windows + 1) * self.prediction_length
            if len(train_data) > min_needed * train_data.num_items:
                if total_windows > 1:
                    self._val_splits = train_data.multi_window_backtest_splits(
                        self.prediction_length, total_windows,
                        val_step_size=val_step_size,
                    )
                    if self._val_splits:
                        train_data = self._val_splits[0][0]
                        val_data = self._val_splits[-1][1]
                    else:
                        train_data, val_data = train_data.train_test_split(
                            self.prediction_length,
                        )
                        self._val_splits = [(train_data, val_data)]
                else:
                    train_data, val_data = train_data.train_test_split(
                        self.prediction_length,
                    )
                    self._val_splits = [(train_data, val_data)]
                logger.info(
                    f"Auto val split: {len(self._val_splits)} window(s), "
                    f"train={len(train_data)}, val={len(val_data)}"
                )
            else:
                logger.warning("Data too short for auto val split.")
                val_data = train_data
                self._val_splits = [(train_data, val_data)]
        else:
            self._val_splits = [(train_data, val_data)]

        self._train_data = train_data
        self._val_data = val_data

        # --- Non-negative target detection ---
        # If all training targets are >= 0, clamp predictions to 0+
        all_targets = train_data[TARGET].values
        self._target_is_nonneg = bool(np.all(all_targets[np.isfinite(all_targets)] >= 0))
        if self._target_is_nonneg:
            logger.info("Target is non-negative → predictions will be clamped to ≥ 0")

        # --- Covariate system: target_scaler + covariate_regressor ---
        # Extract scaler/regressor config from hyperparameters (model-level)
        # or from the preset config (predictor-level)
        scaler_method = config.get("target_scaler", None)
        cov_regressor = config.get("covariate_regressor", False)

        # Wire covariate names from data or predictor
        if self.known_covariates_names:
            train_data.known_covariates_names = self.known_covariates_names

        # target_scaler: scale targets per-item before training
        if scaler_method:
            from cbal.models.wrappers import TargetScaler
            self._target_scaler = TargetScaler(method=scaler_method)
            train_data = self._target_scaler.fit_transform(train_data)
            if val_data is not None and val_data is not train_data:
                val_data = self._target_scaler.transform(val_data)
            # Update splits with scaled data
            self._val_splits = [
                (self._target_scaler.transform(tr), self._target_scaler.transform(te))
                for tr, te in self._val_splits
            ]
            logger.info(f"TargetScaler applied: method={scaler_method}")

        # covariate_regressor: remove covariate effect before base models
        if cov_regressor and (
            self.known_covariates_names
            or train_data.past_covariates_names
            or (train_data.static_features is not None)
        ):
            from cbal.models.wrappers import CovariateRegressor
            static_feat_names = (
                list(train_data.static_features.columns)
                if train_data.static_features is not None else []
            )
            self._cov_regressor = CovariateRegressor(
                known_covariates_names=self.known_covariates_names,
                past_covariates_names=train_data.past_covariates_names,
                static_features_names=static_feat_names,
                backend=cov_regressor if isinstance(cov_regressor, str) else "lightgbm",
            )
            self._cov_regressor.fit(train_data, train_data.static_features)
            train_data = self._cov_regressor.remove_covariate_effect(
                train_data, train_data.static_features,
            )
            logger.info("CovariateRegressor applied: covariate effect removed from target")

        # covariate_scaler: normalize covariate features
        cov_scaler_cfg = config.get("covariate_scaler", None)
        if cov_scaler_cfg:
            from cbal.models.wrappers import CovariateScaler
            method = cov_scaler_cfg if isinstance(cov_scaler_cfg, str) else "global"
            self._cov_scaler = CovariateScaler(method=method)
            train_data, _ = self._cov_scaler.fit_transform(
                train_data, train_data.static_features,
            )
            if val_data is not None and val_data is not train_data:
                val_data, _ = self._cov_scaler.transform(
                    val_data, val_data.static_features if hasattr(val_data, 'static_features') else None,
                )
            logger.info(f"CovariateScaler applied: method={method}")

        # Per-model time limit — dynamic reallocation
        n_models = len(config["models"])
        if time_limit is not None:
            usable_budget = time_limit * 0.8  # 80% for models, 20% for ensemble/overhead
        else:
            usable_budget = None

        # --- Train each model ---
        logger.info(
            f"Fitting {n_models} models "
            f"(pred_len={self.prediction_length}, metric={self.eval_metric}, "
            f"val_windows={total_windows}"
            f"{', HPO=' + hpo_searcher if hpo_enabled else ''})..."
        )

        model_keys = list(config["models"].keys())
        cumulative_fit_time = 0.0

        for model_idx, model_key in enumerate(model_keys):
            model_hparams = config["models"][model_key]
            if time_limit and (time.time() - start_time) > time_limit * 0.9:
                logger.warning("Time limit approaching, skipping remaining.")
                break

            # Dynamic time reallocation: remaining budget / remaining models
            remaining_models = n_models - model_idx
            if usable_budget is not None:
                remaining_budget = max(usable_budget - cumulative_fit_time, 10)
                per_model_limit = max(remaining_budget / remaining_models, 10)
            else:
                per_model_limit = config.get("time_limit_per_model", 300)

            # [Feature 2] Resolve base model name for list-expanded entries
            base_name = model_hparams.get("_base_model_name", model_key)

            # Skip models flagged by data-adaptive logic
            if model_hparams.get("_skip", False):
                logger.info(f"  {model_key:25s} | SKIPPED (data too short)")
                continue

            # [Feature 3] Inject auto context_length if None
            _DL_MODEL_NAMES = {
                "DLinear", "PatchTST", "DeepAR", "TFT", "iTransformer",
                "N-HiTS", "TSMixer", "SegRNN", "TimeMixer", "TimesNet",
                "ModernTCN", "S-Mamba", "MambaTS", "MTGNN", "CrossGNN",
                "SimpleFeedForward", "TiDE",
            }
            if model_hparams.get("context_length") is None and base_name in _DL_MODEL_NAMES:
                model_hparams["context_length"] = self._context_length

            # Multi-GPU: assign DL models to GPUs in round-robin fashion
            if base_name in _DL_MODEL_NAMES and _has_gpu():
                import torch
                n_gpus = torch.cuda.device_count()
                if n_gpus > 1:
                    gpu_idx = model_idx % n_gpus
                    model_hparams["device"] = f"cuda:{gpu_idx}"

            # HPO check
            from cbal.hpo.space import SearchSpace as _SS
            has_search_space = hpo_enabled and any(
                isinstance(v, _SS) for v in model_hparams.values()
            )

            if has_search_space:
                model, score, fit_time = self._fit_with_hpo(
                    base_name, model_hparams, train_data, val_data,
                    per_model_limit, hpo_num_trials, hpo_searcher,
                )
            else:
                model, score, fit_time = self._fit_single_model(
                    base_name, model_hparams, train_data, val_data,
                    per_model_limit, self._resolved_metric,
                )

            if model is not None:
                self._models[model_key] = model
                self._model_scores[model_key] = score
                self._model_fit_times[model_key] = fit_time
                cumulative_fit_time += fit_time

        if not self._models:
            raise RuntimeError("No models were successfully trained.")

        if skip_model_selection:
            # Skip validation scoring — just pick first model
            self._best_model_name = next(iter(self._models))
            logger.info(f"skip_model_selection=True, using {self._best_model_name}")
        else:
            # --- [Feature 1] Multi-window scoring (with prediction caching) ---
            if len(self._val_splits) > 1:
                self._multi_window_rescore()

            # Best individual model (respects metric direction)
            _lower_better = getattr(self._resolved_metric, "sign", -1) <= 0
            self._best_model_name = (
                min(self._model_scores, key=self._model_scores.get)
                if _lower_better
                else max(self._model_scores, key=self._model_scores.get)
            )
            logger.info(
                f"Best model: {self._best_model_name} "
                f"(score={self._model_scores[self._best_model_name]:.4f})"
            )

        # --- [Feature 8] Ensemble ---
        if enable_ensemble and len(self._models) >= 2 and not skip_model_selection:
            ensemble_type = config.get("ensemble", "WeightedEnsemble")
            try:
                self._build_ensemble(ensemble_type, train_data, val_data,
                                     ensemble_hyperparameters)
            except Exception as e:
                logger.warning(f"Ensemble building failed: {e}.")

            # --- [Feature 9] Stacking (multi-layer ensemble) ---
            if stacking_split and len(self._val_splits) > 1 and self._ensemble:
                try:
                    self._build_stacking(stacking_split)
                except Exception as e:
                    logger.warning(f"Stacking failed: {e}.")

        # --- [Feature 4] refit_full ---
        if refit_full and val_data is not None:
            self._do_refit_full(train_data, val_data)

        # --- [Feature 11] Conformal calibration ---
        self._conformal_calibrator = None
        if conformal and val_data is not None:
            try:
                from cbal.models.conformal import ConformalCalibrator
                self._is_fitted = True  # temporarily set for predict
                calibrator = ConformalCalibrator(symmetric=True)
                # Use last val split for calibration
                cal_data = self._val_splits[-1][1] if self._val_splits else val_data
                calibrator.fit_from_predictor(self, cal_data)
                self._conformal_calibrator = calibrator
                logger.info(
                    f"Conformal calibration fitted. "
                    f"Median adjustments: {calibrator.coverage_adjustments}"
                )
            except Exception as e:
                logger.warning(f"Conformal calibration failed: {e}")

        self._is_fitted = True
        total_time = time.time() - start_time
        logger.info(
            f"Training complete. {len(self._models)} models in {total_time:.1f}s. "
            f"Best: {self._best_model_name}"
        )
        return self

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _fit_single_model(self, model_name, hparams, train_data, val_data,
                          time_limit, resolved_metric=None):
        # Apply bag seed if present
        bag_seed = hparams.get("_bag_seed")
        if bag_seed is not None:
            import torch
            np.random.seed(bag_seed)
            random.seed(bag_seed)
            torch.manual_seed(bag_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(bag_seed)

        # Subsample items for bag diversity
        bag_ratio = hparams.get("_bag_subsample_ratio")
        if bag_ratio is not None and bag_ratio < 1.0 and bag_seed is not None:
            rng = np.random.RandomState(bag_seed)
            all_items = list(train_data.item_ids)
            n_keep = max(int(len(all_items) * bag_ratio), 1)
            if n_keep < len(all_items):
                keep_items = rng.choice(all_items, size=n_keep, replace=False)
                train_data = TimeSeriesDataFrame(
                    train_data.loc[train_data.index.get_level_values(
                        ITEMID).isin(keep_items)]
                )
                train_data._cached_freq = self.freq

        # Filter internal keys before passing to model constructor
        clean_hparams = {k: v for k, v in hparams.items()
                         if not k.startswith("_")}

        model = _create_model(
            model_name, self.freq, self.prediction_length,
            clean_hparams, self.eval_metric,
        )
        if model is None:
            return None, float("inf"), 0
        try:
            t0 = time.time()
            model.fit(train_data, val_data=val_data, time_limit=time_limit)
            fit_time = time.time() - t0
            # Use resolved metric object (preserves seasonal_period etc.)
            metric = resolved_metric or self._resolved_metric
            score = model.score(val_data, metric=metric)
            logger.info(
                f"  {model_name:25s} | val_score={score:8.4f} | "
                f"fit_time={fit_time:6.1f}s"
            )
            return model, score, fit_time
        except Exception as e:
            logger.warning(f"  {model_name:25s} | FAILED: {e}")
            return None, float("inf"), 0

    def _fit_with_hpo(self, model_name, hparams, train_data, val_data,
                      time_limit, num_trials, searcher_method):
        from cbal.hpo.runner import tune_model
        from cbal.hpo.space import SearchSpace
        search_space, base_hp = {}, {}
        for k, v in hparams.items():
            (search_space if isinstance(v, SearchSpace) else base_hp)[k] = v
        logger.info(
            f"  {model_name:25s} | HPO: {num_trials} trials, "
            f"tuning {list(search_space.keys())}"
        )
        best_config, best_score, history = tune_model(
            model_name=model_name, search_space=search_space,
            train_data=train_data, val_data=val_data,
            freq=self.freq, prediction_length=self.prediction_length,
            eval_metric=self.eval_metric, num_trials=num_trials,
            searcher=searcher_method, time_limit=time_limit,
            base_hyperparameters=base_hp,
        )
        total_fit_time = sum(h.get("fit_time", 0) for h in history)
        if best_score == float("inf"):
            return None, float("inf"), total_fit_time
        model = _create_model(
            model_name, self.freq, self.prediction_length,
            best_config, self.eval_metric,
        )
        if model is None:
            return None, float("inf"), total_fit_time
        try:
            model.fit(train_data, val_data=val_data, time_limit=time_limit)
            score = model.score(val_data, metric=self.eval_metric)
            return model, score, total_fit_time
        except Exception:
            return None, float("inf"), total_fit_time

    # [Feature 1] Multi-window rescoring (with prediction caching)
    def _multi_window_rescore(self):
        """Re-score all models across multiple validation windows.

        Also caches per-window predictions for efficient ensemble building
        (avoids redundant predict calls in greedy selection).
        """
        logger.info(
            f"Multi-window rescoring across {len(self._val_splits)} windows..."
        )
        # Cache: {model_name: [{item_id: predictions}, ...per window]}
        self._cached_val_predictions: dict[str, list[dict[str, np.ndarray]]] = {}

        for name, model in self._models.items():
            scores = []
            window_preds = []
            for _train_w, val_w in self._val_splits:
                try:
                    s, preds = model.score_with_predictions(
                        val_w, metric=self._resolved_metric
                    )
                    if np.isfinite(s):
                        scores.append(s)
                    window_preds.append(preds)
                except Exception:
                    window_preds.append({})
            self._cached_val_predictions[name] = window_preds
            if scores:
                avg = float(np.mean(scores))
                self._model_scores[name] = avg
                logger.info(
                    f"  {name:25s} | multi-window avg={avg:.4f} "
                    f"(windows={len(scores)})"
                )

    def _build_ensemble(self, ensemble_type: str, train_data, val_data,
                         ensemble_hyperparameters=None):
        from cbal.models.ensemble import WeightedEnsemble, SimpleAverageEnsemble
        ens_hp = {"ensemble_size": 100, "metric": self._resolved_metric}
        if ensemble_hyperparameters:
            ens_hp.update(ensemble_hyperparameters)

        # Pass val_splits and cached predictions for efficient multi-window selection
        val_splits = self._val_splits if len(self._val_splits) > 1 else None
        cached_preds = getattr(self, "_cached_val_predictions", None)

        if ensemble_type == "WeightedEnsemble":
            ens = WeightedEnsemble(
                freq=self.freq, prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
                hyperparameters=ens_hp,
            )
        else:
            ens = SimpleAverageEnsemble(
                freq=self.freq, prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
            )
        ens.fit(train_data, val_data=val_data, base_models=self._models,
                val_splits=val_splits, cached_predictions=cached_preds)

        # Multi-window rescore for ensemble too (consistent with individual models)
        if val_splits is not None and len(val_splits) > 1:
            scores = []
            for _train_w, val_w in val_splits:
                try:
                    s = ens.score(val_w, metric=self.eval_metric)
                    if np.isfinite(s):
                        scores.append(s)
                except Exception:
                    pass
            ens_score = float(np.mean(scores)) if scores else ens.score(val_data, metric=self.eval_metric)
        else:
            ens_score = ens.score(val_data, metric=self.eval_metric)

        self._ensemble = ens
        self._model_scores["WeightedEnsemble"] = ens_score
        self._model_fit_times["WeightedEnsemble"] = getattr(ens, "fit_time", 0)
        logger.info(
            f"  {'Ensemble':25s} | val_score={ens_score:8.4f} | "
            f"weights={getattr(ens, 'weights', 'equal')}"
        )
        # Compare using metric direction
        _lower_better = getattr(self._resolved_metric, "sign", -1) <= 0
        best_current = self._model_scores[self._best_model_name]
        if (_lower_better and ens_score < best_current) or \
           (not _lower_better and ens_score > best_current):
            self._best_model_name = "WeightedEnsemble"

    # [Feature 9] Stacking (multi-layer ensemble)
    def _build_stacking(self, stacking_split: tuple[int, ...]):
        """Build a second-layer ensemble on top of the first ensemble.

        Uses later validation windows as L2 training data. The L2
        ensemble learns to combine L1 ensemble + individual model
        predictions.
        """
        if len(stacking_split) < 2:
            return

        l1_windows = stacking_split[0]
        l2_start = l1_windows
        l2_splits = self._val_splits[l2_start:]
        if not l2_splits:
            return

        logger.info(
            f"Building stacking ensemble (L2) on {len(l2_splits)} windows..."
        )
        from cbal.models.ensemble import WeightedEnsemble

        # For L2, the base models include the L1 ensemble itself
        l2_models = dict(self._models)
        if self._ensemble is not None:
            l2_models["L1_Ensemble"] = self._ensemble

        # Use last L2 window as val
        _, l2_val = l2_splits[-1]
        l2_train = l2_splits[-1][0] if l2_splits else self._train_data

        l2_ens = WeightedEnsemble(
            freq=self.freq, prediction_length=self.prediction_length,
            eval_metric=self.eval_metric,
            hyperparameters={"ensemble_size": 50, "metric": self.eval_metric},
        )
        l2_ens.fit(l2_train, val_data=l2_val, base_models=l2_models)
        l2_score = l2_ens.score(l2_val, metric=self.eval_metric)

        self._stacking_ensemble = l2_ens
        self._model_scores["StackedEnsemble"] = l2_score
        self._model_fit_times["StackedEnsemble"] = getattr(l2_ens, "fit_time", 0)
        logger.info(f"  StackedEnsemble | val_score={l2_score:.4f}")

        _lower_better = getattr(self._resolved_metric, "sign", -1) <= 0
        best_current = self._model_scores[self._best_model_name]
        if (_lower_better and l2_score < best_current) or \
           (not _lower_better and l2_score > best_current):
            self._best_model_name = "StackedEnsemble"
            logger.info("StackedEnsemble is the new best model.")

    # [Feature 4] refit_full
    def _do_refit_full(self, train_data, val_data):
        """Re-train the best model on train + val combined."""
        best = self._best_model_name
        if best in ("WeightedEnsemble", "StackedEnsemble"):
            logger.info("refit_full: ensemble selected, skipping refit.")
            return

        logger.info(f"refit_full: re-training {best} on full data...")
        try:
            full_data = TimeSeriesDataFrame(
                pd.concat([train_data, val_data]).sort_index()
            )
            full_data._cached_freq = self.freq
            # Deduplicate (val might contain train rows in AG convention)
            full_data = TimeSeriesDataFrame(
                full_data[~full_data.index.duplicated(keep="last")]
            )
            full_data._cached_freq = self.freq

            model = self._models[best]
            model.fit(full_data, time_limit=None)
            logger.info(f"refit_full: {best} retrained on {len(full_data)} rows.")
        except Exception as e:
            logger.warning(f"refit_full failed: {e}")

    # =================================================================
    # predict()
    # =================================================================

    def predict(
        self,
        data: TimeSeriesDataFrame,
        model: str | None = None,
        quantile_levels: list[float] | None = None,
        known_covariates: TimeSeriesDataFrame | None = None,
    ) -> TimeSeriesDataFrame:
        """Generate forecasts.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Historical context data.
        model : str, optional
            Model name to use. Default = best model.
        quantile_levels : list of float, optional
        known_covariates : TimeSeriesDataFrame, optional
            Future values of known covariates for each item.
            Required if ``known_covariates_names`` was set.

        Returns
        -------
        TimeSeriesDataFrame
        """

        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        model_name = model or self._best_model_name
        quantile_levels = quantile_levels or self.quantile_levels

        # [Feature 5] Prediction cache
        cached = self._pred_cache.get(model_name, data)
        if cached is not None:
            return cached

        # Pre-process: scale input data if scaler was used during fit
        predict_data = data
        if self._target_scaler is not None:
            predict_data = self._target_scaler.transform(data)

        if model_name == "StackedEnsemble" and self._stacking_ensemble is not None:
            preds = self._stacking_ensemble.predict(
                predict_data, quantile_levels=quantile_levels,
            )
        elif model_name == "WeightedEnsemble" and self._ensemble is not None:
            preds = self._ensemble.predict(
                predict_data, quantile_levels=quantile_levels,
            )
        elif model_name in self._models:
            preds = self._models[model_name].predict(
                predict_data, quantile_levels=quantile_levels,
            )
        else:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self._models.keys())}"
            )

        # Post-process: add covariate effect back
        if self._cov_regressor is not None and self._cov_regressor._is_fitted:
            preds = self._cov_regressor.add_covariate_effect(
                preds, known_covariates=known_covariates,
                static_features=(
                    data.static_features if hasattr(data, "static_features") else None
                ),
            )

        # Post-process: inverse scale
        if self._target_scaler is not None:
            preds = self._target_scaler.inverse_transform_predictions(preds)

        # Post-process: conformal calibration
        if getattr(self, "_conformal_calibrator", None) is not None:
            preds = self._conformal_calibrator.calibrate(
                preds, quantile_levels=quantile_levels,
            )

        # Post-process: clamp non-negative targets
        if getattr(self, "_target_is_nonneg", False):
            for col in preds.columns:
                preds[col] = preds[col].clip(lower=0.0)

        self._pred_cache.put(model_name, data, preds)
        return preds

    # =================================================================
    # leaderboard / score / summary
    # =================================================================

    def leaderboard(self, silent: bool = False) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted.")
        rows = [
            {"model": n, "score_val": self._model_scores[n],
             "fit_time_s": self._model_fit_times.get(n, 0)}
            for n in self._model_scores
        ]
        df = pd.DataFrame(rows).sort_values("score_val").reset_index(drop=True)
        if not silent:
            print(df.to_string(index=False))
        return df

    def fit_summary(self) -> dict[str, Any]:
        return {
            "prediction_length": self.prediction_length,
            "eval_metric": self.eval_metric,
            "freq": self.freq,
            "context_length": self._context_length,
            "n_models_trained": len(self._models),
            "best_model": self._best_model_name,
            "best_score": self._model_scores.get(self._best_model_name),
            "model_scores": dict(self._model_scores),
            "model_fit_times": dict(self._model_fit_times),
            "ensemble_weights": (
                self._ensemble.weights
                if self._ensemble and hasattr(self._ensemble, "weights")
                else None
            ),
        }

    def score(self, data, model=None, metric=None) -> float:
        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted.")
        model_name = model or self._best_model_name
        metric = metric or getattr(self, "_resolved_metric", None) or get_metric(self.eval_metric)
        if model_name == "StackedEnsemble" and self._stacking_ensemble:
            return self._stacking_ensemble.score(data, metric=metric)
        if model_name == "WeightedEnsemble" and self._ensemble:
            return self._ensemble.score(data, metric=metric)
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found.")
        return self._models[model_name].score(data, metric=metric)

    def evaluate(
        self,
        data: "TimeSeriesDataFrame | pd.DataFrame",
        model: str | None = None,
        metrics: "str | list[str] | None" = None,
    ) -> dict[str, float]:
        """Evaluate forecast accuracy (AG-compatible).

        Returns a dict of ``{metric_name: score}`` in **higher-is-better**
        convention (error metrics are negated).
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted.")
        data = self._auto_convert(data)

        if metrics is None:
            metrics = [self.eval_metric]
        elif isinstance(metrics, str):
            metrics = [metrics]

        model_name = model or self._best_model_name
        results = {}
        for m in metrics:
            s = self.score(data, model=model_name, metric=m)
            scorer = get_metric(m)
            # AG convention: higher is better → negate error metrics
            results[scorer.name] = s * scorer.sign
        return results

    def feature_importance(
        self,
        data: "TimeSeriesDataFrame | pd.DataFrame | None" = None,
        model: str | None = None,
        metric: str | None = None,
        num_iterations: int = 5,
        subsample_size: int | None = None,
        random_seed: int = 123,
    ) -> pd.DataFrame:
        """Permutation feature importance (AG-compatible).

        Shuffles each covariate column and measures score degradation.

        Returns
        -------
        pd.DataFrame
            Columns: ``importance``, ``stddev``, ``p_value``.
            Indexed by feature name. Sorted by importance (descending).
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted.")
        if data is None:
            data = self._val_data or self._train_data
        data = self._auto_convert(data)

        metric = metric or self.eval_metric
        model_name = model or self._best_model_name
        rng = np.random.RandomState(random_seed)

        # Baseline score
        base_score = self.score(data, model=model_name, metric=metric)

        # Identify feature columns
        feature_cols = [c for c in data.columns if c != TARGET]
        if not feature_cols:
            return pd.DataFrame(columns=["importance", "stddev", "p_value"])

        records = []
        for col in feature_cols:
            drops = []
            for _ in range(num_iterations):
                shuffled = data.copy()
                shuffled[col] = rng.permutation(shuffled[col].values)
                try:
                    shuffled_score = self.score(
                        shuffled, model=model_name, metric=metric,
                    )
                    drops.append(shuffled_score - base_score)
                except Exception:
                    drops.append(0.0)

            arr = np.array(drops)
            mean_drop = float(np.mean(arr))
            std_drop = float(np.std(arr))
            from scipy import stats as _stats
            if std_drop > 0 and len(arr) > 1:
                t_stat, p_val = _stats.ttest_1samp(arr, 0.0)
                p_val = float(p_val)
            else:
                p_val = 1.0
            records.append({
                "feature": col,
                "importance": mean_drop,
                "stddev": std_drop,
                "p_value": p_val,
            })

        result = pd.DataFrame(records).set_index("feature")
        return result.sort_values("importance", ascending=True)  # lower=more important (error metric)

    @property
    def model_names(self) -> list[str]:
        names = list(self._models.keys())
        if self._ensemble is not None:
            names.append("WeightedEnsemble")
        if self._stacking_ensemble is not None:
            names.append("StackedEnsemble")
        return names

    @property
    def best_model(self) -> str:
        return self._best_model_name

    def model_info(self, model_name=None) -> dict:
        name = model_name or self._best_model_name
        if name == "StackedEnsemble" and self._stacking_ensemble:
            return self._stacking_ensemble.model_info()
        if name == "WeightedEnsemble" and self._ensemble:
            return self._ensemble.model_info()
        if name in self._models:
            return self._models[name].model_info()
        raise ValueError(f"Model '{name}' not found.")

    # =================================================================
    # Save / Load
    # =================================================================

    def save(self) -> str:
        path = Path(self.path)
        path.mkdir(parents=True, exist_ok=True)
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)
        for name, model in self._models.items():
            model.save(str(models_dir / name))
        if self._ensemble is not None:
            with open(path / "ensemble.pkl", "wb") as f:
                pickle.dump(self._ensemble, f)
        if self._stacking_ensemble is not None:
            with open(path / "stacking.pkl", "wb") as f:
                pickle.dump(self._stacking_ensemble, f)
        if self._target_scaler is not None:
            with open(path / "target_scaler.pkl", "wb") as f:
                pickle.dump(self._target_scaler, f)
        if self._cov_regressor is not None:
            with open(path / "cov_regressor.pkl", "wb") as f:
                pickle.dump(self._cov_regressor, f)
        state = {
            "prediction_length": self.prediction_length,
            "eval_metric": self.eval_metric,
            "freq": self.freq,
            "quantile_levels": self.quantile_levels,
            "model_scores": self._model_scores,
            "model_fit_times": self._model_fit_times,
            "best_model_name": self._best_model_name,
            "model_names": list(self._models.keys()),
            "context_length": self._context_length,
            "known_covariates_names": self.known_covariates_names,
            "target_is_nonneg": getattr(self, "_target_is_nonneg", False),
        }
        with open(path / "predictor_state.pkl", "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Predictor saved to {path}")
        return str(path)

    @classmethod
    def load(cls, path: str) -> "TimeSeriesPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Predictor path not found: {path}")
        with open(path / "predictor_state.pkl", "rb") as f:
            state = pickle.load(f)
        predictor = cls(
            prediction_length=state["prediction_length"],
            eval_metric=state["eval_metric"],
            freq=state["freq"],
            quantile_levels=state["quantile_levels"],
            path=str(path),
            known_covariates_names=state.get("known_covariates_names"),
        )
        predictor._model_scores = state["model_scores"]
        predictor._model_fit_times = state["model_fit_times"]
        predictor._best_model_name = state["best_model_name"]
        predictor._context_length = state.get("context_length")
        from cbal.models.abstract_model import AbstractTimeSeriesModel
        models_dir = path / "models"
        for name in state["model_names"]:
            mp = str(models_dir / name)
            if os.path.exists(mp):
                try:
                    predictor._models[name] = AbstractTimeSeriesModel.load(mp)
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
        ens_path = path / "ensemble.pkl"
        if ens_path.exists():
            with open(ens_path, "rb") as f:
                predictor._ensemble = pickle.load(f)
                predictor._ensemble._base_models = predictor._models
        stack_path = path / "stacking.pkl"
        if stack_path.exists():
            with open(stack_path, "rb") as f:
                predictor._stacking_ensemble = pickle.load(f)
        scaler_path = path / "target_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                predictor._target_scaler = pickle.load(f)
        reg_path = path / "cov_regressor.pkl"
        if reg_path.exists():
            with open(reg_path, "rb") as f:
                predictor._cov_regressor = pickle.load(f)
        predictor._target_is_nonneg = state.get("target_is_nonneg", False)
        predictor._resolved_metric = get_metric(
            predictor.eval_metric,
            seasonal_period=predictor.eval_metric_seasonal_period
            if hasattr(predictor, "eval_metric_seasonal_period") else None,
        )
        predictor._is_fitted = True
        return predictor

    # =================================================================
    # helpers
    # =================================================================

    def _auto_convert(self, data) -> TimeSeriesDataFrame:
        """Auto-convert raw pd.DataFrame → TimeSeriesDataFrame (AG compat)."""
        if isinstance(data, TimeSeriesDataFrame):
            return data
        if isinstance(data, pd.DataFrame):
            logger.info("Auto-converting pd.DataFrame → TimeSeriesDataFrame")
            return TimeSeriesDataFrame.from_data_frame(data)
        if isinstance(data, (str, Path)):
            logger.info(f"Auto-loading from path: {data}")
            return TimeSeriesDataFrame.from_path(data)
        return data

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"TimeSeriesPredictor(pred_len={self.prediction_length}, "
                f"metric={self.eval_metric}, "
                f"n_models={len(self._models)}, "
                f"best={self._best_model_name})"
            )
        return (
            f"TimeSeriesPredictor(pred_len={self.prediction_length}, "
            f"metric={self.eval_metric}, not fitted)"
        )
