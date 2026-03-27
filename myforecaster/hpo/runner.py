"""
HPO Runner — orchestrates the hyperparameter tuning loop.

Trains a model with different hyperparameter configs, evaluates each
on validation data, and returns the best configuration.

Usage::

    from myforecaster.hpo.runner import tune_model
    from myforecaster.hpo.space import Int, Real

    best_config, best_score, history = tune_model(
        model_name="PatchTST",
        search_space={"d_model": Int(32, 256, log=True), "n_layers": Int(1, 4)},
        train_data=train_data,
        val_data=val_data,
        freq="D",
        prediction_length=14,
        eval_metric="MASE",
        num_trials=20,
        searcher="random",
        time_limit=600,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
from myforecaster.hpo.searcher import BaseSearcher, get_searcher
from myforecaster.hpo.space import SearchSpace

logger = logging.getLogger(__name__)


def tune_model(
    model_name: str,
    search_space: dict[str, Any],
    train_data: TimeSeriesDataFrame,
    val_data: TimeSeriesDataFrame,
    freq: str,
    prediction_length: int,
    eval_metric: str = "MASE",
    num_trials: int = 10,
    searcher: str | BaseSearcher = "random",
    time_limit: float | None = None,
    seed: int = 0,
    base_hyperparameters: dict[str, Any] | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], float, list[dict]]:
    """Run HPO for a single model type.

    Parameters
    ----------
    model_name : str
    search_space : dict
    train_data, val_data : TimeSeriesDataFrame
    freq : str
    prediction_length : int
    eval_metric : str
    num_trials : int
    searcher : str or BaseSearcher
    time_limit : float, optional
    seed : int
    base_hyperparameters : dict, optional
    n_jobs : int
        Number of parallel workers.  ``-1`` uses all cores.
        Only effective for random search; bayesian search is sequential.
    """
    from myforecaster.predictor import _create_model

    if isinstance(searcher, str):
        searcher_obj = get_searcher(searcher, search_space, seed=seed)
    else:
        searcher_obj = searcher

    base_hp = base_hyperparameters or {}
    start_time = time.time()

    logger.info(
        f"HPO: tuning {model_name} | trials={num_trials} | "
        f"method={type(searcher_obj).__name__}"
        f"{f' | n_jobs={n_jobs}' if n_jobs != 1 else ''}"
    )

    # Parallel random search
    is_random = type(searcher_obj).__name__ == "RandomSearcher"
    if n_jobs != 1 and is_random:
        return _parallel_tune(
            model_name, searcher_obj, base_hp, train_data, val_data,
            freq, prediction_length, eval_metric, num_trials,
            time_limit, start_time, n_jobs,
        )

    # Sequential loop (bayesian or n_jobs=1)
    history = []

    for trial_idx in range(num_trials):
        # Check time limit
        if time_limit and (time.time() - start_time) > time_limit:
            logger.info(f"HPO: time limit reached after {trial_idx} trials.")
            break

        # Suggest config
        sampled = searcher_obj.suggest()
        config = {**base_hp, **sampled}

        # Per-trial time limit
        remaining = None
        if time_limit:
            remaining = max(time_limit - (time.time() - start_time), 10)
            per_trial_limit = min(remaining / max(num_trials - trial_idx, 1), remaining)
        else:
            per_trial_limit = None

        # Create and train model
        try:
            model = _create_model(
                model_name, freq, prediction_length, config, eval_metric,
            )
            if model is None:
                logger.warning(f"  Trial {trial_idx}: model creation failed, skipping.")
                continue

            t0 = time.time()
            model.fit(train_data, val_data=val_data, time_limit=per_trial_limit)
            fit_time = time.time() - t0

            score = model.score(val_data, metric=eval_metric)

            trial_result = {
                "trial_idx": trial_idx,
                "config": config,
                "score": score,
                "fit_time": fit_time,
            }
            history.append(trial_result)
            searcher_obj.report(config, score)

            logger.info(
                f"  Trial {trial_idx:3d} | score={score:8.4f} | "
                f"time={fit_time:5.1f}s | {_config_summary(sampled)}"
            )

        except Exception as e:
            logger.warning(f"  Trial {trial_idx:3d} | FAILED: {e}")
            searcher_obj.report(config, float("inf"))
            history.append({
                "trial_idx": trial_idx, "config": config,
                "score": float("inf"), "fit_time": 0, "error": str(e),
            })

    total_time = time.time() - start_time

    if not history or all(h["score"] == float("inf") for h in history):
        logger.warning("HPO: no successful trials.")
        return base_hp, float("inf"), history

    best = min(history, key=lambda x: x["score"])
    logger.info(
        f"HPO complete: {len(history)} trials in {total_time:.1f}s | "
        f"best_score={best['score']:.4f}"
    )

    return best["config"], best["score"], history


def _config_summary(config: dict, max_items: int = 4) -> str:
    """Compact string representation of a config dict."""
    items = list(config.items())[:max_items]
    parts = []
    for k, v in items:
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    if len(config) > max_items:
        parts.append(f"...+{len(config) - max_items}")
    return ", ".join(parts)


def _run_single_trial(
    model_name, config, train_data, val_data, freq,
    prediction_length, eval_metric, per_trial_limit, trial_idx,
):
    """Run a single HPO trial (picklable for joblib)."""
    from myforecaster.predictor import _create_model
    try:
        model = _create_model(
            model_name, freq, prediction_length, config, eval_metric,
        )
        if model is None:
            return {"trial_idx": trial_idx, "config": config,
                    "score": float("inf"), "fit_time": 0, "error": "model creation failed"}
        t0 = time.time()
        model.fit(train_data, val_data=val_data, time_limit=per_trial_limit)
        fit_time = time.time() - t0
        score = model.score(val_data, metric=eval_metric)
        return {"trial_idx": trial_idx, "config": config,
                "score": score, "fit_time": fit_time}
    except Exception as e:
        return {"trial_idx": trial_idx, "config": config,
                "score": float("inf"), "fit_time": 0, "error": str(e)}


def _parallel_tune(
    model_name, searcher_obj, base_hp, train_data, val_data,
    freq, prediction_length, eval_metric, num_trials,
    time_limit, start_time, n_jobs,
):
    """Parallel HPO using joblib (random search only)."""
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not installed, falling back to sequential HPO.")
        # Fall back to sequential
        from myforecaster.hpo.runner import tune_model
        return tune_model(
            model_name, searcher_obj._search_space if hasattr(searcher_obj, '_search_space') else {},
            train_data, val_data, freq, prediction_length,
            eval_metric, num_trials, searcher_obj,
            time_limit, 0, base_hp, n_jobs=1,
        )

    # Pre-generate all configs
    configs = []
    for i in range(num_trials):
        sampled = searcher_obj.suggest()
        configs.append({**base_hp, **sampled})

    per_trial_limit = None
    if time_limit:
        per_trial_limit = time_limit / max(num_trials, 1)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_single_trial)(
            model_name, cfg, train_data, val_data, freq,
            prediction_length, eval_metric, per_trial_limit, idx,
        )
        for idx, cfg in enumerate(configs)
    )

    total_time = time.time() - start_time
    history = list(results)

    if not history or all(h["score"] == float("inf") for h in history):
        logger.warning("HPO: no successful trials.")
        return base_hp, float("inf"), history

    best = min(history, key=lambda x: x["score"])
    logger.info(
        f"HPO parallel complete: {len(history)} trials in {total_time:.1f}s | "
        f"best_score={best['score']:.4f}"
    )
    return best["config"], best["score"], history


# ---------------------------------------------------------------------------
# Default search spaces per model
# ---------------------------------------------------------------------------

def get_default_search_space(model_name: str) -> dict[str, Any]:
    """Return a sensible default search space for a model.

    These are AutoGluon-inspired defaults that work well across datasets.
    """
    from myforecaster.hpo.space import Categorical, Int, Real

    _spaces = {
        "DLinear": {
            "kernel_size": Int(11, 51, log=False),
            "learning_rate": Real(1e-4, 1e-2, log=True),
            "max_epochs": Categorical(30, 50, 100),
        },
        "DeepAR": {
            "hidden_size": Categorical(32, 64, 128),
            "num_layers": Int(1, 3),
            "learning_rate": Real(1e-4, 1e-2, log=True),
            "dropout": Real(0.0, 0.3),
            "max_epochs": Categorical(30, 50, 100),
        },
        "PatchTST": {
            "d_model": Categorical(32, 64, 128, 256),
            "n_heads": Categorical(2, 4, 8),
            "n_layers": Int(1, 4),
            "d_ff": Categorical(64, 128, 256, 512),
            "patch_len": Categorical(8, 12, 16, 24),
            "stride": Categorical(4, 8, 12),
            "dropout": Real(0.1, 0.4),
            "learning_rate": Real(1e-5, 1e-3, log=True),
        },
        "TFT": {
            "d_model": Categorical(32, 64, 128),
            "n_heads": Categorical(1, 2, 4),
            "dropout": Real(0.05, 0.3),
            "learning_rate": Real(1e-4, 5e-3, log=True),
            "max_epochs": Categorical(20, 30, 50),
        },
        "iTransformer": {
            "d_model": Categorical(64, 128, 256, 512),
            "n_layers": Int(1, 4),
            "n_heads": Categorical(4, 8),
            "dropout": Real(0.05, 0.3),
            "learning_rate": Real(1e-5, 1e-3, log=True),
        },
        "N-HiTS": {
            "hidden_size": Categorical(64, 128, 256, 512),
            "n_stacks": Int(2, 4),
            "n_blocks": Int(1, 3),
            "learning_rate": Real(1e-4, 1e-2, log=True),
            "max_epochs": Categorical(30, 50, 100),
        },
        "TSMixer": {
            "d_ff": Categorical(32, 64, 128),
            "n_layers": Int(2, 8),
            "dropout": Real(0.0, 0.3),
            "learning_rate": Real(1e-4, 1e-2, log=True),
        },
        "TimeMixer": {
            "d_model": Categorical(32, 64, 128),
            "n_scales": Int(2, 5),
            "n_layers": Int(1, 4),
            "dropout": Real(0.0, 0.3),
            "learning_rate": Real(1e-4, 1e-2, log=True),
        },
        "SegRNN": {
            "d_model": Categorical(64, 128, 256),
            "seg_len": Categorical(6, 12, 24),
            "strategy": Categorical("rmr", "pmr"),
            "learning_rate": Real(1e-4, 1e-2, log=True),
        },
        "ModernTCN": {
            "d_model": Categorical(64, 128, 256),
            "n_layers": Int(2, 6),
            "kernel_size": Categorical(21, 31, 51),
            "learning_rate": Real(1e-4, 1e-2, log=True),
        },
        "TimesNet": {
            "d_model": Categorical(32, 64, 128),
            "d_ff": Categorical(32, 64, 128),
            "n_layers": Int(1, 3),
            "top_k": Int(3, 7),
            "learning_rate": Real(1e-4, 1e-2, log=True),
        },
        "RecursiveTabular": {
            "backend": Categorical("lightgbm", "xgboost"),
        },
        "DirectTabular": {
            "backend": Categorical("lightgbm", "xgboost"),
        },
    }
    return _spaces.get(model_name, {})
