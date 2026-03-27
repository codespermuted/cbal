"""
Ensemble — Weighted model combination with greedy selection.

Implements AutoGluon-style ensemble selection:

1. **Greedy Selection**: iteratively adds the model that most improves the
   ensemble score on validation data (with replacement).
2. **Weighted Average**: final predictions are a weighted average of
   selected base model predictions.

Usage::

    from cbal.models.ensemble import WeightedEnsemble

    # Assume base_models is a dict of name→fitted_model
    ensemble = WeightedEnsemble(
        freq="D", prediction_length=24,
        hyperparameters={"ensemble_size": 25},
    )
    ensemble.fit(train_data, val_data=val_data, base_models=base_models)
    pred = ensemble.predict(train_data)

Reference: Caruana et al., "Ensemble Selection from Libraries of Models"
(ICML 2004) — the algorithm AutoGluon uses.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cbal.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame
from cbal.metrics.scorers import DEFAULT_METRIC, TimeSeriesScorer, get_metric
from cbal.models.abstract_model import AbstractTimeSeriesModel
from cbal.models import register_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Greedy Ensemble Selection (Caruana et al. 2004)
# ---------------------------------------------------------------------------

_ENSEMBLE_QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def _compute_per_item_predictions(
    model: AbstractTimeSeriesModel,
    val_data: TimeSeriesDataFrame,
    quantile_levels: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Get predictions per item from a fitted model.

    Returns dict: item_id → array.
    - If ``quantile_levels`` is None: (prediction_length,) mean array.
    - If ``quantile_levels`` is provided: (prediction_length, 1+Q) array
      where columns are [mean, q1, q2, ...].
    """
    q_levels = quantile_levels or _ENSEMBLE_QUANTILE_LEVELS
    try:
        pred_len = model.prediction_length
        item_preds = {}

        # Batch predict: build context for all items at once
        context_tsdf = val_data.slice_by_timestep(None, -pred_len)
        context_tsdf._cached_freq = model.freq
        pred = model.predict(context_tsdf, quantile_levels=q_levels)

        for item_id in val_data.item_ids:
            item_df = val_data.loc[item_id]
            if len(item_df) <= pred_len:
                continue
            try:
                item_pred = pred.loc[item_id]
            except KeyError:
                continue

            y_mean = item_pred["mean"].values[:pred_len] if "mean" in pred.columns else item_pred.iloc[:, 0].values[:pred_len]

            if quantile_levels:
                # Pack [mean, q1, q2, ...] into 2D array
                cols = [y_mean]
                for q in q_levels:
                    q_col = str(q)
                    if q_col in pred.columns:
                        cols.append(item_pred[q_col].values[:pred_len])
                    else:
                        cols.append(y_mean)
                item_preds[item_id] = np.column_stack(cols)  # (H, 1+Q)
            else:
                item_preds[item_id] = y_mean

        return item_preds
    except Exception as e:
        logger.warning(f"Failed to get predictions from {model.name}: {e}")
        return {}


def _compute_multi_window_predictions(
    model: AbstractTimeSeriesModel,
    val_splits: list[tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]],
) -> list[dict[str, np.ndarray]]:
    """Get predictions per item for each validation window.

    Returns list of dicts, one per window:
        [{item_id: predictions_array}, ...]
    """
    window_preds = []
    for _train_w, val_w in val_splits:
        preds = _compute_per_item_predictions(model, val_w)
        window_preds.append(preds)
    return window_preds


def _score_multi_window(
    running_sums: list[dict[str, np.ndarray]],
    n_selected: int,
    val_splits: list[tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]],
    pred_len: int,
    scorer: TimeSeriesScorer,
    train_tails_list: list[dict[str, np.ndarray]] | None = None,
) -> float:
    """Compute average score across multiple validation windows."""
    window_scores = []
    for w_idx, (_train_w, val_w) in enumerate(val_splits):
        running_sum = running_sums[w_idx]
        train_tails = train_tails_list[w_idx] if train_tails_list else None
        s = _score_from_running_avg(
            running_sum, n_selected, val_w, pred_len, scorer, train_tails
        )
        if np.isfinite(s):
            window_scores.append(s)
    return float(np.mean(window_scores)) if window_scores else float("inf")


def _score_from_running_avg(
    running_sum: dict[str, np.ndarray],
    n_selected: int,
    val_data: TimeSeriesDataFrame,
    pred_len: int,
    scorer: TimeSeriesScorer,
    train_tails: dict[str, np.ndarray] | None = None,
    quantile_levels: list[float] | None = None,
) -> float:
    """Compute score from running sum (efficient: no full recompute).

    When ``quantile_levels`` is provided and running_sum values are 2-D
    (shape ``(H, 1+Q)``), passes quantile predictions to the scorer for
    quantile-aware metrics like WQL.
    """
    scores = []
    is_quantile_aware = quantile_levels and any(
        hasattr(running_sum.get(iid), 'ndim') and
        running_sum.get(iid) is not None and
        np.asarray(running_sum.get(iid)).ndim == 2
        for iid in list(running_sum.keys())[:1]
    )

    for item_id in val_data.item_ids:
        item_df = val_data.loc[item_id]
        n = len(item_df)
        if n <= pred_len:
            continue

        y_true = item_df[TARGET].values[-pred_len:]

        if item_id not in running_sum or n_selected == 0:
            continue

        raw = running_sum[item_id] / n_selected
        y_train = train_tails.get(item_id) if train_tails else None

        try:
            if is_quantile_aware and raw.ndim == 2:
                # raw: (H, 1+Q) → point=col0, quantiles=col1:
                y_pred = raw[:pred_len, 0]
                y_pred_q = raw[:pred_len, 1:]  # (H, Q)
                s = scorer(y_true[:pred_len], y_pred_q, y_train=y_train,
                           quantile_levels=quantile_levels)
            else:
                y_pred = raw[:pred_len] if raw.ndim == 1 else raw[:pred_len, 0]
                s = scorer(y_true[:pred_len], y_pred, y_train=y_train)
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            pass

    return float(np.mean(scores)) if scores else float("inf")


def greedy_ensemble_selection(
    model_predictions: dict[str, dict[str, np.ndarray]],
    val_data: TimeSeriesDataFrame,
    prediction_length: int,
    scorer: TimeSeriesScorer,
    ensemble_size: int = 100,
    train_tails: dict[str, np.ndarray] | None = None,
    *,
    multi_window_predictions: dict[str, list[dict[str, np.ndarray]]] | None = None,
    val_splits: list[tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]] | None = None,
    train_tails_list: list[dict[str, np.ndarray]] | None = None,
    quantile_levels: list[float] | None = None,
) -> tuple[dict[str, float], float]:
    """Greedy ensemble selection (Caruana et al., ICML 2004).

    Correct implementation following the original paper + AutoGluon:

    1. **Initialize** with the best individual model
    2. **For each round**: add the model that gives the best ensemble score
       — even if it temporarily worsens the score (no early stopping)
    3. **Track the best intermediate ensemble** across all rounds
    4. **Return** the ensemble at the round with the best validation score

    Handles both lower-is-better (MAE, MASE, ...) and higher-is-better
    (Coverage, R², ...) metrics via ``scorer.sign``.

    Uses incremental running-sum for O(N·M·K) efficiency.

    When ``multi_window_predictions`` and ``val_splits`` are provided, the
    greedy selection scores candidate ensembles across **all** validation
    windows (averaged), producing more robust weight estimates.

    Parameters
    ----------
    model_predictions : dict
        {model_name: {item_id: predictions_array}} — single-window fallback.
    val_data : TimeSeriesDataFrame
        Single validation window (used when multi_window not provided).
    prediction_length : int
    scorer : TimeSeriesScorer
    ensemble_size : int
        Number of selection rounds (default 100, like AutoGluon).
    train_tails : dict, optional
        {item_id: train_history} for scale-dependent metrics (single window).
    multi_window_predictions : dict, optional
        {model_name: [window0_preds, window1_preds, ...]} where each entry is
        {item_id: predictions_array}.
    val_splits : list of (train, val) tuples, optional
        Multiple validation windows for multi-window scoring.
    train_tails_list : list of dicts, optional
        Per-window train tails for scale-dependent metrics.
    quantile_levels : list of float, optional
        When provided and model_predictions contain 2-D arrays (with
        quantile columns), scoring uses these levels for quantile-aware
        metrics (e.g. WQL).

    Returns
    -------
    weights : dict[str, float]
        Model name → weight (sums to 1).
    best_score : float
    """
    use_multi_window = (
        multi_window_predictions is not None
        and val_splits is not None
        and len(val_splits) > 1
    )

    # Determine comparison direction: lower_is_better (sign=-1) vs higher_is_better (sign=1)
    # We normalize to "lower is better" internally by multiplying scores by -sign when sign=1
    lower_is_better = getattr(scorer, "sign", -1) <= 0

    def _is_better(new_score: float, old_score: float) -> bool:
        """Return True if new_score is better than old_score."""
        if lower_is_better:
            return new_score < old_score
        return new_score > old_score

    worst_score = float("inf") if lower_is_better else float("-inf")

    model_names = list(model_predictions.keys())
    if not model_names:
        return {}, float("inf") if lower_is_better else float("-inf")

    all_preds = {name: model_predictions[name] for name in model_names}
    item_ids = list(val_data.item_ids)
    n_windows = len(val_splits) if use_multi_window else 1

    # Multi-window: get item_ids per window
    if use_multi_window:
        mw_preds = multi_window_predictions
        mw_item_ids = [list(val_splits[w][1].item_ids) for w in range(n_windows)]
    else:
        mw_preds = None
        mw_item_ids = None

    def _score_candidate_single(running_sum, n_sel):
        return _score_from_running_avg(
            running_sum, n_sel, val_data, prediction_length, scorer, train_tails,
            quantile_levels=quantile_levels,
        )

    def _score_candidate_multi(running_sums, n_sel):
        return _score_multi_window(
            running_sums, n_sel, val_splits, prediction_length, scorer, train_tails_list
        )

    # --- Step 1: Initialize with best single model ---
    best_init_name = None
    best_init_score = worst_score
    for name in model_names:
        if use_multi_window:
            temp_sums = []
            for w in range(n_windows):
                w_preds = mw_preds[name][w] if name in mw_preds and w < len(mw_preds[name]) else {}
                temp_sum = {}
                for item_id in mw_item_ids[w]:
                    if item_id in w_preds:
                        temp_sum[item_id] = w_preds[item_id].copy()
                temp_sums.append(temp_sum)
            score = _score_candidate_multi(temp_sums, 1)
        else:
            temp_sum = {}
            for item_id in item_ids:
                if item_id in all_preds[name]:
                    temp_sum[item_id] = all_preds[name][item_id].copy()
            score = _score_candidate_single(temp_sum, 1)

        if _is_better(score, best_init_score):
            best_init_score = score
            best_init_name = name

    # Initialize selected list and running sum(s)
    selected: list[str] = [best_init_name]

    if use_multi_window:
        running_sums: list[dict[str, np.ndarray]] = []
        for w in range(n_windows):
            w_preds = mw_preds[best_init_name][w] if best_init_name in mw_preds and w < len(mw_preds[best_init_name]) else {}
            rs = {}
            for item_id in mw_item_ids[w]:
                rs[item_id] = w_preds.get(item_id, np.zeros(prediction_length)).copy()
            running_sums.append(rs)
    else:
        running_sums = None

    running_sum: dict[str, np.ndarray] = {}
    for item_id in item_ids:
        if item_id in all_preds[best_init_name]:
            running_sum[item_id] = all_preds[best_init_name][item_id].copy()
        else:
            running_sum[item_id] = np.zeros(prediction_length)

    # Track best ensemble seen across all rounds
    best_overall_score = best_init_score
    best_overall_selected = list(selected)

    logger.debug(f"  Init: {best_init_name} (score={best_init_score:.4f})")

    # --- Step 2: Greedy rounds (NO early stopping) ---
    for round_idx in range(1, ensemble_size):
        best_candidate = None
        best_candidate_score = worst_score

        for candidate_name in model_names:
            n_trial = len(selected) + 1

            if use_multi_window:
                trial_sums = []
                for w in range(n_windows):
                    w_preds = mw_preds[candidate_name][w] if candidate_name in mw_preds and w < len(mw_preds[candidate_name]) else {}
                    trial = {}
                    for item_id in mw_item_ids[w]:
                        base = running_sums[w].get(item_id, np.zeros(prediction_length))
                        cand = w_preds.get(item_id, np.zeros(prediction_length))
                        min_len = min(len(base), len(cand), prediction_length)
                        t = base.copy()
                        t[:min_len] += cand[:min_len]
                        trial[item_id] = t
                    trial_sums.append(trial)
                score = _score_candidate_multi(trial_sums, n_trial)
            else:
                trial_sum = {}
                for item_id in item_ids:
                    base = running_sum.get(item_id, np.zeros(prediction_length))
                    cand = all_preds[candidate_name].get(item_id, np.zeros(prediction_length))
                    min_len = min(len(base), len(cand), prediction_length)
                    trial = base.copy()
                    trial[:min_len] += cand[:min_len]
                    trial_sum[item_id] = trial
                score = _score_candidate_single(trial_sum, n_trial)

            if _is_better(score, best_candidate_score):
                best_candidate_score = score
                best_candidate = candidate_name

        # ALWAYS add best candidate (no early stopping — per Caruana 2004)
        selected.append(best_candidate)

        if use_multi_window:
            for w in range(n_windows):
                w_preds = mw_preds[best_candidate][w] if best_candidate in mw_preds and w < len(mw_preds[best_candidate]) else {}
                for item_id in mw_item_ids[w]:
                    cand = w_preds.get(item_id, np.zeros(prediction_length))
                    if item_id not in running_sums[w]:
                        running_sums[w][item_id] = np.zeros(prediction_length)
                    min_len = min(len(running_sums[w][item_id]), len(cand), prediction_length)
                    running_sums[w][item_id][:min_len] += cand[:min_len]

        for item_id in item_ids:
            cand = all_preds[best_candidate].get(item_id, np.zeros(prediction_length))
            if item_id not in running_sum:
                running_sum[item_id] = np.zeros(prediction_length)
            min_len = min(len(running_sum[item_id]), len(cand), prediction_length)
            running_sum[item_id][:min_len] += cand[:min_len]

        # Track best intermediate ensemble
        if _is_better(best_candidate_score, best_overall_score):
            best_overall_score = best_candidate_score
            best_overall_selected = list(selected)
            logger.debug(
                f"  Round {round_idx}: +{best_candidate} "
                f"(score={best_overall_score:.4f}★, size={len(selected)})"
            )
        else:
            logger.debug(
                f"  Round {round_idx}: +{best_candidate} "
                f"(score={best_candidate_score:.4f}, size={len(selected)})"
            )

    # --- Step 3: Return best intermediate ensemble ---
    counts = Counter(best_overall_selected)
    total = sum(counts.values())
    weights = {name: count / total for name, count in counts.items()}

    logger.info(
        f"  Greedy selection: {len(best_overall_selected)}/{ensemble_size} rounds, "
        f"best_score={best_overall_score:.4f}"
    )

    return weights, best_overall_score


# ---------------------------------------------------------------------------
# WeightedEnsemble Model
# ---------------------------------------------------------------------------

@register_model("WeightedEnsemble")
class WeightedEnsemble(AbstractTimeSeriesModel):
    """Weighted ensemble of multiple fitted models.

    Uses greedy ensemble selection (Caruana 2004) to learn weights on
    validation data, then combines predictions via weighted average.

    Parameters
    ----------
    freq, prediction_length : same as base models
    hyperparameters : dict, optional
        - ``ensemble_size`` (int, default 25): max greedy rounds
        - ``metric`` (str, default ``"MAE"``): metric for selection

    The ``fit()`` method requires ``base_models`` kwarg:
        ``ensemble.fit(train_data, val_data=val_data, base_models=base_models)``
    where ``base_models`` is a dict of ``{name: fitted_model}``.
    """

    _default_hyperparameters: dict[str, Any] = {
        "ensemble_size": 100,
        "metric": "MASE",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_models: dict[str, AbstractTimeSeriesModel] = {}
        self._weights: dict[str, float] = {}
        self._ensemble_score: float = float("inf")

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        base_models: dict[str, AbstractTimeSeriesModel] | None = None,
        time_limit: float | None = None,
        val_splits: list[tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]] | None = None,
        cached_predictions: dict[str, list[dict[str, np.ndarray]]] | None = None,
        **kwargs,
    ) -> "WeightedEnsemble":
        """Fit the ensemble weights using greedy selection.

        Parameters
        ----------
        train_data : TimeSeriesDataFrame
        val_data : TimeSeriesDataFrame
            Required — used for greedy selection (single-window fallback).
        base_models : dict of {name: fitted_model}
            Required — already fitted base models.
        val_splits : list of (train, val) tuples, optional
            Multiple validation windows. When provided, greedy selection
            scores candidate ensembles across all windows for robustness.
        cached_predictions : dict, optional
            {model_name: [{item_id: predictions}, ...per window]} — pre-computed
            predictions from multi-window rescoring. Avoids redundant predict calls.
        """
        if base_models is None or len(base_models) == 0:
            raise ValueError("WeightedEnsemble requires base_models dict.")
        if val_data is None:
            raise ValueError("WeightedEnsemble requires val_data for weight selection.")

        start_time = time.time()

        # Store references to fitted models
        self._base_models = base_models

        # Infer freq
        if self.freq is None:
            self.freq = val_data.freq

        use_multi_window = val_splits is not None and len(val_splits) > 1
        has_cache = cached_predictions is not None and len(cached_predictions) > 0

        # Get validation predictions from each base model
        if has_cache:
            logger.info(
                f"Using cached predictions from {len(cached_predictions)} base models "
                f"(skipping redundant predict calls)..."
            )
        elif use_multi_window:
            logger.info(
                f"Collecting predictions from {len(base_models)} base models "
                f"across {len(val_splits)} validation windows..."
            )
        else:
            logger.info(f"Collecting predictions from {len(base_models)} base models...")

        # Pre-resolve metric to detect quantile-aware scoring
        metric_name = self.get_hyperparameter("metric")
        scorer = get_metric(metric_name)
        _quantile_metrics = {"WQL", "SQL", "Coverage"}
        _scorer_name = getattr(scorer, "name", "")
        use_quantile_selection = _scorer_name in _quantile_metrics

        model_predictions: dict[str, dict[str, np.ndarray]] = {}
        multi_window_predictions: dict[str, list[dict[str, np.ndarray]]] | None = (
            {} if use_multi_window else None
        )

        q_levels = _ENSEMBLE_QUANTILE_LEVELS if use_quantile_selection else None

        for name, model in base_models.items():
            if has_cache and name in cached_predictions:
                # Use cached multi-window predictions
                mw_preds = cached_predictions[name]
                if use_multi_window and mw_preds:
                    multi_window_predictions[name] = mw_preds
                # Use last window as single-window fallback
                last_window_preds = mw_preds[-1] if mw_preds else {}
                if last_window_preds:
                    model_predictions[name] = last_window_preds
                else:
                    preds = _compute_per_item_predictions(
                        model, val_data, quantile_levels=q_levels)
                    if preds:
                        model_predictions[name] = preds
            else:
                preds = _compute_per_item_predictions(
                    model, val_data, quantile_levels=q_levels)
                if preds:
                    model_predictions[name] = preds
                    logger.debug(f"  {name}: {len(preds)} items predicted")
                else:
                    logger.warning(f"  {name}: failed, skipping")
                    continue

                if use_multi_window:
                    mw_preds = _compute_multi_window_predictions(model, val_splits)
                    multi_window_predictions[name] = mw_preds

        if not model_predictions:
            raise RuntimeError("No base model produced valid predictions.")

        # Store train tails for scale-dependent metrics
        train_tails = {}
        for item_id in val_data.item_ids:
            if item_id in train_data.item_ids:
                item_df = train_data.loc[item_id]
                train_tails[item_id] = item_df[TARGET].values[-100:]

        # Multi-window train tails
        train_tails_list = None
        if use_multi_window:
            train_tails_list = []
            for train_w, val_w in val_splits:
                tails = {}
                for item_id in val_w.item_ids:
                    if item_id in train_w.item_ids:
                        item_df = train_w.loc[item_id]
                        tails[item_id] = item_df[TARGET].values[-100:]
                train_tails_list.append(tails)

        # Greedy selection (scorer already resolved above)
        ensemble_size = self.get_hyperparameter("ensemble_size")

        n_windows_str = f", {len(val_splits)} windows" if use_multi_window else ""
        logger.info(
            f"Running greedy ensemble selection "
            f"(max_rounds={ensemble_size}, metric={metric_name}{n_windows_str})..."
        )
        self._weights, self._ensemble_score = greedy_ensemble_selection(
            model_predictions=model_predictions,
            val_data=val_data,
            prediction_length=self.prediction_length,
            scorer=scorer,
            ensemble_size=ensemble_size,
            train_tails=train_tails,
            multi_window_predictions=multi_window_predictions,
            val_splits=val_splits,
            train_tails_list=train_tails_list,
            quantile_levels=q_levels,
        )

        self._is_fitted = True
        self.fit_time = time.time() - start_time
        self._train_item_ids = list(train_data.item_ids)
        self._train_target_tail = train_tails

        logger.info(
            f"Ensemble fitted: {len(self._weights)} models selected, "
            f"score={self._ensemble_score:.4f}, time={self.fit_time:.1f}s"
        )
        for name, w in sorted(self._weights.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: weight={w:.3f}")

        return self

    def _fit(self, train_data, val_data=None, time_limit=None):
        """Not used — fit() is overridden."""
        pass

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        # Collect predictions from selected models (batch predict per model)
        model_preds: list[tuple[float, TimeSeriesDataFrame]] = []
        for name, weight in self._weights.items():
            if name in self._base_models:
                try:
                    pred = self._base_models[name].predict(
                        data, known_covariates=known_covariates,
                        quantile_levels=quantile_levels,
                    )
                    model_preds.append((weight, pred))
                except Exception as e:
                    logger.warning(f"Ensemble: {name} predict failed: {e}")

        if not model_preds:
            raise RuntimeError("No base model produced predictions.")

        # Vectorized weighted average using the first model's structure as base
        total_w = sum(w for w, _ in model_preds)
        cols_to_blend = ["mean"] + [str(q) for q in quantile_levels]
        result = model_preds[0][1].copy()

        for col in cols_to_blend:
            if col not in result.columns:
                continue
            blended = np.zeros(len(result), dtype=np.float64)
            w_sum = np.zeros(len(result), dtype=np.float64)
            for weight, pred_df in model_preds:
                if col in pred_df.columns:
                    blended += weight * pred_df[col].values
                    w_sum += weight
                elif "mean" in pred_df.columns:
                    blended += weight * pred_df["mean"].values
                    w_sum += weight
            w_sum = np.where(w_sum > 0, w_sum, 1.0)
            result[col] = blended / w_sum

        # Enforce quantile monotonicity to prevent crossing
        sorted_q_cols = [str(q) for q in sorted(quantile_levels)
                         if str(q) in result.columns]
        if len(sorted_q_cols) > 1:
            q_matrix = result[sorted_q_cols].values
            q_sorted = np.sort(q_matrix, axis=1)
            for i, col in enumerate(sorted_q_cols):
                result[col] = q_sorted[:, i]

        return result

    @property
    def weights(self) -> dict[str, float]:
        """Return the learned ensemble weights."""
        return dict(self._weights)

    @property
    def selected_models(self) -> list[str]:
        """Return names of models selected by greedy search."""
        return list(self._weights.keys())

    @property
    def ensemble_score(self) -> float:
        """Return the validation score achieved by the ensemble."""
        return self._ensemble_score

    def model_info(self) -> dict[str, Any]:
        info = super().model_info()
        info["weights"] = self.weights
        info["ensemble_score"] = self.ensemble_score
        info["n_base_models"] = len(self._base_models)
        info["n_selected_models"] = len(self._weights)
        return info

    def __repr__(self) -> str:
        if self._is_fitted:
            models = ", ".join(f"{n}={w:.2f}" for n, w in self._weights.items())
            return f"WeightedEnsemble(pred_len={self.prediction_length}, {models})"
        return f"WeightedEnsemble(pred_len={self.prediction_length}, not fitted)"


# ---------------------------------------------------------------------------
# SimpleAverage Ensemble (no val data needed)
# ---------------------------------------------------------------------------

@register_model("SimpleAverage")
class SimpleAverageEnsemble(AbstractTimeSeriesModel):
    """Simple average of base model predictions (equal weights).

    Unlike WeightedEnsemble, this does NOT require validation data.
    All base models get equal weight.

    Usage::

        ensemble = SimpleAverageEnsemble(freq="D", prediction_length=24)
        ensemble.fit(train_data, base_models=base_models)
    """

    _default_hyperparameters: dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_models: dict[str, AbstractTimeSeriesModel] = {}

    def fit(self, train_data, val_data=None, base_models=None, **kwargs):
        if base_models is None or len(base_models) == 0:
            raise ValueError("SimpleAverageEnsemble requires base_models dict.")

        self._base_models = base_models
        if self.freq is None:
            self.freq = train_data.freq
        self._is_fitted = True
        self.fit_time = 0.0
        self._train_item_ids = list(train_data.item_ids)

        # Store train tails
        self._train_target_tail = {}
        for item_id in train_data.item_ids:
            item_df = train_data.loc[item_id]
            self._train_target_tail[item_id] = item_df[TARGET].values[-100:]

        return self

    def _fit(self, train_data, val_data=None, time_limit=None):
        pass

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        # Collect all predictions (batch predict per model)
        pred_list: list[TimeSeriesDataFrame] = []
        for name, model in self._base_models.items():
            try:
                pred = model.predict(data, quantile_levels=quantile_levels)
                pred_list.append(pred)
            except Exception as e:
                logger.warning(f"SimpleAverage: {name} failed: {e}")

        if not pred_list:
            raise RuntimeError("No base model produced predictions.")

        # Vectorized simple average
        result = pred_list[0].copy()
        cols_to_blend = ["mean"] + [str(q) for q in quantile_levels]

        for col in cols_to_blend:
            if col not in result.columns:
                continue
            total = np.zeros(len(result), dtype=np.float64)
            count = 0
            for pred_df in pred_list:
                if col in pred_df.columns:
                    total += pred_df[col].values
                    count += 1
            result[col] = total / max(count, 1)

        # Enforce quantile monotonicity
        sorted_q_cols = [str(q) for q in sorted(quantile_levels)
                         if str(q) in result.columns]
        if len(sorted_q_cols) > 1:
            q_matrix = result[sorted_q_cols].values
            q_sorted = np.sort(q_matrix, axis=1)
            for i, col in enumerate(sorted_q_cols):
                result[col] = q_sorted[:, i]

        return result

    def __repr__(self):
        return f"SimpleAverageEnsemble(pred_len={self.prediction_length}, n_models={len(self._base_models)})"
