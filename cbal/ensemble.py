"""
Ensemble methods for combining multiple forecasting models.

Two strategies:
1. **SimpleWeightedEnsemble**: uniform or user-specified weights
2. **GreedyEnsembleSelection**: AutoGluon-style greedy forward selection
   that iteratively adds models to minimize validation loss.

Usage::

    from cbal.ensemble import GreedyEnsembleSelection

    ensemble = GreedyEnsembleSelection(
        models=[model_a, model_b, model_c],
        prediction_length=7,
        freq="D",
    )
    ensemble.fit(train_data, val_data=val_data)
    pred = ensemble.predict(test_data)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cbal.dataset.ts_dataframe import (
    ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame,
)
from cbal.metrics.scorers import get_metric, TimeSeriesScorer
from cbal.models.abstract_model import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimpleWeightedEnsemble
# ---------------------------------------------------------------------------

class SimpleWeightedEnsemble(AbstractTimeSeriesModel):
    """Combine predictions from multiple fitted models with static weights.

    Parameters
    ----------
    models : list of AbstractTimeSeriesModel
        Already-fitted models.
    weights : list of float or None
        Per-model weights (must sum to 1). If None, uniform weights.
    prediction_length : int
    freq : str
    """

    _default_hyperparameters: dict[str, Any] = {}

    def __init__(
        self,
        models: list[AbstractTimeSeriesModel],
        weights: list[float] | None = None,
        prediction_length: int = 1,
        freq: str | None = None,
        **kwargs,
    ):
        super().__init__(prediction_length=prediction_length, freq=freq, **kwargs)
        self.models = models
        n = len(models)

        if weights is not None:
            if len(weights) != n:
                raise ValueError(f"weights length {len(weights)} != models length {n}")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / n] * n

        self._name = "SimpleWeightedEnsemble"

    def _fit(self, train_data, val_data=None, time_limit=None):
        """No-op — assumes models are already fitted."""
        # Verify all models are fitted
        for m in self.models:
            if not m._is_fitted:
                raise RuntimeError(
                    f"Model {m.name} is not fitted. "
                    "Fit all models before creating the ensemble."
                )
        # Copy train tails from first model
        if self.models:
            self._train_target_tail = self.models[0]._train_target_tail.copy()

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        # Collect predictions from all models
        all_preds = []
        for model in self.models:
            pred = model.predict(data, quantile_levels=quantile_levels)
            all_preds.append(pred)

        # Weighted average
        return self._weighted_average(all_preds, self.weights, quantile_levels)

    def _weighted_average(
        self,
        predictions: list[TimeSeriesDataFrame],
        weights: list[float],
        quantile_levels: list[float],
    ) -> TimeSeriesDataFrame:
        """Compute weighted average of predictions with monotonicity enforcement."""
        base = predictions[0].copy()

        # Average "mean" column
        mean_vals = np.zeros(len(base))
        for pred, w in zip(predictions, weights):
            mean_vals += w * pred["mean"].values
        base["mean"] = mean_vals

        # Average quantile columns
        for q in quantile_levels:
            col = str(q)
            if col in base.columns:
                q_vals = np.zeros(len(base))
                for pred, w in zip(predictions, weights):
                    if col in pred.columns:
                        q_vals += w * pred[col].values
                    else:
                        q_vals += w * pred["mean"].values
                base[col] = q_vals

        # Enforce quantile monotonicity to prevent crossing
        sorted_q_cols = [str(q) for q in sorted(quantile_levels)]
        if len(sorted_q_cols) > 1:
            q_matrix = base[sorted_q_cols].values  # (n_rows, n_quantiles)
            q_sorted = np.sort(q_matrix, axis=1)
            for i, col in enumerate(sorted_q_cols):
                base[col] = q_sorted[:, i]

        return base

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def summary(self) -> pd.DataFrame:
        """Return model names and their weights."""
        return pd.DataFrame({
            "model": self.model_names,
            "weight": self.weights,
        })


# ---------------------------------------------------------------------------
# GreedyEnsembleSelection
# ---------------------------------------------------------------------------

class GreedyEnsembleSelection(AbstractTimeSeriesModel):
    """AutoGluon-style greedy forward ensemble selection.

    Algorithm (Caruana et al., 2004):
    1. Start with empty ensemble
    2. At each step, try adding each candidate model
    3. Pick the model that minimizes validation loss when averaged in
    4. Repeat for ``max_models`` steps (with replacement)
    5. Final weights = frequency of selection / total steps

    Parameters
    ----------
    models : list of AbstractTimeSeriesModel
        Already-fitted candidate models.
    prediction_length : int
    freq : str
    max_models : int
        Maximum ensemble size (with replacement). Default 25.
    metric : str or TimeSeriesScorer
        Metric to minimize on validation set. Default "MAE".
    """

    _default_hyperparameters: dict[str, Any] = {
        "max_models": 25,
        "metric": "MAE",
    }

    def __init__(
        self,
        models: list[AbstractTimeSeriesModel],
        prediction_length: int = 1,
        freq: str | None = None,
        **kwargs,
    ):
        super().__init__(prediction_length=prediction_length, freq=freq, **kwargs)
        self.models = models
        self._selected_indices: list[int] = []
        self._weights: list[float] = []
        self._val_score: float | None = None
        self._name = "GreedyEnsembleSelection"

    def _fit(self, train_data, val_data=None, time_limit=None):
        """Run greedy selection on validation data.

        If val_data is None, uses last prediction_length of train_data.
        """
        # Verify all models are fitted
        for m in self.models:
            if not m._is_fitted:
                raise RuntimeError(f"Model {m.name} is not fitted.")

        # Copy train tails
        if self.models:
            self._train_target_tail = self.models[0]._train_target_tail.copy()

        # If no val_data, split train_data
        if val_data is None:
            val_data = train_data
            # We'll use the score() method which splits internally

        metric = get_metric(self.get_hyperparameter("metric"))
        max_models = self.get_hyperparameter("max_models")

        n_models = len(self.models)
        if n_models == 0:
            raise ValueError("No candidate models provided.")

        if n_models == 1:
            self._selected_indices = [0]
            self._weights = [1.0]
            logger.info(f"  Only 1 candidate model — using {self.models[0].name}")
            return

        # Pre-compute predictions from all candidates on val context
        logger.info(f"  Pre-computing predictions from {n_models} candidates...")
        val_preds = []
        val_scores = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(
                    self._get_context(val_data), quantile_levels=[0.5]
                )
                val_preds.append(pred)
                s = model.score(val_data, metric=metric)
                val_scores.append(s)
                logger.info(f"    {model.name}: {metric.name}={s:.4f}")
            except Exception as e:
                logger.warning(f"    {model.name} failed: {e}")
                val_preds.append(None)
                val_scores.append(float("inf") if metric.greater_is_better else float("inf"))

        # Pre-filter: exclude models with score > 5x the best valid score
        # Conservative threshold to only remove clearly broken models
        valid_scores = [s for s in val_scores if s < float("inf")]
        if valid_scores:
            best_individual = min(valid_scores)
            threshold = best_individual * 5.0
            for i in range(n_models):
                if val_scores[i] > threshold:
                    logger.info(f"    Excluding {self.models[i].name} "
                                f"(score={val_scores[i]:.4f} > threshold={threshold:.4f})")
                    val_preds[i] = None  # exclude from ensemble

        # Greedy forward selection
        logger.info(f"  Running greedy selection (max_models={max_models})...")
        selected = []
        best_score = float("inf")

        for step in range(max_models):
            best_candidate = -1
            best_step_score = float("inf")

            for i in range(n_models):
                if val_preds[i] is None:
                    continue

                # Try adding model i to current ensemble
                trial = selected + [i]
                trial_weights = self._compute_weights(trial)

                # Compute ensemble prediction
                trial_preds = [val_preds[j] for j in trial]
                trial_w = [trial_weights[j] for j in range(len(trial))]
                ensemble_pred = self._blend_predictions(trial_preds, trial_w)

                # Score
                score = self._evaluate_blend(ensemble_pred, val_data, metric)

                if score < best_step_score:
                    best_step_score = score
                    best_candidate = i

            if best_candidate >= 0:
                selected.append(best_candidate)
                if best_step_score < best_score:
                    best_score = best_step_score
                logger.debug(
                    f"    Step {step+1}: added {self.models[best_candidate].name} "
                    f"→ {metric.name}={best_step_score:.4f}"
                )

        self._selected_indices = selected
        self._weights = self._compute_final_weights(selected, n_models)
        self._val_score = best_score

        # Log final ensemble
        active = [(self.models[i].name, self._weights[i])
                  for i in range(n_models) if self._weights[i] > 0]
        logger.info(f"  Final ensemble ({len(active)} models, {metric.name}={best_score:.4f}):")
        for name, w in sorted(active, key=lambda x: -x[1]):
            logger.info(f"    {name}: {w:.3f}")

    def _get_context(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Get context (everything except last prediction_length) from data."""
        frames = []
        for item_id in data.item_ids:
            item_df = data.loc[[item_id]]
            n = len(item_df)
            if n > self.prediction_length:
                frames.append(item_df.iloc[:n - self.prediction_length])
        if not frames:
            return data
        result = TimeSeriesDataFrame(pd.concat(frames))
        result._cached_freq = self.freq
        return result

    def _compute_weights(self, indices: list[int]) -> list[float]:
        """Compute uniform weights over selected models (with replacement)."""
        n = len(indices)
        return [1.0 / n] * n

    def _compute_final_weights(self, indices: list[int], n_models: int) -> list[float]:
        """Frequency-based weights from selection history."""
        counts = np.zeros(n_models)
        for idx in indices:
            counts[idx] += 1
        total = counts.sum()
        if total == 0:
            return [1.0 / n_models] * n_models
        return (counts / total).tolist()

    def _blend_predictions(
        self,
        predictions: list[TimeSeriesDataFrame],
        weights: list[float],
    ) -> TimeSeriesDataFrame:
        """Weighted average of predictions."""
        base = predictions[0].copy()
        mean_vals = np.zeros(len(base))
        for pred, w in zip(predictions, weights):
            mean_vals += w * pred["mean"].values
        base["mean"] = mean_vals
        return base

    def _evaluate_blend(
        self,
        ensemble_pred: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame,
        metric: TimeSeriesScorer,
    ) -> float:
        """Evaluate blended prediction against validation ground truth."""
        scores = []
        for item_id in val_data.item_ids:
            item_data = val_data.loc[item_id]
            n = len(item_data)
            if n <= self.prediction_length:
                continue
            y_true = item_data[TARGET].values[-self.prediction_length:]

            # Get ensemble prediction for this item
            item_pred = ensemble_pred.loc[item_id]
            y_pred = item_pred["mean"].values

            min_len = min(len(y_true), len(y_pred))
            y_train = self._train_target_tail.get(item_id)
            score = metric(y_true[:min_len], y_pred[:min_len], y_train=y_train)
            scores.append(score)

        return float(np.mean(scores)) if scores else float("inf")

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        # Only use models with non-zero weight
        active_models = []
        active_weights = []
        for i, w in enumerate(self._weights):
            if w > 0:
                active_models.append(self.models[i])
                active_weights.append(w)

        if not active_models:
            # Fallback: use all models equally
            active_models = self.models
            active_weights = [1.0 / len(self.models)] * len(self.models)

        # Normalize weights
        total = sum(active_weights)
        active_weights = [w / total for w in active_weights]

        # Collect predictions
        all_preds = []
        for model in active_models:
            pred = model.predict(data, quantile_levels=quantile_levels)
            all_preds.append(pred)

        return self._weighted_average(all_preds, active_weights, quantile_levels)

    def _weighted_average(self, predictions, weights, quantile_levels):
        base = predictions[0].copy()
        mean_vals = np.zeros(len(base))
        for pred, w in zip(predictions, weights):
            mean_vals += w * pred["mean"].values
        base["mean"] = mean_vals

        for q in quantile_levels:
            col = str(q)
            q_vals = np.zeros(len(base))
            for pred, w in zip(predictions, weights):
                if col in pred.columns:
                    q_vals += w * pred[col].values
                else:
                    q_vals += w * pred["mean"].values
            base[col] = q_vals

        # Enforce quantile monotonicity to prevent crossing
        sorted_q_cols = [str(q) for q in sorted(quantile_levels)]
        if len(sorted_q_cols) > 1:
            q_matrix = base[sorted_q_cols].values
            q_sorted = np.sort(q_matrix, axis=1)
            for i, col in enumerate(sorted_q_cols):
                base[col] = q_sorted[:, i]

        return base

    @property
    def model_weights(self) -> dict[str, float]:
        """Return {model_name: weight} for active models."""
        return {
            self.models[i].name: self._weights[i]
            for i in range(len(self.models))
            if self._weights[i] > 0
        }

    def summary(self) -> pd.DataFrame:
        """Return selection summary."""
        return pd.DataFrame({
            "model": [m.name for m in self.models],
            "weight": self._weights,
            "selected_count": [
                self._selected_indices.count(i)
                for i in range(len(self.models))
            ],
        }).sort_values("weight", ascending=False)
