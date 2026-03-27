"""
Conformal Prediction — Calibrated prediction intervals.

Implements split conformal prediction for time series forecasting:

1. Compute non-conformity scores on a calibration set
2. Adjust quantile predictions using empirical quantiles of the scores

This ensures that prediction intervals have **guaranteed coverage**
(e.g., a 90% interval actually contains the true value ~90% of the time),
regardless of the underlying model's distributional assumptions.

Reference: Stankeviciute et al., "Conformal Time Series Forecasting" (NeurIPS 2021)

Usage::

    calibrator = ConformalCalibrator()
    calibrator.fit(predictor, calibration_data)
    calibrated_preds = calibrator.calibrate(raw_predictions, quantile_levels)

Or integrated into TimeSeriesPredictor via ``conformal=True`` in ``fit()``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from myforecaster.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """Split conformal prediction for time series.

    Computes per-horizon non-conformity scores on a calibration set,
    then adjusts quantile predictions to achieve nominal coverage.

    Parameters
    ----------
    symmetric : bool
        If True, use symmetric absolute residuals.
        If False, use separate upper/lower residuals (adaptive intervals).
    """

    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
        self._is_fitted = False
        # Per-horizon residual quantiles: shape (prediction_length, n_quantiles)
        self._residuals: np.ndarray | None = None
        self._prediction_length: int = 0

    def fit(
        self,
        y_true_per_horizon: dict[int, np.ndarray],
        y_pred_per_horizon: dict[int, np.ndarray],
        prediction_length: int,
    ) -> "ConformalCalibrator":
        """Compute conformity scores from calibration predictions.

        Parameters
        ----------
        y_true_per_horizon : dict
            {horizon_step: array of true values} from calibration set.
        y_pred_per_horizon : dict
            {horizon_step: array of predicted means} from calibration set.
        prediction_length : int
        """
        self._prediction_length = prediction_length

        # Compute absolute residuals per horizon
        residuals_per_h = {}
        for h in range(prediction_length):
            if h in y_true_per_horizon and h in y_pred_per_horizon:
                true = np.asarray(y_true_per_horizon[h])
                pred = np.asarray(y_pred_per_horizon[h])
                min_len = min(len(true), len(pred))
                if min_len > 0:
                    residuals_per_h[h] = np.abs(true[:min_len] - pred[:min_len])

        if not residuals_per_h:
            logger.warning("ConformalCalibrator: no valid residuals, skipping.")
            return self

        self._residuals = residuals_per_h
        self._is_fitted = True
        logger.info(
            f"ConformalCalibrator fitted: {prediction_length} horizons, "
            f"~{np.mean([len(r) for r in residuals_per_h.values()]):.0f} "
            f"calibration samples per horizon"
        )
        return self

    def fit_from_predictor(
        self,
        predictor,
        calibration_data: TimeSeriesDataFrame,
    ) -> "ConformalCalibrator":
        """Convenience: fit from a predictor and calibration data.

        Uses the predictor's best model to make predictions on the
        calibration set and computes conformity scores.

        Parameters
        ----------
        predictor : TimeSeriesPredictor
            A fitted predictor.
        calibration_data : TimeSeriesDataFrame
            Data with full series (context + future). Last prediction_length
            steps per item are used as ground truth.
        """
        pred_len = predictor.prediction_length

        # Build context (remove last pred_len) and get predictions
        context = calibration_data.slice_by_timestep(None, -pred_len)
        preds = predictor.predict(context, quantile_levels=[0.5])

        # Collect per-horizon residuals across all items
        y_true_per_h: dict[int, list] = {h: [] for h in range(pred_len)}
        y_pred_per_h: dict[int, list] = {h: [] for h in range(pred_len)}

        for item_id in calibration_data.item_ids:
            item_data = calibration_data.loc[item_id]
            n = len(item_data)
            if n <= pred_len:
                continue

            y_true = item_data[TARGET].values[-pred_len:]

            try:
                item_pred = preds.loc[item_id]
                y_pred = (
                    item_pred["mean"].values
                    if "mean" in preds.columns
                    else item_pred.iloc[:, 0].values
                )
            except (KeyError, IndexError):
                continue

            for h in range(min(pred_len, len(y_true), len(y_pred))):
                y_true_per_h[h].append(y_true[h])
                y_pred_per_h[h].append(y_pred[h])

        y_true_arrays = {h: np.array(v) for h, v in y_true_per_h.items() if v}
        y_pred_arrays = {h: np.array(v) for h, v in y_pred_per_h.items() if v}

        return self.fit(y_true_arrays, y_pred_arrays, pred_len)

    def calibrate(
        self,
        predictions: TimeSeriesDataFrame,
        quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> TimeSeriesDataFrame:
        """Adjust quantile predictions using conformal scores.

        For each quantile level q, the conformal interval is computed
        using the finite-sample-valid conformal quantile::

            conformal_q = ceil((n + 1) * level) / n

        which guarantees marginal coverage >= nominal level.

        After calibration, quantiles are **sorted** per-step to prevent
        crossing (e.g., ensuring q0.1 <= q0.5 <= q0.9).

        Parameters
        ----------
        predictions : TimeSeriesDataFrame
            Raw predictions with at least a ``mean`` column.
        quantile_levels : sequence of float
            Quantile levels to calibrate.

        Returns
        -------
        TimeSeriesDataFrame
            Calibrated predictions with adjusted quantile columns.
        """
        if not self._is_fitted or self._residuals is None:
            return predictions

        result = predictions.copy()
        sorted_qs = sorted(quantile_levels)

        for item_id in predictions.item_ids:
            try:
                item_pred = predictions.loc[item_id]
            except KeyError:
                continue

            mean_vals = (
                item_pred["mean"].values
                if "mean" in predictions.columns
                else item_pred.iloc[:, 0].values
            )

            for h in range(min(len(mean_vals), self._prediction_length)):
                if h not in self._residuals:
                    continue

                residuals = self._residuals[h]
                n = len(residuals)

                calibrated_vals = {}
                for q in quantile_levels:
                    q_str = str(q)
                    if q_str not in result.columns:
                        continue

                    if q < 0.5:
                        # Lower quantile: mean - Q(residuals, conformal_level)
                        conf_level = min(np.ceil((n + 1) * (1 - q)) / n, 1.0)
                        correction = np.quantile(residuals, conf_level)
                        calibrated_vals[q] = mean_vals[h] - correction
                    elif q > 0.5:
                        # Upper quantile: mean + Q(residuals, conformal_level)
                        conf_level = min(np.ceil((n + 1) * q) / n, 1.0)
                        correction = np.quantile(residuals, conf_level)
                        calibrated_vals[q] = mean_vals[h] + correction
                    else:
                        # Median: keep as is
                        calibrated_vals[q] = mean_vals[h]

                # Enforce monotonicity: sort calibrated quantiles to prevent crossing
                if len(calibrated_vals) > 1:
                    present_qs = sorted(q for q in sorted_qs if q in calibrated_vals)
                    sorted_vals = sorted(calibrated_vals[q] for q in present_qs)
                    for q, val in zip(present_qs, sorted_vals):
                        calibrated_vals[q] = val

                # Write back
                idx = result.loc[item_id].index[h]
                for q, val in calibrated_vals.items():
                    result.at[(item_id, idx), str(q)] = val

        return result

    @property
    def coverage_adjustments(self) -> dict[int, float]:
        """Return the median absolute residual per horizon (for diagnostics)."""
        if not self._is_fitted or self._residuals is None:
            return {}
        return {
            h: float(np.median(r)) for h, r in self._residuals.items()
        }
