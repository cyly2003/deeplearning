"""Evaluation metrics for QSAR regression tasks."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
    prediction_interval_low: Any | None = None,
    prediction_interval_high: Any | None = None,
    interval_confidence: float = 0.90,
    prefix: str | None = None,
) -> dict[str, float | int]:
    """Return core regression metrics with NaN-safe filtering.

    Metrics are computed on rows where true and predicted values are finite.
    MAPE excludes rows with true values numerically equal to zero.
    """

    true = _as_float_array(y_true, "y_true")
    pred = _as_float_array(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    weights = None if sample_weight is None else _as_float_array(sample_weight, "sample_weight")
    if weights is not None and weights.shape != true.shape:
        raise ValueError("sample_weight must have the same shape as y_true.")

    finite_mask = np.isfinite(true) & np.isfinite(pred)
    if weights is not None:
        finite_mask &= np.isfinite(weights)
    true_valid = true[finite_mask]
    pred_valid = pred[finite_mask]
    weights_valid = None if weights is None else weights[finite_mask]

    metrics: dict[str, float | int] = {
        "n": int(true_valid.size),
        "r2": _r2_score(true_valid, pred_valid, weights_valid),
        "rmse": _rmse(true_valid, pred_valid, weights_valid),
        "mae": _mae(true_valid, pred_valid, weights_valid),
        "mape": _mape(true_valid, pred_valid, weights_valid),
        "spearman_rho": _spearman_rho(true_valid, pred_valid),
        "bias": _weighted_mean(pred_valid - true_valid, weights_valid),
    }

    if prediction_interval_low is not None and prediction_interval_high is not None:
        low = _as_float_array(prediction_interval_low, "prediction_interval_low")
        high = _as_float_array(prediction_interval_high, "prediction_interval_high")
        if low.shape != true.shape or high.shape != true.shape:
            raise ValueError("prediction intervals must have the same shape as y_true.")
        interval_metrics = _interval_metrics(
            true,
            low,
            high,
            finite_mask,
            interval_confidence,
            weights_valid,
        )
        metrics.update(interval_metrics)

    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def compute_regression_metrics(*args: Any, **kwargs: Any) -> dict[str, float | int]:
    """Compatibility alias for regression_metrics."""

    return regression_metrics(*args, **kwargs)


def _as_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return array


def _r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    if y_true.size < 2:
        return math.nan
    residual = y_true - y_pred
    if sample_weight is None:
        y_mean = float(np.mean(y_true))
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))
    else:
        y_mean = float(np.average(y_true, weights=sample_weight))
        ss_res = float(np.sum(sample_weight * residual**2))
        ss_tot = float(np.sum(sample_weight * (y_true - y_mean) ** 2))
    if ss_tot <= 0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def _rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    if y_true.size == 0:
        return math.nan
    return math.sqrt(_weighted_mean((y_true - y_pred) ** 2, sample_weight))


def _mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    if y_true.size == 0:
        return math.nan
    return _weighted_mean(np.abs(y_true - y_pred), sample_weight)


def _mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    if y_true.size == 0:
        return math.nan
    non_zero = np.abs(y_true) > 1.0e-12
    if not np.any(non_zero):
        return math.nan
    weights = None if sample_weight is None else sample_weight[non_zero]
    return _weighted_mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]), weights)


def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return math.nan
    true_ranks = pd.Series(y_true).rank(method="average").to_numpy(dtype="float64")
    pred_ranks = pd.Series(y_pred).rank(method="average").to_numpy(dtype="float64")
    true_centered = true_ranks - float(np.mean(true_ranks))
    pred_centered = pred_ranks - float(np.mean(pred_ranks))
    denominator = math.sqrt(float(np.sum(true_centered**2) * np.sum(pred_centered**2)))
    if denominator <= 0:
        return math.nan
    return float(np.sum(true_centered * pred_centered) / denominator)


def _interval_metrics(
    y_true: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    base_mask: np.ndarray,
    interval_confidence: float,
    sample_weight: np.ndarray | None,
) -> Mapping[str, float]:
    interval_mask = base_mask & np.isfinite(low) & np.isfinite(high)
    true_valid = y_true[interval_mask]
    low_valid = low[interval_mask]
    high_valid = high[interval_mask]
    if true_valid.size == 0:
        return {
            "interval_coverage": math.nan,
            "calibration_error": math.nan,
            "mean_prediction_interval_width": math.nan,
        }

    weights = None
    if sample_weight is not None:
        weights = sample_weight[interval_mask[base_mask]]
    covered = (true_valid >= low_valid) & (true_valid <= high_valid)
    coverage = _weighted_mean(covered.astype("float64"), weights)
    width = _weighted_mean(high_valid - low_valid, weights)
    return {
        "interval_coverage": coverage,
        "calibration_error": abs(coverage - float(interval_confidence)),
        "mean_prediction_interval_width": width,
    }


def _weighted_mean(values: np.ndarray, sample_weight: np.ndarray | None) -> float:
    if values.size == 0:
        return math.nan
    if sample_weight is None:
        return float(np.mean(values))
    weight_sum = float(np.sum(sample_weight))
    if weight_sum <= 0 or not math.isfinite(weight_sum):
        return math.nan
    return float(np.average(values, weights=sample_weight))
