"""Lightweight uncertainty helpers for ensemble QSAR predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


def ensemble_predict(
    members: Sequence[Any] | np.ndarray | pd.DataFrame,
    X: Any | None = None,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ddof: int = 0,
    include_members: bool = False,
) -> pd.DataFrame:
    """Run an ensemble or summarize precomputed member predictions.

    If ``X`` is provided, each member must expose ``predict(X)`` or be callable.
    If ``X`` is omitted, ``members`` is interpreted as prediction arrays with
    shape ``(n_members, n_samples)``. For pandas inputs, columns are treated as
    members and rows as samples.
    """

    if X is None:
        member_matrix = _as_member_prediction_matrix(members)
    else:
        member_matrix = _predict_members(members, X)

    summary = summarize_uncertainty(member_matrix, quantiles=quantiles, ddof=ddof)
    if include_members:
        for member_index, predictions in enumerate(member_matrix):
            summary[f"member_{member_index:02d}_prediction"] = predictions
    return summary


def summarize_uncertainty(
    member_predictions: Sequence[Any] | np.ndarray | pd.DataFrame,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ddof: int = 0,
) -> pd.DataFrame:
    """Summarize per-sample ensemble uncertainty from member predictions."""

    if ddof < 0:
        raise ValueError("ddof must be greater than or equal to 0.")
    checked_quantiles = _validate_quantiles(quantiles)
    matrix = _as_member_prediction_matrix(member_predictions)
    n_members, n_samples = matrix.shape

    means = np.full(n_samples, np.nan, dtype=float)
    stds = np.full(n_samples, np.nan, dtype=float)
    mins = np.full(n_samples, np.nan, dtype=float)
    maxs = np.full(n_samples, np.nan, dtype=float)
    finite_counts = np.isfinite(matrix).sum(axis=0)
    quantile_values = {
        quantile: np.full(n_samples, np.nan, dtype=float)
        for quantile in checked_quantiles
    }

    for sample_index in range(n_samples):
        values = matrix[:, sample_index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        means[sample_index] = float(np.mean(finite))
        mins[sample_index] = float(np.min(finite))
        maxs[sample_index] = float(np.max(finite))
        if finite.size > ddof:
            stds[sample_index] = float(np.std(finite, ddof=ddof))
        for quantile in checked_quantiles:
            quantile_values[quantile][sample_index] = float(np.quantile(finite, quantile))

    summary = pd.DataFrame(
        {
            "prediction_mean": means,
            "prediction_std": stds,
            "prediction_var": stds**2,
            "prediction_min": mins,
            "prediction_max": maxs,
            "prediction_n_members": n_members,
            "prediction_n_finite_members": finite_counts.astype(int),
        }
    )
    for quantile in checked_quantiles:
        summary[_quantile_column(quantile)] = quantile_values[quantile]

    if 0.05 in checked_quantiles and 0.95 in checked_quantiles:
        summary["prediction_interval_width_90"] = (
            summary["prediction_q95"] - summary["prediction_q05"]
        )
    return summary


def calibrate_intervals(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    predictions: Sequence[float] | np.ndarray | pd.Series | pd.DataFrame,
    *,
    lower: Sequence[float] | np.ndarray | pd.Series | None = None,
    upper: Sequence[float] | np.ndarray | pd.Series | None = None,
    coverage: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calibrate prediction intervals with a split-conformal residual offset."""

    _validate_coverage(coverage)
    y = _as_1d_float_array(y_true, "y_true")
    point, inferred_lower, inferred_upper, index = _extract_prediction_inputs(
        predictions,
        coverage=coverage,
    )
    if lower is not None:
        inferred_lower = _as_1d_float_array(lower, "lower")
    if upper is not None:
        inferred_upper = _as_1d_float_array(upper, "upper")

    _validate_same_length(y, point, "predictions")
    has_base_interval = inferred_lower is not None and inferred_upper is not None
    if has_base_interval:
        base_lower = inferred_lower
        base_upper = inferred_upper
        _validate_same_length(y, base_lower, "lower")
        _validate_same_length(y, base_upper, "upper")
        scores = np.maximum.reduce(
            [
                base_lower - y,
                y - base_upper,
                np.zeros_like(y, dtype=float),
            ]
        )
    else:
        base_lower = point.copy()
        base_upper = point.copy()
        scores = np.abs(y - point)

    finite_mask = (
        np.isfinite(y)
        & np.isfinite(point)
        & np.isfinite(base_lower)
        & np.isfinite(base_upper)
        & np.isfinite(scores)
    )
    if not finite_mask.any():
        raise ValueError("No finite calibration rows are available.")

    offset = _conformal_quantile(scores[finite_mask], coverage)
    calibrated_lower = base_lower - offset
    calibrated_upper = base_upper + offset
    before_covered = (y >= base_lower) & (y <= base_upper) & finite_mask
    after_covered = (y >= calibrated_lower) & (y <= calibrated_upper) & finite_mask

    calibrated = pd.DataFrame(
        {
            "prediction": point,
            "interval_lower": calibrated_lower,
            "interval_upper": calibrated_upper,
            "interval_width": calibrated_upper - calibrated_lower,
            "interval_half_width": (calibrated_upper - calibrated_lower) / 2.0,
            "calibration_offset": offset,
            "covered_by_calibrated_interval": after_covered,
        },
        index=index,
    )
    report = {
        "method": "split_conformal_absolute_residual",
        "target_coverage": coverage,
        "n_calibration": int(finite_mask.sum()),
        "calibration_offset": float(offset),
        "base_interval_provided": has_base_interval,
        "empirical_coverage_before": float(before_covered[finite_mask].mean()),
        "empirical_coverage_after": float(after_covered[finite_mask].mean()),
    }
    return calibrated, report


def _predict_members(members: Sequence[Any], X: Any) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for member in members:
        if hasattr(member, "predict"):
            predicted = member.predict(X)
        elif callable(member):
            predicted = member(X)
        else:
            raise TypeError("Each ensemble member must be callable or expose predict(X).")
        predictions.append(_as_1d_float_array(predicted, "member prediction"))
    if not predictions:
        raise ValueError("At least one ensemble member is required.")
    return np.vstack(predictions)


def _as_member_prediction_matrix(values: Any) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        matrix = values.to_numpy(dtype=float).T
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("member_predictions must be a 1D or 2D numeric array.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("member_predictions must contain at least one member and one sample.")
    return matrix.astype(float, copy=False)


def _extract_prediction_inputs(
    predictions: Sequence[float] | np.ndarray | pd.Series | pd.DataFrame,
    *,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, pd.Index | None]:
    if isinstance(predictions, pd.DataFrame):
        if "prediction_mean" in predictions.columns:
            point = predictions["prediction_mean"].to_numpy(dtype=float)
        elif "prediction" in predictions.columns:
            point = predictions["prediction"].to_numpy(dtype=float)
        else:
            numeric_columns = predictions.select_dtypes(include="number").columns
            if len(numeric_columns) == 0:
                raise ValueError("prediction DataFrame must contain numeric predictions.")
            point = predictions[numeric_columns[0]].to_numpy(dtype=float)

        lower_column, upper_column = _infer_interval_columns(predictions, coverage)
        lower = predictions[lower_column].to_numpy(dtype=float) if lower_column else None
        upper = predictions[upper_column].to_numpy(dtype=float) if upper_column else None
        return point, lower, upper, predictions.index

    return _as_1d_float_array(predictions, "predictions"), None, None, None


def _infer_interval_columns(
    predictions: pd.DataFrame,
    coverage: float,
) -> tuple[str | None, str | None]:
    if {"prediction_lower", "prediction_upper"}.issubset(predictions.columns):
        return "prediction_lower", "prediction_upper"
    alpha = 1.0 - coverage
    lower_column = _quantile_column(alpha / 2.0)
    upper_column = _quantile_column(1.0 - alpha / 2.0)
    if {lower_column, upper_column}.issubset(predictions.columns):
        return lower_column, upper_column
    return None, None


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D numeric array.")
    return array


def _validate_same_length(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{name} must have length {len(left)}.")


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    checked = tuple(float(quantile) for quantile in quantiles)
    if not checked:
        raise ValueError("At least one quantile is required.")
    for quantile in checked:
        if not 0 <= quantile <= 1:
            raise ValueError("quantiles must be between 0 and 1.")
    return checked


def _validate_coverage(coverage: float) -> None:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1.")


def _conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    finite_scores = np.sort(scores[np.isfinite(scores)])
    if finite_scores.size == 0:
        raise ValueError("No finite residual scores are available.")
    quantile_level = math.ceil((finite_scores.size + 1) * coverage) / finite_scores.size
    quantile_level = min(quantile_level, 1.0)
    return float(np.quantile(finite_scores, quantile_level, method="higher"))


def _quantile_column(quantile: float) -> str:
    percent = quantile * 100
    if abs(percent - round(percent)) < 1e-8:
        label = f"{int(round(percent)):02d}"
    else:
        label = f"{percent:.1f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"prediction_q{label}"
