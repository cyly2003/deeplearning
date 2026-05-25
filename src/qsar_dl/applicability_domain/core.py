"""Applicability-domain helpers for QSAR transfer checks.

These functions intentionally use lightweight tabular checks. They are meant
for audit features and early screening before a calibrated AD model exists.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TAXONOMY_COLUMNS = (
    "kingdom",
    "phylum_division",
    "class",
    "tax_order",
    "family",
    "genus",
    "species",
    "species_ecotox_group",
)
FINGERPRINT_PREFIXES = ("morgan_fp_", "fp_", "fingerprint_")


def compute_descriptor_range_ad(
    reference_table: pd.DataFrame,
    query_table: pd.DataFrame,
    descriptor_columns: Sequence[str] | None = None,
    *,
    tolerance: float = 0.0,
    min_score: float = 1.0,
) -> pd.DataFrame:
    """Score query chemicals by whether descriptors fall inside reference ranges.

    Parameters
    ----------
    reference_table:
        Training/reference descriptor table.
    query_table:
        Query descriptor table to score.
    descriptor_columns:
        Descriptor columns to audit. When omitted, common numeric non-fingerprint
        columns are used.
    tolerance:
        Fractional expansion of each reference range. A value of 0.05 allows a
        5% margin on both sides of the reference min/max span.
    min_score:
        Minimum descriptor AD score required for ``descriptor_ad_in_domain``.
    """

    _require_dataframe(reference_table, "reference_table")
    _require_dataframe(query_table, "query_table")
    if tolerance < 0:
        raise ValueError("tolerance must be greater than or equal to 0.")
    _validate_fraction(min_score, "min_score")

    columns = _resolve_descriptor_columns(reference_table, query_table, descriptor_columns)
    reference_ranges = _descriptor_ranges(reference_table, columns, tolerance)
    reference_count = sum(1 for bounds in reference_ranges.values() if bounds is not None)

    records: list[dict[str, object]] = []
    for _idx, row in query_table.iterrows():
        in_range = 0
        out_of_range = 0
        missing = 0
        evaluated = 0
        missing_columns: list[str] = []
        out_of_range_columns: list[str] = []
        reference_missing_columns: list[str] = []

        for column in columns:
            bounds = reference_ranges[column]
            if bounds is None:
                reference_missing_columns.append(column)
                continue

            value = _to_float(row.get(column))
            if not _is_finite(value):
                missing += 1
                missing_columns.append(column)
                continue

            evaluated += 1
            lower, upper = bounds
            if lower <= float(value) <= upper:
                in_range += 1
            else:
                out_of_range += 1
                out_of_range_columns.append(column)

        available_fraction = _safe_divide(evaluated, reference_count)
        in_range_fraction = _safe_divide(in_range, evaluated)
        if math.isnan(available_fraction) or math.isnan(in_range_fraction):
            score = np.nan
        else:
            score = available_fraction * in_range_fraction

        records.append(
            {
                "descriptor_ad_applicable": reference_count > 0,
                "descriptor_ad_in_domain": bool(_is_finite(score) and score >= min_score),
                "descriptor_ad_score": score,
                "descriptor_reference_count": reference_count,
                "descriptor_evaluated_count": evaluated,
                "descriptor_in_range_count": in_range,
                "descriptor_out_of_range_count": out_of_range,
                "descriptor_missing_count": missing,
                "descriptor_available_fraction": available_fraction,
                "descriptor_in_range_fraction": in_range_fraction,
                "descriptor_missing_columns": ";".join(missing_columns),
                "descriptor_out_of_range_columns": ";".join(out_of_range_columns),
                "descriptor_reference_missing_columns": ";".join(reference_missing_columns),
            }
        )

    return pd.DataFrame.from_records(records, index=query_table.index)


def compute_fingerprint_tanimoto_ad(
    reference_table: pd.DataFrame,
    query_table: pd.DataFrame,
    fingerprint_columns: Sequence[str] | None = None,
    *,
    tanimoto_threshold: float = 0.35,
) -> pd.DataFrame:
    """Score query chemicals by nearest simplified binary Tanimoto similarity."""

    _require_dataframe(reference_table, "reference_table")
    _require_dataframe(query_table, "query_table")
    _validate_fraction(tanimoto_threshold, "tanimoto_threshold")

    columns = _resolve_fingerprint_columns(reference_table, query_table, fingerprint_columns)
    if not columns:
        return _empty_fingerprint_ad(query_table.index)

    reference_bits, reference_finite = _binary_matrix(reference_table, columns)
    query_bits, query_finite = _binary_matrix(query_table, columns)
    reference_valid = (reference_finite.sum(axis=1) > 0) & (reference_bits.sum(axis=1) > 0)
    valid_reference_bits = reference_bits[reference_valid]

    records: list[dict[str, object]] = []
    for row_bits, row_finite in zip(query_bits, query_finite):
        query_valid = bool(row_finite.sum() > 0 and row_bits.sum() > 0)
        if not query_valid or valid_reference_bits.size == 0:
            records.append(
                {
                    "fingerprint_ad_applicable": False,
                    "fingerprint_ad_in_domain": False,
                    "fingerprint_ad_score": np.nan,
                    "fingerprint_max_tanimoto": np.nan,
                    "fingerprint_nearest_reference_position": None,
                    "fingerprint_reference_count": int(reference_valid.sum()),
                    "fingerprint_bit_count": int(row_bits.sum()),
                }
            )
            continue

        similarities = _tanimoto_against_reference(row_bits, valid_reference_bits)
        best_position = int(np.nanargmax(similarities))
        best_similarity = float(similarities[best_position])
        original_positions = np.flatnonzero(reference_valid)
        records.append(
            {
                "fingerprint_ad_applicable": True,
                "fingerprint_ad_in_domain": best_similarity >= tanimoto_threshold,
                "fingerprint_ad_score": best_similarity,
                "fingerprint_max_tanimoto": best_similarity,
                "fingerprint_nearest_reference_position": int(original_positions[best_position]),
                "fingerprint_reference_count": int(reference_valid.sum()),
                "fingerprint_bit_count": int(row_bits.sum()),
            }
        )

    return pd.DataFrame.from_records(records, index=query_table.index)


def compute_chemical_ad(
    reference_table: pd.DataFrame,
    query_table: pd.DataFrame,
    descriptor_columns: Sequence[str] | None = None,
    fingerprint_columns: Sequence[str] | None = None,
    *,
    descriptor_tolerance: float = 0.0,
    descriptor_min_score: float = 1.0,
    tanimoto_threshold: float = 0.35,
) -> pd.DataFrame:
    """Combine descriptor range and fingerprint similarity chemical AD checks."""

    descriptor_columns_used = _resolve_descriptor_columns(
        reference_table,
        query_table,
        descriptor_columns,
    )
    fingerprint_columns_used = _resolve_fingerprint_columns(
        reference_table,
        query_table,
        fingerprint_columns,
    )
    descriptor_ad = compute_descriptor_range_ad(
        reference_table,
        query_table,
        descriptor_columns_used,
        tolerance=descriptor_tolerance,
        min_score=descriptor_min_score,
    )
    fingerprint_ad = compute_fingerprint_tanimoto_ad(
        reference_table,
        query_table,
        fingerprint_columns_used,
        tanimoto_threshold=tanimoto_threshold,
    )

    combined = pd.concat([descriptor_ad, fingerprint_ad], axis=1)
    expected_components = [
        ("descriptor", bool(descriptor_columns_used)),
        ("fingerprint", bool(fingerprint_columns_used)),
    ]
    component_count = sum(1 for _name, enabled in expected_components if enabled)
    scores: list[float] = []
    in_domain: list[bool] = []
    for row_index, row in combined.iterrows():
        row_scores: list[float] = []
        row_domains: list[bool] = []
        if descriptor_columns_used:
            row_scores.append(_finite_or_zero(row.get("descriptor_ad_score")))
            row_domains.append(bool(row.get("descriptor_ad_in_domain", False)))
        if fingerprint_columns_used:
            row_scores.append(_finite_or_zero(row.get("fingerprint_ad_score")))
            row_domains.append(bool(row.get("fingerprint_ad_in_domain", False)))

        scores.append(float(np.mean(row_scores)) if row_scores else np.nan)
        in_domain.append(bool(row_domains and all(row_domains)))

    combined["chemical_ad_component_count"] = component_count
    combined["chemical_ad_score"] = scores
    combined["chemical_ad_in_domain"] = in_domain
    combined["chemical_ad_applicable"] = component_count > 0
    return combined


def compute_species_ad(
    reference_table: pd.DataFrame,
    query_table: pd.DataFrame,
    taxonomy_columns: Sequence[str] | None = None,
    *,
    primary_medium_column: str = "primary_medium",
    min_support_score: float = 0.5,
) -> pd.DataFrame:
    """Score species AD from observed taxonomy and primary-medium support."""

    _require_dataframe(reference_table, "reference_table")
    _require_dataframe(query_table, "query_table")
    _validate_fraction(min_support_score, "min_support_score")

    taxonomy = _resolve_taxonomy_columns(reference_table, query_table, taxonomy_columns)
    support_sets = {
        column: _normalized_value_set(reference_table[column])
        for column in taxonomy
        if column in reference_table.columns
    }
    supported_taxonomy_columns = [
        column for column in taxonomy if support_sets.get(column)
    ]
    medium_support = (
        _normalized_value_set(reference_table[primary_medium_column])
        if primary_medium_column in reference_table.columns
        else set()
    )

    records: list[dict[str, object]] = []
    for _idx, row in query_table.iterrows():
        supported = 0
        evaluated = 0
        missing_columns: list[str] = []
        unsupported_fields: list[str] = []

        for column in supported_taxonomy_columns:
            value = _clean_text(row.get(column))
            if value is None:
                missing_columns.append(column)
                continue
            evaluated += 1
            if value in support_sets[column]:
                supported += 1
            else:
                unsupported_fields.append(f"{column}={row.get(column)}")

        taxonomy_reference_count = len(supported_taxonomy_columns)
        taxonomy_supported_fraction = _safe_divide(supported, evaluated)
        taxonomy_score = _safe_divide(supported, taxonomy_reference_count)

        medium_value = _clean_text(row.get(primary_medium_column))
        medium_applicable = bool(medium_support and medium_value is not None)
        medium_supported = bool(medium_applicable and medium_value in medium_support)
        if medium_support and medium_value is None:
            unsupported_fields.append(f"{primary_medium_column}=missing")
        elif medium_applicable and not medium_supported:
            unsupported_fields.append(f"{primary_medium_column}={row.get(primary_medium_column)}")

        component_scores: list[float] = []
        if taxonomy_reference_count > 0:
            component_scores.append(_finite_or_zero(taxonomy_score))
        if medium_support:
            component_scores.append(1.0 if medium_supported else 0.0)
        species_score = float(np.mean(component_scores)) if component_scores else np.nan

        records.append(
            {
                "species_ad_applicable": bool(component_scores),
                "species_ad_in_domain": bool(
                    component_scores and species_score >= min_support_score
                ),
                "species_ad_score": species_score,
                "taxonomy_reference_count": taxonomy_reference_count,
                "taxonomy_evaluated_count": evaluated,
                "taxonomy_supported_count": supported,
                "taxonomy_missing_count": len(missing_columns),
                "taxonomy_supported_fraction": taxonomy_supported_fraction,
                "taxonomy_support_score": taxonomy_score,
                "taxonomy_missing_columns": ";".join(missing_columns),
                "taxonomy_unsupported_fields": ";".join(unsupported_fields),
                "primary_medium_ad_applicable": medium_applicable,
                "primary_medium_supported": medium_supported,
                "primary_medium_reference_count": len(medium_support),
            }
        )

    return pd.DataFrame.from_records(records, index=query_table.index)


def compute_rule_ad(
    rule_table: pd.DataFrame,
    rule_names: Sequence[str] | None = None,
    *,
    min_applicable_fraction: float = 0.5,
    max_missing_fraction: float = 0.5,
) -> pd.DataFrame:
    """Summarize rule-layer missing and applicable coverage per row."""

    _require_dataframe(rule_table, "rule_table")
    _validate_fraction(min_applicable_fraction, "min_applicable_fraction")
    _validate_fraction(max_missing_fraction, "max_missing_fraction")

    names = list(rule_names) if rule_names is not None else _infer_rule_names(rule_table)
    total = len(names)
    records: list[dict[str, object]] = []
    for _idx, row in rule_table.iterrows():
        applicable_names: list[str] = []
        missing_names: list[str] = []
        non_applicable_names: list[str] = []

        for name in names:
            applicable_column = f"rule_{name}_applicable"
            missing_column = f"rule_{name}_missing_inputs"
            applicable = _as_bool(row.get(applicable_column))
            missing_inputs = _clean_text(row.get(missing_column))
            if applicable:
                applicable_names.append(name)
            else:
                non_applicable_names.append(name)
            if missing_inputs:
                missing_names.append(name)

        applicable_fraction = _safe_divide(len(applicable_names), total)
        missing_fraction = _safe_divide(len(missing_names), total)
        if total == 0:
            score = np.nan
        else:
            score = (applicable_fraction + (1.0 - missing_fraction)) / 2.0

        records.append(
            {
                "rule_ad_applicable": total > 0,
                "rule_ad_in_domain": bool(
                    total > 0
                    and applicable_fraction >= min_applicable_fraction
                    and missing_fraction <= max_missing_fraction
                ),
                "rule_ad_score": score,
                "rule_evaluated_count": total,
                "rule_applicable_count": len(applicable_names),
                "rule_missing_input_count": len(missing_names),
                "rule_applicable_fraction": applicable_fraction,
                "rule_missing_fraction": missing_fraction,
                "rule_applicable_names": ";".join(applicable_names),
                "rule_missing_names": ";".join(missing_names),
                "rule_non_applicable_names": ";".join(non_applicable_names),
            }
        )

    return pd.DataFrame.from_records(records, index=rule_table.index)


def summarize_ad(
    ad_table: pd.DataFrame,
    domain_columns: Sequence[str] | None = None,
    score_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return compact counts and score summaries for AD output tables."""

    _require_dataframe(ad_table, "ad_table")
    domains = list(domain_columns) if domain_columns is not None else _infer_domain_columns(ad_table)
    scores = list(score_columns) if score_columns is not None else _infer_score_columns(ad_table)
    n_rows = int(len(ad_table))

    domain_summary: dict[str, dict[str, float | int]] = {}
    for column in domains:
        series = ad_table[column] if column in ad_table.columns else pd.Series(dtype=object)
        true_count = int(series.map(_as_bool).sum()) if len(series) else 0
        missing_count = int(series.isna().sum()) if len(series) else n_rows
        denominator = n_rows - missing_count
        domain_summary[column] = {
            "in_domain_count": true_count,
            "out_of_domain_count": max(denominator - true_count, 0),
            "missing_count": missing_count,
            "in_domain_fraction": _safe_divide(true_count, denominator),
        }

    score_summary: dict[str, dict[str, float | int]] = {}
    for column in scores:
        if column not in ad_table.columns:
            values = pd.Series(dtype=float)
        else:
            values = pd.to_numeric(ad_table[column], errors="coerce")
        finite_values = values[np.isfinite(values)]
        score_summary[column] = {
            "mean": float(finite_values.mean()) if len(finite_values) else np.nan,
            "min": float(finite_values.min()) if len(finite_values) else np.nan,
            "max": float(finite_values.max()) if len(finite_values) else np.nan,
            "missing_count": int(values.isna().sum()) if len(values) else n_rows,
        }

    if domains:
        domain_frame = pd.DataFrame(
            {column: ad_table[column].map(_as_bool) for column in domains if column in ad_table}
        )
        overall = domain_frame.all(axis=1) if not domain_frame.empty else pd.Series(False, index=ad_table.index)
        overall_count = int(overall.sum())
        overall_fraction = _safe_divide(overall_count, n_rows)
    else:
        overall_count = 0
        overall_fraction = np.nan

    return {
        "n_rows": n_rows,
        "domain_columns": domain_summary,
        "score_columns": score_summary,
        "overall_in_domain_count": overall_count,
        "overall_in_domain_fraction": overall_fraction,
    }


chemical_ad = compute_chemical_ad
species_ad = compute_species_ad
rule_ad = compute_rule_ad


def _require_dataframe(value: Any, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")


def _validate_fraction(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _resolve_descriptor_columns(
    reference_table: pd.DataFrame,
    query_table: pd.DataFrame,
    descriptor_columns: Sequence[str] | None,
) -> list[str]:
    if descriptor_columns is not None:
        return list(dict.fromkeys(str(column) for column in descriptor_columns))
    return [
        str(column)
        for column in reference_table.columns
        if column in query_table.columns
        and pd.api.types.is_numeric_dtype(reference_table[column])
        and pd.api.types.is_numeric_dtype(query_table[column])
        and not _looks_like_fingerprint(str(column))
    ]


def _resolve_fingerprint_columns(
    reference_table: pd.DataFrame,
    query_table: pd.DataFrame,
    fingerprint_columns: Sequence[str] | None,
) -> list[str]:
    if fingerprint_columns is not None:
        return list(dict.fromkeys(str(column) for column in fingerprint_columns))
    return [
        str(column)
        for column in reference_table.columns
        if column in query_table.columns and _looks_like_fingerprint(str(column))
    ]


def _resolve_taxonomy_columns(
    reference_table: pd.DataFrame,
    query_table: pd.DataFrame,
    taxonomy_columns: Sequence[str] | None,
) -> list[str]:
    candidates = taxonomy_columns if taxonomy_columns is not None else DEFAULT_TAXONOMY_COLUMNS
    return [
        str(column)
        for column in candidates
        if column in reference_table.columns or column in query_table.columns
    ]


def _looks_like_fingerprint(column: str) -> bool:
    return column.startswith(FINGERPRINT_PREFIXES)


def _descriptor_ranges(
    reference_table: pd.DataFrame,
    columns: Sequence[str],
    tolerance: float,
) -> dict[str, tuple[float, float] | None]:
    ranges: dict[str, tuple[float, float] | None] = {}
    for column in columns:
        if column not in reference_table.columns:
            ranges[column] = None
            continue
        values = pd.to_numeric(reference_table[column], errors="coerce")
        finite_values = values[np.isfinite(values)]
        if finite_values.empty:
            ranges[column] = None
            continue
        lower = float(finite_values.min())
        upper = float(finite_values.max())
        span = upper - lower
        margin = span * tolerance if span > 0 else abs(lower) * tolerance
        ranges[column] = (lower - margin, upper + margin)
    return ranges


def _binary_matrix(table: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    finite_masks = []
    for column in columns:
        if column in table.columns:
            numeric = pd.to_numeric(table[column], errors="coerce")
        else:
            numeric = pd.Series(np.nan, index=table.index)
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        values.append((numeric.fillna(0).to_numpy(dtype=float) > 0).astype(np.int8))
        finite_masks.append(finite)
    return np.column_stack(values), np.column_stack(finite_masks)


def _empty_fingerprint_ad(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fingerprint_ad_applicable": False,
            "fingerprint_ad_in_domain": False,
            "fingerprint_ad_score": np.nan,
            "fingerprint_max_tanimoto": np.nan,
            "fingerprint_nearest_reference_position": None,
            "fingerprint_reference_count": 0,
            "fingerprint_bit_count": 0,
        },
        index=index,
    )


def _tanimoto_against_reference(query_bits: np.ndarray, reference_bits: np.ndarray) -> np.ndarray:
    intersections = np.logical_and(reference_bits, query_bits).sum(axis=1)
    unions = np.logical_or(reference_bits, query_bits).sum(axis=1)
    similarities = np.full(reference_bits.shape[0], np.nan, dtype=float)
    valid = unions > 0
    similarities[valid] = intersections[valid] / unions[valid]
    return similarities


def _infer_rule_names(rule_table: pd.DataFrame) -> list[str]:
    names: list[str] = []
    pattern = re.compile(r"^rule_(.+)_applicable$")
    for column in rule_table.columns:
        match = pattern.match(str(column))
        if not match:
            continue
        name = match.group(1)
        if name != "ad":
            names.append(name)
    return names


def _infer_domain_columns(ad_table: pd.DataFrame) -> list[str]:
    return [str(column) for column in ad_table.columns if str(column).endswith("_in_domain")]


def _infer_score_columns(ad_table: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in ad_table.columns
        if str(column).endswith("_ad_score") or str(column).endswith("_support_score")
    ]


def _normalized_value_set(values: pd.Series) -> set[str]:
    return {cleaned for value in values if (cleaned := _clean_text(value)) is not None}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "missing"}:
        return None
    return text.lower()


def _to_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_or_zero(value: Any) -> float:
    numeric = _to_float(value)
    return float(numeric) if _is_finite(numeric) else 0.0


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
