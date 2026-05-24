"""Shared types and helpers for explicit toxicology rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence


@dataclass
class RuleOutput:
    """Output from one mechanistic rule.

    Features are model-ready intermediate values. Corrections are optional
    pTox adjustment candidates and are not applied to labels by this layer.
    Flags carry missing/applicability/quality masks for downstream auditing.
    """

    features: dict[str, float | int | None]
    corrections: dict[str, float | None]
    flags: dict[str, bool | str | None]
    explanation: str


class MechanisticRule(Protocol):
    """Protocol implemented by all explicit toxicology rules."""

    name: str
    required_inputs: list[str]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        ...


def is_missing(value: Any) -> bool:
    """Return True for common scalar missing-value markers."""

    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return text == "" or text.lower() in {"na", "nan", "none", "null", "missing"}
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any) -> float | None:
    """Convert a scalar value to float, preserving missing values as None."""

    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_str(value: Any) -> str | None:
    """Convert a scalar value to a stripped string, preserving missing values."""

    if is_missing(value):
        return None
    return str(value).strip()


def get_first(row: Mapping[str, Any], aliases: Sequence[str]) -> tuple[str | None, Any]:
    """Return the first non-missing value found for a set of column aliases."""

    for key in aliases:
        if key in row and not is_missing(row[key]):
            return key, row[key]
    return None, None


def get_float(row: Mapping[str, Any], aliases: Sequence[str]) -> tuple[str | None, float | None]:
    """Return the first non-missing value converted to float."""

    key, value = get_first(row, aliases)
    return key, as_float(value)


def get_text(row: Mapping[str, Any], aliases: Sequence[str]) -> tuple[str | None, str | None]:
    """Return the first non-missing value converted to text."""

    key, value = get_first(row, aliases)
    return key, as_str(value)


def join_missing(names: Sequence[str]) -> str:
    """Represent missing inputs as a stable comma-separated string."""

    return ",".join(dict.fromkeys(names))


def clip(value: float, lower: float, upper: float) -> float:
    """Clip a float to a closed interval."""

    return max(lower, min(upper, value))


def safe_log10(value: float | None) -> float | None:
    """Return log10 for positive values, otherwise None."""

    if value is None or value <= 0:
        return None
    return math.log10(value)


def softplus(value: float) -> float:
    """Numerically stable softplus."""

    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))
