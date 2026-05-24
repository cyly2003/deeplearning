"""Descriptor prior grouping utilities.

This module is intentionally independent of PyTorch so descriptor dictionary
checks and fixed group-feature export can run in lightweight baseline jobs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on runtime packaging.
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


CHEMICAL_ID_COLUMN = "chemical_id"
_EPSILON = 1.0e-12


def load_descriptor_group_dictionary(path: Path) -> dict[str, Any]:
    """Load and validate descriptor grouping YAML."""

    if yaml is None:
        raise ImportError(
            "PyYAML is required to load descriptor group dictionaries. "
            "Install pyyaml or pass an already-loaded dictionary to downstream APIs."
        ) from _YAML_IMPORT_ERROR

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Descriptor group dictionary not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        group_dict = yaml.safe_load(file)

    _validate_group_dictionary(group_dict)
    return group_dict


def validate_descriptor_coverage(
    descriptor_columns: list[str], group_dict: dict[str, Any]
) -> pd.DataFrame:
    """Return coverage table for grouped, ungrouped, and missing descriptors."""

    _validate_group_dictionary(group_dict)
    observed_descriptors = _deduplicate_descriptor_columns(descriptor_columns)
    observed_set = set(observed_descriptors)
    configured = _descriptor_lookup(group_dict)

    rows: list[dict[str, Any]] = []
    for descriptor in observed_descriptors:
        if descriptor in configured:
            spec = configured[descriptor]
            rows.append(
                {
                    "descriptor": descriptor,
                    "status": "grouped",
                    "group": spec["group"],
                    "role": spec["role"],
                    "initial_weight": spec["initial_weight"],
                    "present": True,
                    "grouped": True,
                }
            )
        else:
            rows.append(
                {
                    "descriptor": descriptor,
                    "status": "ungrouped",
                    "group": None,
                    "role": None,
                    "initial_weight": np.nan,
                    "present": True,
                    "grouped": False,
                }
            )

    for descriptor, spec in configured.items():
        if descriptor not in observed_set:
            rows.append(
                {
                    "descriptor": descriptor,
                    "status": "missing",
                    "group": spec["group"],
                    "role": spec["role"],
                    "initial_weight": spec["initial_weight"],
                    "present": False,
                    "grouped": True,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "descriptor",
            "status",
            "group",
            "role",
            "initial_weight",
            "present",
            "grouped",
        ],
    )


def build_fixed_group_features(
    descriptor_df: pd.DataFrame, group_dict: dict[str, Any]
) -> pd.DataFrame:
    """Export frozen-weight group-level features for traditional ML baselines.

    The returned group value is a row-wise weighted average of valid descriptors
    within a group plus the configured group bias. Missing or non-numeric values
    do not contribute to the numerator or denominator for that row.
    """

    if not isinstance(descriptor_df, pd.DataFrame):
        raise TypeError("descriptor_df must be a pandas DataFrame.")

    _validate_group_dictionary(group_dict)
    if CHEMICAL_ID_COLUMN not in descriptor_df.columns:
        raise ValueError(
            f"descriptor_df must contain a '{CHEMICAL_ID_COLUMN}' column for traceability."
        )
    if descriptor_df.columns.duplicated().any():
        duplicates = sorted(descriptor_df.columns[descriptor_df.columns.duplicated()])
        raise ValueError(f"descriptor_df has duplicate columns: {duplicates}")

    output = pd.DataFrame({CHEMICAL_ID_COLUMN: descriptor_df[CHEMICAL_ID_COLUMN].copy()})
    for group_name, group_config in group_dict["groups"].items():
        descriptor_specs = group_config["descriptors"]
        descriptor_names = list(descriptor_specs)
        present_descriptors = [
            descriptor for descriptor in descriptor_names if descriptor in descriptor_df.columns
        ]
        total_count = len(descriptor_names)

        if present_descriptors:
            values = descriptor_df[present_descriptors].apply(pd.to_numeric, errors="coerce")
            valid_mask = values.notna()
            weights = pd.Series(
                {
                    descriptor: descriptor_specs[descriptor]["initial_weight"]
                    for descriptor in present_descriptors
                },
                dtype="float64",
            )
            weighted_mask = valid_mask.mul(weights, axis=1)
            weight_sums = weighted_mask.sum(axis=1)
            weighted_values = values.fillna(0.0).mul(weights, axis=1).sum(axis=1)
            group_values = weighted_values / weight_sums.where(weight_sums > _EPSILON)
            valid_counts = valid_mask.sum(axis=1).astype("int64")
        else:
            group_values = pd.Series(np.nan, index=descriptor_df.index, dtype="float64")
            valid_counts = pd.Series(0, index=descriptor_df.index, dtype="int64")

        missing_rates = (total_count - valid_counts) / total_count
        output[f"desc_group_{group_name}"] = group_values + group_config["bias_init"]
        output[f"desc_group_{group_name}_missing_rate"] = missing_rates.astype("float64")
        output[f"desc_group_{group_name}_coverage"] = valid_counts

    return output


def _validate_group_dictionary(group_dict: Any) -> None:
    if not isinstance(group_dict, dict):
        raise ValueError("Descriptor group dictionary must be a mapping.")

    descriptor_source = group_dict.get("descriptor_source")
    if not isinstance(descriptor_source, str) or not descriptor_source.strip():
        raise ValueError("'descriptor_source' must be a non-empty string.")

    standardization = group_dict.get("standardization")
    if not isinstance(standardization, dict):
        raise ValueError("'standardization' must be a mapping.")
    for key in ("method", "missing_strategy"):
        value = standardization.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'standardization.{key}' must be a non-empty string.")

    groups = group_dict.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("'groups' must be a non-empty mapping.")

    seen_descriptors: dict[str, str] = {}
    positive_group_weight_count = 0
    for group_name, group_config in groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError("Every group name must be a non-empty string.")
        if not isinstance(group_config, dict):
            raise ValueError(f"Group '{group_name}' must be a mapping.")

        description = group_config.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Group '{group_name}' requires a non-empty description.")

        group_weight = _finite_number(
            group_config.get("initial_group_weight"),
            f"groups.{group_name}.initial_group_weight",
            minimum=0.0,
        )
        if group_weight > 0:
            positive_group_weight_count += 1
        group_config["initial_group_weight"] = group_weight
        group_config["bias_init"] = _finite_number(
            group_config.get("bias_init"),
            f"groups.{group_name}.bias_init",
        )

        descriptors = group_config.get("descriptors")
        if not isinstance(descriptors, dict) or not descriptors:
            raise ValueError(f"Group '{group_name}' must contain descriptors.")

        positive_descriptor_weight_count = 0
        for descriptor_name, descriptor_config in descriptors.items():
            if not isinstance(descriptor_name, str) or not descriptor_name.strip():
                raise ValueError(f"Group '{group_name}' has an empty descriptor name.")
            if descriptor_name in seen_descriptors:
                other_group = seen_descriptors[descriptor_name]
                raise ValueError(
                    f"Descriptor '{descriptor_name}' appears in both "
                    f"'{other_group}' and '{group_name}'."
                )
            seen_descriptors[descriptor_name] = group_name

            if not isinstance(descriptor_config, dict):
                raise ValueError(
                    f"Descriptor '{descriptor_name}' in group '{group_name}' must be a mapping."
                )
            role = descriptor_config.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    f"Descriptor '{descriptor_name}' in group '{group_name}' requires a role."
                )

            descriptor_weight = _finite_number(
                descriptor_config.get("initial_weight"),
                f"groups.{group_name}.descriptors.{descriptor_name}.initial_weight",
                minimum=0.0,
            )
            if descriptor_weight > 0:
                positive_descriptor_weight_count += 1
            descriptor_config["initial_weight"] = descriptor_weight

        if positive_descriptor_weight_count == 0:
            raise ValueError(
                f"Group '{group_name}' must have at least one positive descriptor weight."
            )

    if positive_group_weight_count == 0:
        raise ValueError("At least one group must have a positive initial_group_weight.")


def _descriptor_lookup(group_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for group_name, group_config in group_dict["groups"].items():
        for descriptor, descriptor_config in group_config["descriptors"].items():
            lookup[descriptor] = {
                "group": group_name,
                "role": descriptor_config["role"],
                "initial_weight": descriptor_config["initial_weight"],
            }
    return lookup


def _deduplicate_descriptor_columns(descriptor_columns: list[str]) -> list[str]:
    if not isinstance(descriptor_columns, list):
        raise TypeError("descriptor_columns must be a list of descriptor names.")

    duplicates = sorted(
        {descriptor for descriptor in descriptor_columns if descriptor_columns.count(descriptor) > 1}
    )
    if duplicates:
        raise ValueError(f"descriptor_columns contains duplicates: {duplicates}")
    return list(descriptor_columns)


def _finite_number(value: Any, path: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{path}' must be a finite number.")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"'{path}' must be finite.")
    if minimum is not None and number < minimum:
        raise ValueError(f"'{path}' must be >= {minimum}.")
    return number
