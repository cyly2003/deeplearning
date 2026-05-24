"""Registry and batch computation for explicit toxicology rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .aquatic_hydrophobicity import AquaticHydrophobicityRule
from .base import MechanisticRule, RuleOutput
from .duration import DurationRule
from .ionization import IonizationRule
from .molecular_weight import MolecularWeightRule
from .route_access import RouteAccessRule
from .solubility import SolubilityRule
from .stubs import (
    BioavailabilityRule,
    ChemicalActivityRule,
    MoaExcessToxicityRule,
    TktdRule,
    VolatilityRule,
)


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return dict(config)


def _rule_config(config: Mapping[str, Any], rule_name: str) -> dict[str, Any]:
    rules_config = config.get("rules", config)
    section = rules_config.get(rule_name, {}) if isinstance(rules_config, Mapping) else {}
    return dict(section or {})


def get_rule_registry(config: Mapping[str, Any] | str | Path | None = None) -> list[MechanisticRule]:
    """Return registered rules in deterministic order."""

    _ = _load_config(config)
    return [
        AquaticHydrophobicityRule(),
        SolubilityRule(),
        ChemicalActivityRule(),
        MoaExcessToxicityRule(),
        DurationRule(),
        TktdRule(),
        VolatilityRule(),
        BioavailabilityRule(),
        IonizationRule(),
        MolecularWeightRule(),
        RouteAccessRule(),
    ]


def _flatten_output(rule: MechanisticRule, output: RuleOutput) -> dict[str, float | int | bool | str | None]:
    flat: dict[str, float | int | bool | str | None] = {}
    flat.update(output.features)
    flat.update(output.corrections)
    flat.update(output.flags)
    flat[f"rule_{rule.name}_explanation"] = output.explanation
    return flat


def compute_rule_layer(
    batch_df: pd.DataFrame,
    config: Mapping[str, Any] | str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return rule feature/flag table and coverage report.

    The function only computes in-memory artifacts. Writing
    outputs/features/rule_features.parquet and
    outputs/reports/rule_coverage_report.json should be handled by the caller.
    """

    loaded_config = _load_config(config)
    registry = get_rule_registry(loaded_config)
    rows: list[dict[str, float | int | bool | str | None]] = []
    stats = {
        rule.name: {
            "n_rows": int(len(batch_df)),
            "applicable_count": 0,
            "missing_count": 0,
            "disabled_count": 0,
        }
        for rule in registry
    }

    for _, row in batch_df.iterrows():
        row_mapping = row.to_dict()
        flat_row: dict[str, float | int | bool | str | None] = {}
        for rule in registry:
            output = rule.compute(row_mapping, _rule_config(loaded_config, rule.name))
            flat = _flatten_output(rule, output)
            flat_row.update(flat)

            applicable = flat.get(f"rule_{rule.name}_applicable")
            missing = flat.get(f"rule_{rule.name}_missing_inputs")
            disabled = flat.get(f"rule_{rule.name}_disabled")
            if applicable is True:
                stats[rule.name]["applicable_count"] += 1
            if isinstance(missing, str) and missing:
                stats[rule.name]["missing_count"] += 1
            if disabled is True:
                stats[rule.name]["disabled_count"] += 1
        rows.append(flat_row)

    feature_table = pd.DataFrame(rows, index=batch_df.index)
    for rule_name, rule_stats in stats.items():
        n_rows = rule_stats["n_rows"]
        rule_stats["applicable_fraction"] = rule_stats["applicable_count"] / n_rows if n_rows else 0.0
        rule_stats["missing_fraction"] = rule_stats["missing_count"] / n_rows if n_rows else 0.0

    report = {
        "n_rows": int(len(batch_df)),
        "n_rules": len(registry),
        "rules": stats,
        "columns": list(feature_table.columns),
        "note": "Rule corrections are candidates only; this layer does not rewrite labels.",
    }
    return feature_table, report
