"""Evaluation split builders."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qsar_dl.evaluation.chemical_categories import assign_chemical_categories


DEFAULT_CONFIG: dict[str, Any] = {
    "split_strategy": "chemical_category_holdout",
    "chemical_id_column": "chemical_id",
    "category_column": "chemical_category",
    "confidence_column": "category_confidence",
    "evidence_column": "category_evidence",
    "split_column": "split",
    "split_strategy_column": "split_strategy",
    "holdout_categories": [],
    "validation_fraction_within_train_categories": 0.15,
    "random_seed": 20260524,
}


def build_category_holdout_splits(
    modeling_table: pd.DataFrame,
    category_table: pd.DataFrame | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Create train/validation/test splits by chemical category.

    All records from configured holdout categories are assigned to ``test``.
    Remaining chemical IDs are split into train/validation groups so duplicate
    observations for the same chemical cannot cross split boundaries.
    """

    if not isinstance(modeling_table, pd.DataFrame):
        raise TypeError("modeling_table must be a pandas DataFrame.")
    if category_table is not None and not isinstance(category_table, pd.DataFrame):
        raise TypeError("category_table must be a pandas DataFrame or None.")

    cfg = _load_config(config)
    chemical_id_column = str(cfg["chemical_id_column"])
    category_column = str(cfg["category_column"])
    split_column = str(cfg["split_column"])
    strategy = str(cfg["split_strategy"])
    validation_fraction = float(cfg["validation_fraction_within_train_categories"])
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction_within_train_categories must be in [0, 1).")

    base = modeling_table.copy()
    if chemical_id_column not in base.columns:
        raise ValueError(f"modeling_table must contain '{chemical_id_column}'.")

    categories = _category_lookup(base, category_table, cfg)
    output = _merge_categories(base, categories, cfg)
    holdout_categories = {str(category) for category in cfg.get("holdout_categories", [])}

    chemical_splits = _assign_chemical_splits(
        output[[chemical_id_column, category_column]].drop_duplicates(),
        chemical_id_column=chemical_id_column,
        category_column=category_column,
        split_column=split_column,
        holdout_categories=holdout_categories,
        validation_fraction=validation_fraction,
        seed=int(cfg["random_seed"]),
    )

    output = output.merge(chemical_splits, on=chemical_id_column, how="left")
    if output[split_column].isna().any():
        missing = sorted(output.loc[output[split_column].isna(), chemical_id_column].unique())
        raise RuntimeError(f"Unable to assign splits for chemical IDs: {missing}")

    output[str(cfg["split_strategy_column"])] = strategy
    output["holdout_category_flag"] = output[category_column].isin(holdout_categories)
    if "modeling_split_group" in output.columns:
        output["modeling_split_group"] = output[category_column]
    return output


def _category_lookup(
    modeling_table: pd.DataFrame,
    category_table: pd.DataFrame | None,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    chemical_id_column = str(cfg["chemical_id_column"])
    category_column = str(cfg["category_column"])
    confidence_column = str(cfg["confidence_column"])
    evidence_column = str(cfg["evidence_column"])

    if category_table is None:
        if category_column in modeling_table.columns:
            source = modeling_table
        else:
            source = assign_chemical_categories(modeling_table, cfg)
    else:
        source = category_table

    if chemical_id_column not in source.columns:
        raise ValueError(f"category_table must contain '{chemical_id_column}'.")
    if category_column not in source.columns:
        source = assign_chemical_categories(source, cfg)

    columns = [chemical_id_column, category_column]
    for optional_column in (confidence_column, evidence_column, "category_source"):
        if optional_column in source.columns:
            columns.append(optional_column)

    lookup = source[columns].drop_duplicates(subset=[chemical_id_column], keep="first")
    return lookup.reset_index(drop=True)


def _merge_categories(
    modeling_table: pd.DataFrame,
    category_lookup: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    chemical_id_column = str(cfg["chemical_id_column"])
    category_columns = [
        str(cfg["category_column"]),
        str(cfg["confidence_column"]),
        str(cfg["evidence_column"]),
        "category_source",
    ]
    output = modeling_table.drop(
        columns=[column for column in category_columns if column in modeling_table.columns]
    )
    output = output.merge(category_lookup, on=chemical_id_column, how="left")
    missing_category = output[str(cfg["category_column"])].isna()
    if missing_category.any():
        output.loc[missing_category, str(cfg["category_column"])] = "other_unknown"
    if str(cfg["confidence_column"]) not in output.columns:
        output[str(cfg["confidence_column"])] = np.nan
    if str(cfg["evidence_column"]) not in output.columns:
        output[str(cfg["evidence_column"])] = ""
    output.loc[missing_category, str(cfg["confidence_column"])] = 0.0
    output.loc[missing_category, str(cfg["evidence_column"])] = "missing category assignment"
    return output


def _assign_chemical_splits(
    chemical_categories: pd.DataFrame,
    *,
    chemical_id_column: str,
    category_column: str,
    split_column: str,
    holdout_categories: set[str],
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assignments: list[dict[str, object]] = []
    seen_ids: set[object] = set()

    for category, group in chemical_categories.groupby(category_column, dropna=False):
        category_key = str(category)
        chemical_ids = sorted(group[chemical_id_column].dropna().unique().tolist(), key=str)
        if category_key in holdout_categories:
            for chemical_id in chemical_ids:
                assignments.append({chemical_id_column: chemical_id, split_column: "test"})
                seen_ids.add(chemical_id)
            continue

        validation_ids: set[object] = set()
        if validation_fraction > 0 and len(chemical_ids) > 1:
            validation_count = int(round(len(chemical_ids) * validation_fraction))
            validation_count = min(max(validation_count, 1), len(chemical_ids) - 1)
            validation_ids = set(
                rng.choice(np.array(chemical_ids), size=validation_count, replace=False).tolist()
            )
        for chemical_id in chemical_ids:
            split = "validation" if chemical_id in validation_ids else "train"
            assignments.append({chemical_id_column: chemical_id, split_column: split})
            seen_ids.add(chemical_id)

    missing_ids = set(chemical_categories[chemical_id_column].dropna()) - seen_ids
    if missing_ids:
        for chemical_id in sorted(missing_ids, key=str):
            assignments.append({chemical_id_column: chemical_id, split_column: "train"})

    return pd.DataFrame.from_records(assignments)


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is None:
        return cfg
    if isinstance(config, (str, Path)):
        loaded = _read_yaml(Path(config))
    elif isinstance(config, Mapping):
        loaded = dict(config)
    else:
        raise TypeError("config must be a mapping, path, string, or None.")

    if isinstance(loaded.get("experiment"), Mapping) and "seed" in loaded["experiment"]:
        cfg["random_seed"] = loaded["experiment"]["seed"]
    if isinstance(loaded.get("evaluation"), Mapping):
        loaded = dict(loaded["evaluation"])
    return _deep_update(cfg, loaded)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on packaging.
        raise RuntimeError("PyYAML is required to load evaluation configs.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Evaluation config must contain a mapping: {path}")
    return dict(data)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
