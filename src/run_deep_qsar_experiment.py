"""Run real-data residual QSAR deep baselines on standardized ECOTOX outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qsar_dl.training.train_deep import run_real_data_deep_qsar
from run_baseline_ml_experiment import add_category_splits


DEFAULT_MODELING_TABLE = PROJECT_ROOT / "outputs" / "tables" / "modeling_toxicity_long.parquet"
DEFAULT_CHEMICAL_FEATURES = PROJECT_ROOT / "outputs" / "features" / "chemical_features.parquet"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "baseline_deep.yaml"
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation" / "chemical_category_holdout.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "baseline_deep_v001"


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)
    evaluation_config = _read_yaml(args.evaluation_config)
    training_config = dict(config.get("training", {}))
    if args.max_rows is not None:
        training_config["max_rows"] = args.max_rows
    if args.max_epochs is not None:
        training_config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.device is not None:
        training_config["device"] = args.device
    config = dict(config)
    config["training"] = training_config

    data = load_deep_dataset(
        modeling_table=args.modeling_table,
        chemical_features=args.chemical_features,
        evaluation_config=evaluation_config,
        scope=args.scope,
        max_rows=training_config.get("max_rows"),
        random_seed=_random_seed(config),
    )
    result = run_real_data_deep_qsar(
        data,
        config=config,
        target_column=str(config.get("model", {}).get("target_column", "target_ptox")),
    )

    output_dir = args.output_dir
    tables_dir = output_dir / "表格"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = tables_dir / "深度基线_训练与验证集预测结果.parquet"
    validation_prediction_path = tables_dir / "深度基线_验证集预测结果.parquet"
    result.predictions.to_parquet(prediction_path, index=False)
    result.predictions.loc[result.predictions["数据集"] == "验证集"].to_parquet(
        validation_prediction_path,
        index=False,
    )
    report = dict(result.report)
    report["artifacts"] = {
        "modeling_table": str(args.modeling_table),
        "chemical_features": str(args.chemical_features),
        "output_dir": str(output_dir),
        "prediction_table": str(prediction_path),
        "validation_prediction_table": str(validation_prediction_path),
    }
    report_path = output_dir / "deep_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, ensure_ascii=False, indent=2)

    print(json.dumps(_console_summary(report, report_path), ensure_ascii=False, indent=2))


def load_deep_dataset(
    *,
    modeling_table: Path,
    chemical_features: Path,
    evaluation_config: Mapping[str, Any],
    scope: str,
    max_rows: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Load modeling rows, assign chemical-category splits and merge features."""

    modeling = pd.read_parquet(modeling_table)
    filtered = _filter_scope(modeling, scope)
    split_data = add_category_splits(filtered, evaluation_config)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows < 2:
            raise ValueError("max_rows must be >= 2.")
        if len(split_data) > max_rows:
            split_data = (
                split_data.sample(n=max_rows, random_state=random_seed)
                .sort_index()
                .reset_index(drop=True)
            )

    chemicals = pd.read_parquet(chemical_features)
    split_data = _normalize_join_key(split_data, "chemical_id")
    chemicals = _normalize_join_key(chemicals, "chemical_id")
    feature_columns = [
        column for column in chemicals.columns if column not in split_data.columns or column == "chemical_id"
    ]
    merged = split_data.merge(
        chemicals[feature_columns],
        on="chemical_id",
        how="left",
        validate="many_to_one",
    )
    return merged.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a residual QSAR deep baseline.")
    parser.add_argument("--modeling-table", type=Path, default=DEFAULT_MODELING_TABLE)
    parser.add_argument("--chemical-features", type=Path, default=DEFAULT_CHEMICAL_FEATURES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scope",
        choices=["main_water_task", "transfer_model_ready"],
        default="main_water_task",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _filter_scope(modeling: pd.DataFrame, scope: str) -> pd.DataFrame:
    normalized = scope.strip().lower()
    if normalized == "main_water_task":
        mask = modeling["is_main_water_task"].fillna(False)
    elif normalized == "transfer_model_ready":
        mask = modeling["is_transfer_model_ready"].fillna(False)
    else:
        raise ValueError(f"Unsupported scope: {scope}")
    mask = mask & modeling["target_ptox"].notna()
    filtered = modeling.loc[mask].copy()
    if filtered.empty:
        raise ValueError(f"No rows available for scope={scope!r}.")
    return filtered.reset_index(drop=True)


def _normalize_join_key(data: pd.DataFrame, column: str) -> pd.DataFrame:
    output = data.copy()
    output[column] = output[column].astype("string").str.strip()
    return output


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return dict(data)


def _random_seed(config: Mapping[str, Any]) -> int:
    experiment = config.get("experiment")
    if isinstance(experiment, Mapping) and "seed" in experiment:
        return int(experiment["seed"])
    return 20260524


def _console_summary(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "report_path": str(report_path),
        "dataset": report.get("dataset", {}),
        "model": report.get("model", {}),
        "metrics": report.get("metrics", {}),
    }


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
