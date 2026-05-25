"""Run full-data hyperparameter tuning for the species-context deep QSAR model."""

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
from run_deep_qsar_experiment import (
    DEFAULT_CHEMICAL_FEATURES,
    DEFAULT_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_MODELING_TABLE,
    export_deep_ablation_figures,
    load_deep_dataset,
    _build_ablation_metrics_table,
    _jsonable,
    _merge_ablation_config,
    _read_yaml,
    _random_seed,
    _value_counts,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "deep_tuning_batch01"


TUNING_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "ctx_baseline_lr1e3_wd1e4_do10",
        "label": "baseline lr=1e-3 wd=1e-4 dropout=0.10",
        "description": "Reference species-context model from the full ablation run.",
        "training": {
            "batch_size": 1024,
            "max_epochs": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 3,
        },
        "model": {
            "chemical_encoder": {"dropout": 0.10},
        },
    },
    {
        "id": "ctx_lr3e4_wd1e3_do20",
        "label": "lr=3e-4 wd=1e-3 dropout=0.20",
        "description": "Lower learning rate with stronger regularization.",
        "training": {
            "batch_size": 1024,
            "max_epochs": 12,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "patience": 4,
        },
        "model": {
            "chemical_encoder": {"dropout": 0.20},
        },
    },
    {
        "id": "ctx_lr3e4_wd1e4_do20",
        "label": "lr=3e-4 wd=1e-4 dropout=0.20",
        "description": "Lower learning rate while keeping weight decay moderate.",
        "training": {
            "batch_size": 1024,
            "max_epochs": 12,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "patience": 4,
        },
        "model": {
            "chemical_encoder": {"dropout": 0.20},
        },
    },
    {
        "id": "ctx_lr1e3_wd1e3_do20",
        "label": "lr=1e-3 wd=1e-3 dropout=0.20",
        "description": "Baseline learning rate with stronger regularization.",
        "training": {
            "batch_size": 1024,
            "max_epochs": 12,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "patience": 4,
        },
        "model": {
            "chemical_encoder": {"dropout": 0.20},
        },
    },
    {
        "id": "ctx_lr1e3_wd1e3_do30_bs512",
        "label": "lr=1e-3 wd=1e-3 dropout=0.30 bs=512",
        "description": "Stronger dropout with smaller batches.",
        "training": {
            "batch_size": 512,
            "max_epochs": 12,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "patience": 4,
        },
        "model": {
            "chemical_encoder": {"dropout": 0.30},
        },
    },
    {
        "id": "ctx_lr2e3_wd1e3_do20",
        "label": "lr=2e-3 wd=1e-3 dropout=0.20",
        "description": "Faster learning with stronger regularization.",
        "training": {
            "batch_size": 1024,
            "max_epochs": 12,
            "learning_rate": 0.002,
            "weight_decay": 0.001,
            "patience": 4,
        },
        "model": {
            "chemical_encoder": {"dropout": 0.20},
        },
    },
]


SPECIES_CONTEXT_ABLATION = {
    "id": "chemical_species_context",
    "label": "Chemical + endpoint + duration + species",
    "model": {
        "context_encoder": {
            "use_endpoint": True,
            "use_duration": True,
        },
    },
    "deep_features": {
        "species_context_columns": [
            "primary_medium",
            "organism_lifestage",
            "taxon_group_l1",
            "taxon_group_l2",
            "taxon_group_l3",
            "is_standard_test_species",
            "is_us_invasive_species",
            "is_us_threatened_endangered",
        ]
    },
}


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)
    evaluation_config = _read_yaml(args.evaluation_config)
    config = dict(config)
    config.setdefault("training", {})
    config["training"] = dict(config["training"])
    config["training"]["max_rows"] = None if args.max_rows is None or args.max_rows < 0 else args.max_rows
    if args.device is not None:
        config["training"]["device"] = args.device

    candidates = _selected_candidates(args.trial)
    data = load_deep_dataset(
        modeling_table=args.modeling_table,
        chemical_features=args.chemical_features,
        evaluation_config=evaluation_config,
        scope=args.scope,
        max_rows=config["training"].get("max_rows"),
        random_seed=_random_seed(config),
    )

    output_dir = args.output_dir
    tables_dir = output_dir / "表格"
    figures_dir = output_dir / "图表"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    for candidate in candidates:
        trial_id = str(candidate["id"])
        ablation_config = _merge_ablation_config(SPECIES_CONTEXT_ABLATION, candidate)
        run_config = _merge_ablation_config(config, ablation_config)
        result = run_real_data_deep_qsar(
            data,
            config=run_config,
            target_column=str(run_config.get("model", {}).get("target_column", "target_ptox")),
        )
        predictions = result.predictions.copy()
        predictions["消融实验"] = trial_id
        predictions["调参实验"] = trial_id
        all_predictions.append(predictions)

        prediction_path = tables_dir / f"{trial_id}_训练与验证集预测结果.parquet"
        validation_prediction_path = tables_dir / f"{trial_id}_验证集预测结果.parquet"
        predictions.to_parquet(prediction_path, index=False)
        predictions.loc[predictions["数据集"] == "验证集"].to_parquet(
            validation_prediction_path,
            index=False,
        )

        report = dict(result.report)
        report["ablation"] = {
            "id": trial_id,
            "label": candidate.get("label", trial_id),
            "description": candidate.get("description"),
        }
        report["tuning"] = {
            "trial_id": trial_id,
            "candidate": candidate,
        }
        report["artifacts"] = {
            "modeling_table": str(args.modeling_table),
            "chemical_features": str(args.chemical_features),
            "output_dir": str(output_dir),
            "prediction_table": str(prediction_path),
            "validation_prediction_table": str(validation_prediction_path),
        }
        trial_report_path = output_dir / f"{trial_id}_deep_metrics.json"
        with trial_report_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(report), handle, ensure_ascii=False, indent=2)
        report["artifacts"]["report_path"] = str(trial_report_path)
        reports.append(report)

    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_prediction_path = tables_dir / "深度调参_全部预测结果.parquet"
    combined_predictions.to_parquet(combined_prediction_path, index=False)
    metrics_table = _build_ablation_metrics_table(reports)
    if "消融实验" in metrics_table.columns and "调参实验" not in metrics_table.columns:
        insert_at = metrics_table.columns.get_loc("消融实验") + 1
        metrics_table.insert(insert_at, "调参实验", metrics_table["消融实验"])
    metrics_path = tables_dir / "深度调参_模型指标汇总.csv"
    metrics_table.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    best_trial = _best_validation_trial(metrics_table)
    figure_paths = (
        export_deep_ablation_figures(
            metrics_table=metrics_table,
            predictions=combined_predictions,
            reports=reports,
            output_dir=figures_dir,
            formats=args.figure_formats,
        )
        if args.export_figures
        else []
    )
    figure_paths = _rename_tuning_figures(figure_paths)
    suite_report = {
        "suite": {
            "scope": args.scope,
            "trial_count": len(reports),
            "trial_ids": [str(candidate["id"]) for candidate in candidates],
            "best_trial_by_validation_R2": best_trial,
        },
        "dataset": {
            "row_count": int(len(data)),
            "chemical_count": int(data["chemical_id"].nunique()) if "chemical_id" in data.columns else None,
            "split_counts": _value_counts(data.get("split")),
        },
        "reports": reports,
        "artifacts": {
            "metrics_table": str(metrics_path),
            "combined_prediction_table": str(combined_prediction_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    report_path = output_dir / "deep_tuning_batch_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(suite_report), handle, ensure_ascii=False, indent=2)
    print(json.dumps(_console_summary(suite_report, report_path), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-data deep QSAR tuning trials.")
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
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only for smoke tests. Default/null runs all rows.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--trial", action="append", default=None)
    parser.add_argument("--figure-formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--no-figures", dest="export_figures", action="store_false")
    parser.set_defaults(export_figures=True)
    return parser.parse_args()


def _selected_candidates(selected: list[str] | None) -> list[dict[str, Any]]:
    if not selected:
        return [dict(candidate) for candidate in TUNING_CANDIDATES]
    wanted = {str(item) for item in selected}
    candidates = [dict(candidate) for candidate in TUNING_CANDIDATES if str(candidate["id"]) in wanted]
    if not candidates:
        raise ValueError(f"No tuning trials selected from ids: {sorted(wanted)}")
    return candidates


def _best_validation_trial(metrics_table: pd.DataFrame) -> dict[str, Any]:
    validation = metrics_table.loc[metrics_table["数据集"] == "validation"].copy()
    if validation.empty:
        return {}
    validation = validation.sort_values(["R2", "RMSE"], ascending=[False, True])
    row = validation.iloc[0].to_dict()
    return {str(key): _json_scalar(value) for key, value in row.items()}


def _json_scalar(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _rename_tuning_figures(paths: list[Path]) -> list[Path]:
    renamed: list[Path] = []
    for path in paths:
        target = path.with_name(path.name.replace("深度消融", "深度调参"))
        if target != path:
            path.replace(target)
            renamed.append(target)
        else:
            renamed.append(path)
    return renamed


def _console_summary(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for item in report.get("reports", []):
        trial_id = str(item.get("tuning", {}).get("trial_id", "unknown"))
        metrics[trial_id] = item.get("metrics", {})
    return {
        "report_path": str(report_path),
        "suite": report.get("suite", {}),
        "dataset": report.get("dataset", {}),
        "metrics": metrics,
        "artifacts": report.get("artifacts", {}),
    }


if __name__ == "__main__":
    main()
