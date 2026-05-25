"""Run endpoint-semantics and fingerprint ablations for ECx, LOEC, and NOEC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qsar_dl.training.baseline_ml import evaluate_regression
from run_baseline_ml_experiment import (
    DEFAULT_CHEMICAL_FEATURES,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_MODELING_TABLE,
    add_category_splits,
    build_metrics_table,
    export_baseline_figures,
    run_baseline_experiment_with_artifacts,
    _jsonable,
    _read_yaml,
    _value_counts,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "endpoint_fingerprint_ablation_v001"
RANDOM_SEED = 20260524
DESCRIPTOR_COLUMNS = [
    "rdkit_descriptor_mol_wt",
    "rdkit_descriptor_exact_mol_wt",
    "rdkit_descriptor_mol_logp",
    "rdkit_descriptor_tpsa",
    "rdkit_descriptor_h_bond_donors",
    "rdkit_descriptor_h_bond_acceptors",
    "rdkit_descriptor_rotatable_bonds",
    "rdkit_descriptor_ring_count",
    "rdkit_descriptor_heavy_atom_count",
    "rdkit_descriptor_fraction_csp3",
    "rdkit_descriptor_formal_charge",
    "molecular_weight_g_mol",
    "logkow",
    "molecular_weight_g_mol_missing_flag",
    "logkow_missing_flag",
]
SEMANTIC_CONTEXT_COLUMNS = [
    "primary_medium",
    "organism_lifestage",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "is_standard_test_species",
    "is_us_invasive_species",
    "is_us_threatened_endangered",
    "chemical_class_l1",
    "chemical_class_l2",
    "chemical_class_l3",
    "use_source_class",
    "endpoint_stat_type",
    "response_domain",
    "effect_code_clean",
    "measurement_code_clean",
    "trend_clean",
    "duration_h",
    "log_duration_h",
    "effect_percent",
    "effect_fraction",
    "effect_level_logit",
    "is_lethal_response",
    "is_chronic_threshold",
]
TASKS = ("ecx", "loec", "noec")
FEATURE_SET_LABELS = {
    "A_descriptors_no_fp": "A descriptors only",
    "B_descriptors_fp512": "B descriptors + first 512 FP bits",
    "C_descriptors_fp2048": "C descriptors + 2048 FP bits",
    "D_descriptors_filtered_fp": "D descriptors + filtered FP bits",
}


def main() -> None:
    args = parse_args()
    evaluation_config = _read_yaml(args.evaluation_config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_endpoint_semantic_dataset(
        modeling_table=args.modeling_table,
        chemical_features=args.chemical_features,
    )
    data = data.loc[data["endpoint_task"].isin(args.task)].copy()
    if args.max_rows is not None:
        data = pd.concat(
            [
                _sample_group(group, args.max_rows, RANDOM_SEED)
                for _task, group in data.groupby("endpoint_task", sort=False)
            ],
            ignore_index=True,
        )

    reports: list[dict[str, Any]] = []
    metrics_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    for task in args.task:
        task_data = data.loc[data["endpoint_task"] == task].copy()
        if task_data.empty:
            continue
        task_data = add_category_splits(task_data, evaluation_config)
        feature_sets = build_ablation_feature_sets(task_data, max_filtered_bits=args.max_filtered_bits)
        config = build_experiment_config(feature_sets, models=args.models, task=task)
        result, predictions = run_baseline_experiment_with_artifacts(task_data, config)
        result["task"] = task
        result["endpoint_semantics"] = task_dataset_summary(task_data)

        task_dir = output_dir / task
        tables_dir = task_dir / "表格"
        figures_dir = task_dir / "图表"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        metrics = build_metrics_table(result)
        metrics.insert(0, "endpoint任务", task)
        metrics.insert(1, "endpoint任务标签", _task_label(task))
        metrics["特征集说明"] = metrics["特征集"].map(FEATURE_SET_LABELS).fillna(metrics["特征集"])
        metrics_path = tables_dir / f"{task}_ABCD_模型指标汇总.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        predictions = predictions.copy()
        predictions.insert(0, "endpoint任务", task)
        prediction_path = tables_dir / f"{task}_ABCD_训练与验证集预测结果.parquet"
        predictions.to_parquet(prediction_path, index=False)

        figure_paths = []
        if args.export_figures:
            figure_paths.extend(
                export_baseline_figures(
                    data=task_data,
                    metrics_table=metrics,
                    predictions=predictions,
                    output_dir=figures_dir,
                    formats=args.figure_formats,
                )
            )
            figure_paths.extend(
                export_endpoint_ablation_figures(
                    metrics,
                    output_dir=figures_dir,
                    formats=args.figure_formats,
                    task=task,
                )
            )

        report = {
            "task": task,
            "dataset": task_dataset_summary(task_data),
            "feature_sets": feature_set_summary(result),
            "metrics_table": str(metrics_path),
            "prediction_table": str(prediction_path),
            "figures": [str(path) for path in figure_paths],
            "result": result,
        }
        report_path = task_dir / f"{task}_endpoint_fingerprint_ablation_metrics.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(report), handle, ensure_ascii=False, indent=2)
        report["report_path"] = str(report_path)
        reports.append(report)
        metrics_tables.append(metrics)
        prediction_tables.append(predictions)

    combined_metrics = pd.concat(metrics_tables, ignore_index=True) if metrics_tables else pd.DataFrame()
    combined_predictions = (
        pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
    )
    tables_dir = output_dir / "表格"
    figures_dir = output_dir / "图表"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    combined_metrics_path = tables_dir / "endpoint_ABCD_全部任务模型指标汇总.csv"
    combined_predictions_path = tables_dir / "endpoint_ABCD_全部任务预测结果.parquet"
    combined_metrics.to_csv(combined_metrics_path, index=False, encoding="utf-8-sig")
    combined_predictions.to_parquet(combined_predictions_path, index=False)
    combined_figure_paths = (
        export_endpoint_ablation_figures(
            combined_metrics,
            output_dir=figures_dir,
            formats=args.figure_formats,
            task="all_tasks",
        )
        if args.export_figures and not combined_metrics.empty
        else []
    )
    suite_report = {
        "suite": {
            "tasks": list(args.task),
            "models": args.models,
            "feature_sets": FEATURE_SET_LABELS,
        },
        "dataset": {
            "row_count": int(len(data)),
            "task_counts": _value_counts(data.get("endpoint_task")),
            "chemical_count": int(data["chemical_id"].nunique()) if not data.empty else 0,
        },
        "reports": reports,
        "artifacts": {
            "combined_metrics_table": str(combined_metrics_path),
            "combined_prediction_table": str(combined_predictions_path),
            "combined_figures": [str(path) for path in combined_figure_paths],
        },
    }
    suite_report_path = output_dir / "endpoint_fingerprint_ablation_suite_metrics.json"
    with suite_report_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(suite_report), handle, ensure_ascii=False, indent=2)
    print(json.dumps(compact_summary(suite_report, suite_report_path), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run endpoint semantic ABCD fingerprint ablations.")
    parser.add_argument("--modeling-table", type=Path, default=DEFAULT_MODELING_TABLE)
    parser.add_argument("--chemical-features", type=Path, default=DEFAULT_CHEMICAL_FEATURES)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", choices=TASKS, action="append", default=None)
    parser.add_argument("--models", nargs="+", default=["lightgbm"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-filtered-bits", type=int, default=512)
    parser.add_argument("--figure-formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--no-figures", dest="export_figures", action="store_false")
    parser.set_defaults(export_figures=True)
    args = parser.parse_args()
    if args.task is None:
        args.task = list(TASKS)
    return args


def load_endpoint_semantic_dataset(*, modeling_table: Path, chemical_features: Path) -> pd.DataFrame:
    modeling = pd.read_parquet(modeling_table)
    chemicals = pd.read_parquet(chemical_features)
    modeling["chemical_id"] = modeling["chemical_id"].astype("string").str.strip()
    chemicals["chemical_id"] = chemicals["chemical_id"].astype("string").str.strip()
    base = modeling.loc[
        (modeling["target_unit_family"] == "water_mg_l")
        & modeling["target_ptox"].notna()
        & modeling["chemical_id"].notna()
        & modeling["smiles"].notna()
    ].copy()
    base = add_endpoint_semantics(base)
    base = base.loc[base["endpoint_task"].notna()].copy()
    feature_columns = [
        column for column in chemicals.columns if column not in base.columns or column == "chemical_id"
    ]
    merged = base.merge(chemicals[feature_columns], on="chemical_id", how="left", validate="many_to_one")
    return merged.reset_index(drop=True)


def add_endpoint_semantics(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    parsed = output.apply(parse_endpoint_semantics, axis=1, result_type="expand")
    for column in parsed.columns:
        output[column] = parsed[column]
    output["effect_code_clean"] = output["effect"].map(clean_code)
    output["measurement_code_clean"] = output["measurement"].map(clean_code)
    output["trend_clean"] = output["trend"].map(clean_code)
    output["log_duration_h"] = np.log1p(pd.to_numeric(output["duration_h"], errors="coerce").clip(lower=0.0))
    return output


def parse_endpoint_semantics(row: pd.Series) -> dict[str, Any]:
    raw = normalize_endpoint(row.get("endpoint_raw"))
    ecx_match = re.match(r"^(EC|LC)(\d+(?:\.\d+)?)$", raw)
    if ecx_match:
        family = ecx_match.group(1)
        percent = float(ecx_match.group(2))
        response = "mortality" if family == "LC" else infer_response_domain(row)
        fraction = percent / 100.0
        logit = math.log(fraction / (1.0 - fraction)) if 0.0 < fraction < 1.0 else np.nan
        return {
            "endpoint_task": "ecx",
            "endpoint_stat_type": "point_estimate",
            "endpoint_semantic_family": "ECx",
            "endpoint_source_family": family,
            "effect_percent": percent,
            "effect_fraction": fraction,
            "effect_level_logit": logit,
            "response_domain": response,
            "is_lethal_response": response == "mortality",
            "is_chronic_threshold": False,
        }
    if raw.startswith("LOEC"):
        response = infer_response_domain(row)
        return {
            "endpoint_task": "loec",
            "endpoint_stat_type": "threshold_observed_effect",
            "endpoint_semantic_family": "LOEC",
            "endpoint_source_family": "LOEC",
            "effect_percent": np.nan,
            "effect_fraction": np.nan,
            "effect_level_logit": np.nan,
            "response_domain": response,
            "is_lethal_response": response == "mortality",
            "is_chronic_threshold": True,
        }
    if raw.startswith("NOEC"):
        response = infer_response_domain(row)
        return {
            "endpoint_task": "noec",
            "endpoint_stat_type": "threshold_no_observed_effect",
            "endpoint_semantic_family": "NOEC",
            "endpoint_source_family": "NOEC",
            "effect_percent": np.nan,
            "effect_fraction": np.nan,
            "effect_level_logit": np.nan,
            "response_domain": response,
            "is_lethal_response": response == "mortality",
            "is_chronic_threshold": True,
        }
    return {
        "endpoint_task": None,
        "endpoint_stat_type": None,
        "endpoint_semantic_family": None,
        "endpoint_source_family": None,
        "effect_percent": np.nan,
        "effect_fraction": np.nan,
        "effect_level_logit": np.nan,
        "response_domain": None,
        "is_lethal_response": False,
        "is_chronic_threshold": False,
    }


def normalize_endpoint(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9.]", "", str(value).strip().upper())


def clean_code(value: object) -> str:
    if value is None or pd.isna(value):
        return "missing"
    text = re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())
    return text or "missing"


def infer_response_domain(row: pd.Series) -> str:
    effect = clean_code(row.get("effect")).lstrip("~")
    measurement = clean_code(row.get("measurement")).lstrip("~")
    for code in (effect, measurement):
        if code in {"MOR", "MORT", "SURV"}:
            return "mortality"
        if code in {"ITX", "IMBL"}:
            return "immobilization"
        if code in {"GRO", "PGRT", "GGRO", "LGTH", "WGHT", "BMAS"}:
            return "growth"
        if code in {"REP", "FERZ", "PROG", "FCND"}:
            return "reproduction"
        if code in {"DVP", "HTCH", "DFRM", "NORM", "ABNM"}:
            return "development"
        if code in {"BEH", "LOCO", "MOTL", "SWIM", "EQUL", "GBHV"}:
            return "behavior"
        if code in {"PHY", "PSYN", "CHLA", "CHLO", "RESP"}:
            return "physiology"
        if code in {"BCM", "ENZ", "ACHE", "CTLS", "SODA", "GSTR", "GLPX"}:
            return "biochemical"
        if code in {"POP", "ABND", "GPOP"}:
            return "population"
        if code in {"CEL"}:
            return "cellular"
        if code in {"GEN"}:
            return "genetic"
        if code in {"HRM"}:
            return "endocrine"
        if code in {"HIS", "MPH"}:
            return "morphology_histology"
        if code in {"FDB"}:
            return "feeding"
    return "other"


def build_ablation_feature_sets(data: pd.DataFrame, *, max_filtered_bits: int) -> list[dict[str, Any]]:
    fingerprint_columns = sorted(column for column in data.columns if column.startswith("morgan_fp_"))
    first_512 = fingerprint_columns[:512]
    filtered = select_frequency_filtered_fingerprints(
        data,
        fingerprint_columns,
        max_bits=max_filtered_bits,
    )
    return [
        {
            "name": "A_descriptors_no_fp",
            "type": "standard",
            "feature_columns": present_columns(data, DESCRIPTOR_COLUMNS),
        },
        {
            "name": "B_descriptors_fp512",
            "type": "standard",
            "feature_columns": present_columns(data, DESCRIPTOR_COLUMNS + first_512),
        },
        {
            "name": "C_descriptors_fp2048",
            "type": "standard",
            "feature_columns": present_columns(data, DESCRIPTOR_COLUMNS + fingerprint_columns),
        },
        {
            "name": "D_descriptors_filtered_fp",
            "type": "standard",
            "feature_columns": present_columns(data, DESCRIPTOR_COLUMNS + filtered),
        },
    ]


def select_frequency_filtered_fingerprints(
    data: pd.DataFrame,
    fingerprint_columns: list[str],
    *,
    max_bits: int,
    min_frequency: float = 0.01,
    max_frequency: float = 0.99,
) -> list[str]:
    if not fingerprint_columns:
        return []
    if "split" in data.columns:
        train = data.loc[data["split"].astype("string") == "train", fingerprint_columns]
        if train.empty:
            train = data[fingerprint_columns]
    else:
        train = data[fingerprint_columns]
    frequency = train.apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=0)
    keep = frequency[(frequency >= min_frequency) & (frequency <= max_frequency)]
    variance = (keep * (1.0 - keep)).sort_values(ascending=False)
    return variance.head(int(max_bits)).index.astype(str).tolist()


def present_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in data.columns]


def build_experiment_config(
    feature_sets: list[dict[str, Any]],
    *,
    models: list[str],
    task: str,
) -> dict[str, Any]:
    context_columns = list(SEMANTIC_CONTEXT_COLUMNS)
    if task != "ecx":
        context_columns = [
            column
            for column in context_columns
            if column not in {"effect_percent", "effect_fraction", "effect_level_logit"}
        ]
    return {
        "experiment": {"seed": RANDOM_SEED},
        "target": {"column": "target_ptox"},
        "baseline_ml": {
            "models": models,
            "split_column": "split",
            "train_values": ["train"],
            "test_values": ["test", "validation", "val"],
            "test_size": 0.25,
            "context_columns": context_columns,
            "feature_sets": feature_sets,
            "model_params": {
                "lightgbm": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "n_jobs": -1,
                },
                "xgboost": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "tree_method": "hist",
                    "n_jobs": -1,
                },
                "random_forest": {
                    "n_estimators": 80,
                    "min_samples_leaf": 5,
                    "n_jobs": -1,
                },
                "elasticnet": {
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                },
            },
        },
    }


def task_dataset_summary(data: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(data)),
        "chemical_count": int(data["chemical_id"].nunique()),
        "species_count": int(data["species_id"].nunique()) if "species_id" in data.columns else None,
        "split_counts": _value_counts(data.get("split")),
        "response_domain_counts": _value_counts(data.get("response_domain")),
        "endpoint_source_family_counts": _value_counts(data.get("endpoint_source_family")),
        "effect_percent_counts": _value_counts(data.get("effect_percent")),
    }


def feature_set_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature_name, feature_result in dict(result.get("feature_sets", {})).items():
        output[str(feature_name)] = {
            "n_features": feature_result.get("n_features"),
            "train_rows": feature_result.get("train_rows"),
            "test_rows": feature_result.get("test_rows"),
        }
    return output


def export_endpoint_ablation_figures(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    formats: list[str],
    task: str,
) -> list[Path]:
    if metrics.empty:
        return []
    import matplotlib.pyplot as plt

    from qsar_dl.visualization import save_figure, set_publication_style

    set_publication_style(language="zh", palette="journal")
    validation = metrics.loc[metrics["状态"] == "trained"].copy()
    if validation.empty:
        return []
    saved: list[Path] = []
    for model_name, group in validation.groupby("模型", sort=True):
        ordered = group.sort_values("特征集")
        labels = ordered["特征集"].astype(str).tolist()
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
        axes[0].bar(labels, ordered["R2"], color="#3B7EA1")
        axes[0].set_ylabel(r"$\mathrm{R^2}$")
        axes[0].set_title("验证集 R2")
        axes[1].bar(labels, ordered["RMSE"], color="#D95F02")
        axes[1].set_ylabel("RMSE (pTox)")
        axes[1].set_title("验证集 RMSE")
        axes[2].plot(ordered["特征数"], ordered["R2"], marker="o", color="#1B9E77")
        axes[2].set_xlabel("特征数")
        axes[2].set_ylabel(r"$\mathrm{R^2}$")
        axes[2].set_title("维度-性能关系")
        for ax in axes[:2]:
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", linestyle="--", alpha=0.28)
        axes[2].grid(True, linestyle="--", alpha=0.28)
        fig.tight_layout()
        saved.extend(
            save_figure(
                fig,
                output_dir / f"{task}_ABCD_指纹消融_{model_name}",
                formats=formats,
                close=True,
            )
        )
    return saved


def _task_label(task: str) -> str:
    labels = {
        "ecx": "ECx with LCx mortality merged",
        "loec": "LOEC threshold",
        "noec": "NOEC threshold",
    }
    return labels.get(task, task)


def _sample_group(group: pd.DataFrame, max_rows: int, random_seed: int) -> pd.DataFrame:
    if len(group) <= int(max_rows):
        return group
    return group.sample(n=int(max_rows), random_state=random_seed).sort_index()


def compact_summary(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for item in report.get("reports", []):
        task = str(item.get("task"))
        task_metrics = pd.read_csv(item["metrics_table"])
        metrics[task] = task_metrics[
            ["特征集", "模型", "R2", "RMSE", "MAE", "MAPE", "训练样本数", "验证样本数", "特征数"]
        ].to_dict(orient="records")
    return {
        "report_path": str(report_path),
        "dataset": report.get("dataset", {}),
        "metrics": metrics,
        "artifacts": report.get("artifacts", {}),
    }


if __name__ == "__main__":
    main()
