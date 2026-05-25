"""Run real-data residual QSAR deep baselines on standardized ECOTOX outputs."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from qsar_dl.visualization import save_figure, set_publication_style
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
        training_config["max_rows"] = None if args.max_rows < 0 else args.max_rows
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
    suite = _resolve_ablation_suite(config, args.ablation)
    output_dir = args.output_dir
    tables_dir = output_dir / "表格"
    figures_dir = output_dir / "图表"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    for ablation in suite:
        ablation_id = str(ablation["id"])
        run_config = _merge_ablation_config(config, ablation)
        result = run_real_data_deep_qsar(
            data,
            config=run_config,
            target_column=str(run_config.get("model", {}).get("target_column", "target_ptox")),
        )
        predictions = result.predictions.copy()
        predictions["消融实验"] = ablation_id
        all_predictions.append(predictions)

        prediction_path = tables_dir / f"{ablation_id}_训练与验证集预测结果.parquet"
        validation_prediction_path = tables_dir / f"{ablation_id}_验证集预测结果.parquet"
        predictions.to_parquet(prediction_path, index=False)
        predictions.loc[predictions["数据集"] == "验证集"].to_parquet(
            validation_prediction_path,
            index=False,
        )
        report = dict(result.report)
        report["ablation"] = {
            "id": ablation_id,
            "label": ablation.get("label", ablation_id),
            "description": ablation.get("description"),
        }
        report["artifacts"] = {
            "modeling_table": str(args.modeling_table),
            "chemical_features": str(args.chemical_features),
            "output_dir": str(output_dir),
            "prediction_table": str(prediction_path),
            "validation_prediction_table": str(validation_prediction_path),
        }
        ablation_report_path = output_dir / f"{ablation_id}_deep_metrics.json"
        with ablation_report_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(report), handle, ensure_ascii=False, indent=2)
        report["artifacts"]["report_path"] = str(ablation_report_path)
        reports.append(report)

    combined_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    combined_prediction_path = tables_dir / "深度消融_全部预测结果.parquet"
    combined_predictions.to_parquet(combined_prediction_path, index=False)
    metrics_table = _build_ablation_metrics_table(reports)
    metrics_path = tables_dir / "深度消融_模型指标汇总.csv"
    metrics_table.to_csv(metrics_path, index=False, encoding="utf-8-sig")
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
    report = {
        "suite": {
            "scope": args.scope,
            "ablation_count": len(reports),
            "ablation_ids": [report["ablation"]["id"] for report in reports],
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
    report_path = output_dir / "deep_ablation_suite_metrics.json"
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
    parser.add_argument(
        "--ablation",
        action="append",
        default=None,
        help="Ablation id to run. Can be repeated. Defaults to all configured ablations.",
    )
    parser.add_argument("--figure-formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--no-figures", dest="export_figures", action="store_false")
    parser.set_defaults(export_figures=True)
    return parser.parse_args()


def export_deep_ablation_figures(
    *,
    metrics_table: pd.DataFrame,
    predictions: pd.DataFrame,
    reports: list[Mapping[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    """Export standard visual diagnostics for the deep ablation suite."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visualization dependency.
        raise RuntimeError("matplotlib is required to export deep ablation figures.") from exc

    set_publication_style(language="zh", palette="journal")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    saved.extend(_plot_ablation_metrics(metrics_table, output_dir, formats, plt))
    saved.extend(_plot_training_curves(reports, output_dir, formats, plt))
    saved.extend(_plot_prediction_panels(predictions, output_dir, formats, plt))
    saved.extend(_plot_residual_boxplots(predictions, output_dir, formats, plt))
    return saved


def _plot_ablation_metrics(
    metrics_table: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    plt: Any,
) -> list[Path]:
    validation = metrics_table.loc[metrics_table["数据集"] == "validation"].copy()
    if validation.empty:
        return []
    labels = validation["消融实验"].astype(str).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(labels, validation["R2"], color="#3B7EA1")
    axes[0].set_ylabel(r"$\mathrm{R^2}$")
    axes[0].set_title("Validation R2")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.28)

    axes[1].bar(labels, validation["RMSE"], color="#D95F02")
    axes[1].set_ylabel("RMSE (pTox)")
    axes[1].set_title("Validation RMSE")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.28)
    fig.tight_layout()
    return save_figure(fig, output_dir / "深度消融_验证集指标对比", formats=formats, close=True)


def _plot_training_curves(
    reports: list[Mapping[str, Any]],
    output_dir: Path,
    formats: list[str],
    plt: Any,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for report in reports:
        ablation_id = str(report.get("ablation", {}).get("id", "unknown"))
        history = report.get("training", {}).get("history", [])
        if not history:
            continue
        frame = pd.DataFrame(history)
        ax.plot(frame["epoch"], frame["train_loss"], marker="o", label=f"{ablation_id} train")
        ax.plot(frame["epoch"], frame["validation_loss"], marker="s", linestyle="--", label=f"{ablation_id} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (standardized pTox)")
    ax.set_title("Deep Ablation Learning Curves")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir / "深度消融_训练验证损失曲线", formats=formats, close=True)


def _plot_prediction_panels(
    predictions: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    plt: Any,
) -> list[Path]:
    validation = predictions.loc[predictions["数据集"] == "验证集"].copy()
    if validation.empty:
        return []
    ablations = validation["消融实验"].dropna().astype(str).unique().tolist()
    ncols = min(3, max(1, len(ablations)))
    nrows = int((len(ablations) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows), squeeze=False)
    lower = float(min(validation["观测pTox"].min(), validation["预测pTox"].min()))
    upper = float(max(validation["观测pTox"].max(), validation["预测pTox"].max()))
    margin = (upper - lower) * 0.05 if upper > lower else 0.5
    limits = [lower - margin, upper + margin]
    for ax, ablation_id in zip(axes.ravel(), ablations):
        group = validation.loc[validation["消融实验"] == ablation_id]
        plot_group = group.sample(n=min(len(group), 25000), random_state=20260524)
        ax.scatter(
            plot_group["观测pTox"],
            plot_group["预测pTox"],
            s=18,
            alpha=0.55,
            color="#3B7EA1",
            edgecolors="white",
            linewidths=0.3,
        )
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(ablation_id)
        ax.set_xlabel("Observed pTox")
        ax.set_ylabel("Predicted pTox")
        ax.grid(True, linestyle="--", alpha=0.25)
    for ax in axes.ravel()[len(ablations):]:
        ax.axis("off")
    fig.tight_layout()
    return save_figure(fig, output_dir / "深度消融_验证集真实预测散点", formats=formats, close=True)


def _plot_residual_boxplots(
    predictions: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    plt: Any,
) -> list[Path]:
    validation = predictions.loc[predictions["数据集"] == "验证集"].copy()
    if validation.empty:
        return []
    saved: list[Path] = []
    for column, label in (
        ("endpoint_family", "终点类型"),
        ("chemical_class_l2", "化学类别"),
        ("taxon_group_l2", "物种类群"),
    ):
        if column not in validation.columns:
            continue
        top_levels = validation[column].astype("string").fillna("缺失").value_counts().head(8).index
        plot_data = validation.loc[validation[column].astype("string").fillna("缺失").isin(top_levels)].copy()
        if plot_data.empty:
            continue
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        grouped_labels: list[str] = []
        grouped_values: list[pd.Series] = []
        for ablation_id in plot_data["消融实验"].dropna().astype(str).unique():
            group = plot_data.loc[plot_data["消融实验"] == ablation_id]
            for level in top_levels:
                values = group.loc[group[column].astype("string").fillna("缺失") == level, "预测残差"]
                if len(values) >= 5:
                    grouped_labels.append(f"{ablation_id}\n{level}")
                    grouped_values.append(values)
        if not grouped_values:
            plt.close(fig)
            continue
        ax.boxplot(grouped_values, tick_labels=grouped_labels, showfliers=False)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.set_ylabel("Residual (predicted - observed pTox)")
        ax.set_title(f"Validation residuals by {label}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        fig.tight_layout()
        saved.extend(save_figure(fig, output_dir / f"深度消融_验证集残差分层_{label}", formats=formats, close=True))
    return saved


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


def _resolve_ablation_suite(
    config: Mapping[str, Any],
    selected: list[str] | None,
) -> list[dict[str, Any]]:
    suite = config.get("ablation_suite", [])
    if not suite:
        suite = [
            {
                "id": "chemical_only",
                "label": "Chemical only",
                "model": {"context_encoder": {"use_endpoint": False, "use_duration": False}},
                "deep_features": {"species_context_columns": []},
            },
            {
                "id": "chemical_endpoint_duration",
                "label": "Chemical + endpoint + duration",
                "model": {"context_encoder": {"use_endpoint": True, "use_duration": True}},
                "deep_features": {"species_context_columns": []},
            },
            {
                "id": "chemical_species_context",
                "label": "Chemical + endpoint + duration + species",
                "model": {"context_encoder": {"use_endpoint": True, "use_duration": True}},
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
            },
        ]
    normalized_suite = [dict(item) for item in suite if isinstance(item, Mapping)]
    if selected:
        wanted = {str(item) for item in selected}
        normalized_suite = [item for item in normalized_suite if str(item.get("id")) in wanted]
    if not normalized_suite:
        raise ValueError("No ablations selected.")
    return normalized_suite


def _merge_ablation_config(
    base_config: Mapping[str, Any],
    ablation: Mapping[str, Any],
) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    for key in ("model", "training", "deep_features"):
        if key in ablation:
            config[key] = _deep_merge(dict(config.get(key, {})), dict(ablation[key]))
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), dict(value))
        else:
            result[key] = value
    return result


def _build_ablation_metrics_table(reports: list[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in reports:
        ablation = dict(report.get("ablation", {}))
        dataset = dict(report.get("dataset", {}))
        model = dict(report.get("model", {}))
        for split_name, metrics in dict(report.get("metrics", {})).items():
            row = {
                "消融实验": ablation.get("id"),
                "实验标签": ablation.get("label"),
                "数据集": split_name,
                "样本数": dataset.get("train_rows") if split_name == "train" else dataset.get("validation_rows"),
                "化合物数": dataset.get("chemical_count"),
                "描述符维度": model.get("descriptor_group_dim"),
                "指纹维度": model.get("fingerprint_dim"),
                "endpoint维度": model.get("endpoint_dim"),
                "物种上下文维度": model.get("species_context_dim"),
                "启用duration": model.get("use_duration"),
                "启用species": model.get("use_species"),
            }
            row.update(dict(metrics))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _normalize_join_key(data: pd.DataFrame, column: str) -> pd.DataFrame:
    output = data.copy()
    output[column] = output[column].astype("string").str.strip()
    return output


def _value_counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


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
    if "reports" in report:
        metrics: dict[str, Any] = {}
        for item in report.get("reports", []):
            ablation_id = str(item.get("ablation", {}).get("id", "unknown"))
            metrics[ablation_id] = item.get("metrics", {})
        return {
            "report_path": str(report_path),
            "suite": report.get("suite", {}),
            "dataset": report.get("dataset", {}),
            "metrics": metrics,
            "artifacts": report.get("artifacts", {}),
        }
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
