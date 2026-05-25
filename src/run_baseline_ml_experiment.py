"""Run traditional ML baselines from standardized ECOTOX modeling outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qsar_dl.evaluation.chemical_categories import assign_chemical_categories
from qsar_dl.evaluation.splits import build_category_holdout_splits
from qsar_dl.training import baseline_ml as baseline_lib
from qsar_dl.training.baseline_ml import (
    build_feature_matrix,
    evaluate_regression,
    train_regressor,
)
from qsar_dl.visualization import get_palette, save_figure, set_publication_style


DEFAULT_MODELING_TABLE = PROJECT_ROOT / "outputs" / "tables" / "modeling_toxicity_long.parquet"
DEFAULT_CHEMICAL_FEATURES = PROJECT_ROOT / "outputs" / "features" / "chemical_features.parquet"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "baseline_ml.yaml"
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation" / "chemical_category_holdout.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "baseline_ml_v001"
PREDICTION_DIAGNOSTIC_STYLE = {
    "train_color": "#8E8D8D",
    "validation_color": "#de5d00",
    "scatter_size": 32,
    "scatter_alpha": 0.78,
    "scatter_edgecolor": "white",
    "scatter_edgewidth": 0.6,
    "hist_alpha": 0.32,
    "hist_bins": 40,
    "hist_edgecolor": "white",
    "hist_edgewidth": 0.8,
    "ideal_line_color": "black",
    "ideal_line_style": "--",
    "ideal_line_width": 1.5,
    "grid_style": "--",
    "grid_alpha": 0.28,
}
AD_TAXONOMY_COLUMNS = (
    "taxonomy_kingdom",
    "taxonomy_phylum",
    "taxonomy_class",
    "taxonomy_order",
    "taxonomy_family",
    "taxonomy_genus",
    "species_ecotox_group",
)
STRATIFICATION_SPECS = (
    ("endpoint", "endpoint_family", "终点类型"),
    ("chemical_category", "chemical_class_l2", "化学结构类别"),
    ("taxon", "taxon_group_l2", "标准生物类群"),
    ("applicability_domain", "总体AD", "适用域"),
)


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)
    evaluation_config = _read_yaml(args.evaluation_config)
    baseline_config = _baseline_config(config)

    if args.models:
        baseline_config["models"] = args.models
    if args.max_rows is not None:
        baseline_config["max_rows"] = args.max_rows

    scope = str(args.scope or baseline_config.get("scope", "main_water_task"))
    data = load_baseline_dataset(
        modeling_table=args.modeling_table,
        chemical_features=args.chemical_features,
        scope=scope,
        max_rows=baseline_config.get("max_rows"),
        random_seed=_random_seed(config, baseline_config),
    )
    data = add_category_splits(data, evaluation_config)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    use_config_split_paths = (
        scope == "main_water_task"
        and baseline_config.get("max_rows") is None
        and output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
    )
    _write_split_artifacts(
        data,
        evaluation_config,
        output_dir=output_dir,
        use_config_paths=use_config_split_paths,
    )

    resolved_config = dict(config)
    resolved_config["baseline_ml"] = baseline_config
    result, predictions = run_baseline_experiment_with_artifacts(data, resolved_config)
    result["dataset"] = _dataset_summary(data)
    result["artifacts"] = {
        "modeling_table": str(args.modeling_table),
        "chemical_features": str(args.chemical_features),
        "output_dir": str(output_dir),
    }

    tables_dir = output_dir / "表格"
    figures_dir = output_dir / "图表"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if args.export_predictions:
        prediction_path = tables_dir / "全量基线_训练与验证集预测结果.parquet"
        predictions.to_parquet(prediction_path, index=False)
        result["artifacts"]["prediction_table"] = str(prediction_path)
        if "数据集" in predictions.columns:
            validation_predictions = predictions.loc[predictions["数据集"] != "训练集"].copy()
            validation_prediction_path = tables_dir / "全量基线_验证集预测结果.parquet"
            validation_predictions.to_parquet(validation_prediction_path, index=False)
            result["artifacts"]["validation_prediction_table"] = str(validation_prediction_path)
    metrics_table = build_metrics_table(result)
    metrics_path = tables_dir / "全量基线_模型指标汇总.csv"
    metrics_table.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    result["artifacts"]["metrics_table"] = str(metrics_path)
    stratified_metrics = build_stratified_metrics_table(predictions)
    if not stratified_metrics.empty:
        stratified_metrics_path = tables_dir / "全量基线_分层指标_endpoint_chemical_category_taxon_AD.csv"
        stratified_metrics.to_csv(stratified_metrics_path, index=False, encoding="utf-8-sig")
        result["artifacts"]["stratified_metrics_table"] = str(stratified_metrics_path)

    if args.export_figures:
        figure_paths = export_baseline_figures(
            data=data,
            metrics_table=metrics_table,
            predictions=predictions,
            output_dir=figures_dir,
            formats=args.figure_formats,
        )
        result["artifacts"]["figures"] = [str(path) for path in figure_paths]

    report_path = output_dir / "baseline_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(result), handle, ensure_ascii=False, indent=2)
    print(json.dumps(_compact_console_summary(result, report_path), ensure_ascii=False, indent=2))


def load_baseline_dataset(
    *,
    modeling_table: Path,
    chemical_features: Path,
    scope: str,
    max_rows: int | None,
    random_seed: int,
) -> pd.DataFrame:
    modeling = pd.read_parquet(modeling_table)
    chemicals = pd.read_parquet(chemical_features)
    _require_columns(modeling, ["chemical_id", "target_ptox", "target_unit_family"])
    _require_columns(chemicals, ["chemical_id"])
    modeling = _normalize_join_key(modeling, "chemical_id")
    chemicals = _normalize_join_key(chemicals, "chemical_id")

    scoped = _filter_scope(modeling, scope)
    feature_columns = [column for column in chemicals.columns if column not in scoped.columns or column == "chemical_id"]
    merged = scoped.merge(chemicals[feature_columns], on="chemical_id", how="left", validate="many_to_one")
    if max_rows is not None:
        if int(max_rows) < 2:
            raise ValueError("max_rows must be >= 2 when provided.")
        if len(merged) > int(max_rows):
            merged = (
                merged.sample(n=int(max_rows), random_state=random_seed)
                .sort_index()
                .reset_index(drop=True)
            )
    return merged.reset_index(drop=True)


def add_category_splits(data: pd.DataFrame, evaluation_config: Mapping[str, Any]) -> pd.DataFrame:
    evaluation = dict(evaluation_config.get("evaluation", evaluation_config))
    chemical_id_column = str(evaluation.get("chemical_id_column", "chemical_id"))
    category_columns = [
        chemical_id_column,
        "casrn",
        "cas_number",
        "dtxsid",
        "smiles",
        "chemical_category",
        "chemical_class_l1",
        "chemical_class_l2",
        "chemical_class_l3",
        "chemical_class_confidence",
        "chemical_class_evidence",
        "ecotox_group",
    ]
    chemical_table = data[[column for column in category_columns if column in data.columns]].drop_duplicates(
        subset=[chemical_id_column]
    )
    if "chemical_class_l2" in chemical_table.columns:
        optional_columns = [
            column
            for column in ("chemical_class_l2", "chemical_class_confidence", "chemical_class_evidence")
            if column in chemical_table.columns
        ]
        curated = chemical_table[[chemical_id_column, *optional_columns]].copy()
        curated["chemical_category"] = curated["chemical_class_l2"].fillna("unknown")
        curated = curated.drop_duplicates(subset=[chemical_id_column])
        if "chemical_class_confidence" in curated.columns:
            curated["category_confidence"] = curated["chemical_class_confidence"]
        else:
            curated["category_confidence"] = 0.80
        if "chemical_class_evidence" in curated.columns:
            curated["category_evidence"] = curated["chemical_class_evidence"]
        else:
            curated["category_evidence"] = "curated_chemical_class_l2"
        curated["category_source"] = "chemical_category_curated"
        return build_category_holdout_splits(data, curated, evaluation_config)
    category_table = assign_chemical_categories(chemical_table, evaluation_config)
    return build_category_holdout_splits(data, category_table, evaluation_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traditional ML baselines on standardized ECOTOX outputs."
    )
    parser.add_argument("--modeling-table", type=Path, default=DEFAULT_MODELING_TABLE)
    parser.add_argument("--chemical-features", type=Path, default=DEFAULT_CHEMICAL_FEATURES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scope",
        choices=["main_water_task", "transfer_model_ready"],
        default=None,
        help="Rows to model after standardized unit filtering.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--no-predictions", dest="export_predictions", action="store_false")
    parser.add_argument("--no-figures", dest="export_figures", action="store_false")
    parser.add_argument("--figure-formats", nargs="+", default=["png", "pdf"])
    parser.set_defaults(export_predictions=True, export_figures=True)
    return parser.parse_args()


def run_baseline_experiment_with_artifacts(
    data: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run configured baselines and keep validation predictions for figures."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    config = dict(config or {})
    baseline_config = dict(config.get("baseline_ml", config.get("baseline", {})))
    target_column = baseline_lib._target_column(config)
    random_state = baseline_lib._random_state(config, baseline_config)
    models = list(baseline_config.get("models", baseline_lib.DEFAULT_MODELS))
    test_size = float(baseline_config.get("test_size", 0.25))
    model_params = dict(baseline_config.get("model_params", {}))

    output: dict[str, Any] = {
        "target_column": target_column,
        "random_state": random_state,
        "feature_sets": {},
    }
    prediction_frames: list[pd.DataFrame] = []

    for feature_config in baseline_lib._feature_set_configs(baseline_config):
        feature_name = str(feature_config.get("name", feature_config.get("type", "standard")))
        matrix = build_feature_matrix(
            data,
            target_column=target_column,
            feature_set=str(feature_config.get("type", feature_name)),
            feature_columns=feature_config.get("feature_columns"),
            context_columns=feature_config.get("context_columns", baseline_config.get("context_columns")),
            species_context_columns=feature_config.get(
                "species_context_columns",
                baseline_config.get("species_context_columns"),
            ),
            descriptor_group_dict=baseline_lib._descriptor_group_dict(
                feature_config, baseline_config, config
            ),
            descriptor_group_path=feature_config.get(
                "descriptor_group_path",
                baseline_config.get("descriptor_group_path"),
            ),
            chemical_id_column=str(
                feature_config.get(
                    "chemical_id_column",
                    baseline_config.get("chemical_id_column", "chemical_id"),
                )
            ),
        )
        X_train, X_test, y_train, y_test = baseline_lib._split_xy(
            data.loc[matrix.X.index],
            matrix.X,
            matrix.y,
            split_column=str(baseline_config.get("split_column", "split")),
            train_values=baseline_config.get("train_values", ("train",)),
            test_values=baseline_config.get("test_values", ("test", "validation", "val")),
            test_size=test_size,
            random_state=random_state,
        )
        train_metadata = data.loc[X_train.index]
        test_metadata = data.loc[X_test.index]
        train_ad = _compute_prediction_ad(
            reference_metadata=train_metadata,
            query_metadata=train_metadata,
            reference_features=X_train,
            query_features=X_train,
        )
        test_ad = _compute_prediction_ad(
            reference_metadata=train_metadata,
            query_metadata=test_metadata,
            reference_features=X_train,
            query_features=X_test,
        )

        model_results: dict[str, dict[str, Any]] = {}
        for model_name in models:
            normalized_name = baseline_lib._normalize_model_name(str(model_name))
            trained = train_regressor(
                normalized_name,
                X_train,
                y_train,
                random_state=random_state,
                model_params=model_params.get(normalized_name, {}),
            )
            if trained.status == "skipped":
                model_results[normalized_name] = {
                    "status": trained.status,
                    "metrics": baseline_lib._empty_metrics(),
                    "skip_reason": trained.skip_reason,
                }
                continue

            y_train_pred = np.asarray(trained.estimator.predict(X_train), dtype="float64").reshape(-1)
            y_pred = np.asarray(trained.estimator.predict(X_test), dtype="float64").reshape(-1)
            model_results[normalized_name] = {
                "status": "trained",
                "metrics": evaluate_regression(y_test, y_pred),
                "skip_reason": None,
            }
            prediction_frames.append(
                _prediction_frame(
                    train_metadata,
                    y_true=y_train,
                    y_pred=y_train_pred,
                    feature_set=feature_name,
                    model_name=normalized_name,
                    dataset_label="训练集",
                    ad_frame=train_ad,
                )
            )
            prediction_frames.append(
                _prediction_frame(
                    test_metadata,
                    y_true=y_test,
                    y_pred=y_pred,
                    feature_set=feature_name,
                    model_name=normalized_name,
                    dataset_label="验证集",
                    ad_frame=test_ad,
                )
            )

        output["feature_sets"][feature_name] = {
            "feature_set": matrix.feature_set,
            "n_features": int(matrix.X.shape[1]),
            "feature_columns": matrix.feature_columns,
            "source_columns": matrix.source_columns,
            "skipped_columns": matrix.skipped_columns,
            "skipped_context_columns": matrix.skipped_context_columns,
            "missing_descriptor_columns": matrix.missing_descriptor_columns,
            "dropped_rows": matrix.dropped_rows,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "models": model_results,
        }

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return output, predictions


def build_metrics_table(result: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name, feature_result in result["feature_sets"].items():
        for model_name, model_result in feature_result["models"].items():
            metrics = model_result["metrics"]
            rows.append(
                {
                    "特征集": feature_name,
                    "模型": model_name,
                    "状态": model_result["status"],
                    "R2": metrics.get("R2"),
                    "RMSE": metrics.get("RMSE"),
                    "MAE": metrics.get("MAE"),
                    "MAPE": metrics.get("MAPE"),
                    "训练样本数": feature_result["train_rows"],
                    "验证样本数": feature_result["test_rows"],
                    "特征数": feature_result["n_features"],
                    "跳过原因": model_result.get("skip_reason"),
                }
            )
    return pd.DataFrame.from_records(rows)


def build_stratified_metrics_table(
    predictions: pd.DataFrame,
    *,
    dataset_values: Iterable[str] = ("验证集", "测试集"),
) -> pd.DataFrame:
    """Build validation metrics stratified by endpoint, chemistry, taxon, and AD."""

    if predictions.empty:
        return pd.DataFrame()
    required = {"特征集", "模型", "观测pTox", "预测pTox"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()

    data = predictions.copy()
    if "数据集" in data.columns:
        data = data.loc[data["数据集"].isin(list(dataset_values))].copy()
    if data.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature_set, model_name in data[["特征集", "模型"]].drop_duplicates().itertuples(index=False):
        model_data = data.loc[(data["特征集"] == feature_set) & (data["模型"] == model_name)]
        for dimension_key, column, dimension_label in STRATIFICATION_SPECS:
            if column not in model_data.columns:
                continue
            values = model_data[column].astype("string").fillna("缺失")
            for stratum, group in model_data.groupby(values, dropna=False, sort=True):
                metrics = evaluate_regression(group["观测pTox"], group["预测pTox"])
                residual = pd.to_numeric(group["预测残差"], errors="coerce")
                rows.append(
                    {
                        "特征集": feature_set,
                        "模型": model_name,
                        "分层维度": dimension_key,
                        "分层名称": dimension_label,
                        "分层字段": column,
                        "分层取值": str(stratum),
                        "样本数": int(len(group)),
                        "化合物数": int(group["chemical_id"].nunique()) if "chemical_id" in group.columns else np.nan,
                        "R2": metrics["R2"],
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"],
                        "MAPE": metrics["MAPE"],
                        "平均残差": float(residual.mean()) if residual.notna().any() else np.nan,
                        "残差标准差": float(residual.std(ddof=1)) if residual.notna().sum() > 1 else np.nan,
                        "绝对误差中位数": float(
                            pd.to_numeric(group["绝对误差"], errors="coerce").median()
                        )
                        if "绝对误差" in group.columns
                        else np.nan,
                    }
                )
    return pd.DataFrame.from_records(rows)


def export_baseline_figures(
    *,
    data: pd.DataFrame,
    metrics_table: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to export baseline figures.") from exc

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    set_publication_style(language="zh", palette="journal")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    saved.extend(_plot_data_flow(data, output_dir, formats, plt))
    saved.extend(_plot_metric_comparison(metrics_table, output_dir, formats, plt))
    if not predictions.empty:
        diagnostic_groups: list[tuple[str, str, pd.DataFrame]] = []
        for (feature_set, model_name), group in predictions.groupby(["特征集", "模型"], sort=True):
            diagnostic_groups.append((str(feature_set), str(model_name), group))
            saved.extend(
                _plot_prediction_diagnostic(
                    group,
                    feature_set=str(feature_set),
                    model_name=str(model_name),
                    output_dir=output_dir,
                    formats=formats,
                    plt=plt,
                    sns=sns,
                )
            )
        saved.extend(
            _plot_prediction_diagnostic_mosaic(
                diagnostic_groups,
                output_dir=output_dir,
                formats=formats,
                plt=plt,
                sns=sns,
            )
        )
        for feature_set, model_name, group in diagnostic_groups:
            saved.extend(
                _plot_stratified_prediction_residuals(
                    group,
                    feature_set=feature_set,
                    model_name=model_name,
                    output_dir=output_dir,
                    formats=formats,
                    plt=plt,
                )
            )
    return saved


def _prediction_frame(
    data: pd.DataFrame,
    *,
    y_true: pd.Series,
    y_pred: np.ndarray,
    feature_set: str,
    model_name: str,
    dataset_label: str,
    ad_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metadata_columns = [
        "record_id",
        "chemical_id",
        "casrn",
        "dtxsid",
        "species_id",
        "scientific_name",
        "taxonomy_kingdom",
        "taxonomy_phylum",
        "taxonomy_class",
        "taxonomy_order",
        "taxonomy_family",
        "taxonomy_genus",
        "taxon_group_l1",
        "taxon_group_l2",
        "taxon_group_l3",
        "is_standard_test_species",
        "is_us_invasive_species",
        "is_us_threatened_endangered",
        "endpoint_family",
        "effect_level",
        "duration_h",
        "primary_medium",
        "organism_lifestage",
        "species_ecotox_group",
        "chemical_category",
        "chemical_class_l1",
        "chemical_class_l2",
        "chemical_class_l3",
        "use_source_class",
        "structure_flags",
        "split",
        "target_unit_family",
    ]
    present_columns = [column for column in metadata_columns if column in data.columns]
    output = data.loc[y_true.index, present_columns].reset_index(drop=True).copy()
    output.insert(0, "模型", model_name)
    output.insert(0, "特征集", feature_set)
    output.insert(2, "数据集", dataset_label)
    output["观测pTox"] = np.asarray(y_true, dtype="float64")
    output["预测pTox"] = np.asarray(y_pred, dtype="float64")
    output["预测残差"] = output["预测pTox"] - output["观测pTox"]
    output["绝对误差"] = output["预测残差"].abs()
    if ad_frame is not None and not ad_frame.empty:
        ad = ad_frame.loc[y_true.index].reset_index(drop=True)
        output = pd.concat([output, ad.reset_index(drop=True)], axis=1)
    return output


def _plot_data_flow(
    data: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    plt: Any,
) -> list[Path]:
    split_counts = data["split"].value_counts(dropna=False) if "split" in data.columns else pd.Series(dtype=int)
    endpoint_counts = (
        data["endpoint_family"].value_counts(dropna=False)
        if "endpoint_family" in data.columns
        else pd.Series(dtype=int)
    )
    labels = ["主任务建模记录", "训练记录", "验证记录", "化合物数量"]
    values = [
        len(data),
        int(split_counts.get("train", 0)),
        int(split_counts.get("validation", 0) + split_counts.get("test", 0)),
        int(data["chemical_id"].nunique()) if "chemical_id" in data.columns else 0,
    ]
    for endpoint in ("LC", "EC", "LOEC"):
        if endpoint in endpoint_counts.index:
            labels.append(f"{endpoint} 记录")
            values.append(int(endpoint_counts[endpoint]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = get_palette("journal", len(labels))
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("数量")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:,}",
            va="center",
            ha="left",
        )
    ax.margins(x=0.14)
    return save_figure(
        fig,
        output_dir / "全量基线_数据筛选与切分流程",
        formats=formats,
        close=True,
    )


def _plot_metric_comparison(
    metrics_table: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    plt: Any,
) -> list[Path]:
    trained = metrics_table.loc[metrics_table["状态"] == "trained"].copy()
    if trained.empty:
        return []
    trained["组合"] = [
        f"{_feature_label(feature)}-{_model_label(model)}"
        for feature, model in zip(trained["特征集"], trained["模型"])
    ]
    metrics = [("R2", "R2 越高越好"), ("RMSE", "RMSE 越低越好"), ("MAE", "MAE 越低越好")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.2), sharey=True)
    colors = get_palette("journal", len(trained))
    for axis_index, (ax, (metric, label)) in enumerate(zip(axes, metrics)):
        values = pd.to_numeric(trained[metric], errors="coerce")
        bars = ax.barh(trained["组合"], values, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        if axis_index > 0:
            ax.tick_params(axis="y", labelleft=False)
        finite_values = values[np.isfinite(values)]
        if not finite_values.empty:
            ax.set_xlim(0.0, float(finite_values.max()) * 1.28)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f" {value:.3f}",
                    va="center",
                    ha="left",
                )
        ax.margins(y=0.04)
    fig.subplots_adjust(left=0.26, right=0.98, wspace=0.14)
    return save_figure(
        fig,
        output_dir / "全量基线_模型性能对比",
        formats=formats,
        close=True,
    )


def _plot_prediction_diagnostic(
    data: pd.DataFrame,
    *,
    feature_set: str,
    model_name: str,
    output_dir: Path,
    formats: Iterable[str],
    plt: Any,
    sns: Any | None,
) -> list[Path]:
    plot_data = _prediction_plot_data(data)
    if plot_data.empty:
        return []

    fig = plt.figure(figsize=(8.0, 10.0), facecolor="white")
    gs = fig.add_gridspec(
        5,
        5,
        wspace=0.10,
        hspace=0.10,
        width_ratios=[3, 3, 3, 3, 2],
        height_ratios=[2, 3, 3, 3, 3],
    )
    _draw_prediction_diagnostic_panel(
        fig=fig,
        grid=gs,
        data=plot_data,
        feature_set=feature_set,
        model_name=model_name,
        sns=sns,
        show_legend=True,
    )
    return save_figure(
        fig,
        output_dir
        / (
            "全量基线_单模型综合诊断图_真实预测散点_边缘分布_残差分布_"
            f"{_safe_filename(_feature_label(feature_set))}_{_safe_filename(_model_label(model_name))}"
        ),
        formats=formats,
        close=True,
    )


def _plot_prediction_diagnostic_mosaic(
    groups: list[tuple[str, str, pd.DataFrame]],
    *,
    output_dir: Path,
    formats: Iterable[str],
    plt: Any,
    sns: Any | None,
) -> list[Path]:
    prepared = [
        (feature_set, model_name, _prediction_plot_data(group))
        for feature_set, model_name, group in groups
    ]
    prepared = [(feature_set, model_name, group) for feature_set, model_name, group in prepared if not group.empty]
    if not prepared:
        return []

    cols = min(3, len(prepared))
    rows = int(np.ceil(len(prepared) / cols))
    fig = plt.figure(figsize=(cols * 6.1, rows * 7.4), facecolor="white")
    outer = fig.add_gridspec(rows, cols, wspace=0.22, hspace=0.24)
    for index, (feature_set, model_name, group) in enumerate(prepared):
        row, col = divmod(index, cols)
        gs = outer[row, col].subgridspec(
            5,
            5,
            wspace=0.08,
            hspace=0.08,
            width_ratios=[3, 3, 3, 3, 2],
            height_ratios=[2, 3, 3, 3, 3],
        )
        _draw_prediction_diagnostic_panel(
            fig=fig,
            grid=gs,
            data=group,
            feature_set=feature_set,
            model_name=model_name,
            sns=sns,
            show_legend=index == 0,
        )

    return save_figure(
        fig,
        output_dir / "全量基线_全部模型综合诊断拼图_真实预测散点_边缘分布_残差分布",
        formats=formats,
        close=True,
    )


def _draw_prediction_diagnostic_panel(
    *,
    fig: Any,
    grid: Any,
    data: pd.DataFrame,
    feature_set: str,
    model_name: str,
    sns: Any | None,
    show_legend: bool,
) -> None:
    style = PREDICTION_DIAGNOSTIC_STYLE
    ax_top = fig.add_subplot(grid[0, 0:4])
    ax_main = fig.add_subplot(grid[1:4, 0:4])
    ax_right = fig.add_subplot(grid[1:4, 4], sharey=ax_main)
    ax_residual = fig.add_subplot(grid[4, 0:4], sharex=ax_main)

    axis_limit = _axis_limit(data["观测pTox"], data["预测pTox"])
    split_styles = _split_plot_styles(data)
    for split_label, split_data in split_styles:
        subset = data.loc[data["数据集"] == split_label]
        if subset.empty:
            continue
        color = split_data["color"]
        label = split_data["label"]
        _draw_histogram(
            ax_top,
            values=subset["观测pTox"],
            color=color,
            orientation="x",
            sns=sns,
        )
        _draw_histogram(
            ax_right,
            values=subset["预测pTox"],
            color=color,
            orientation="y",
            sns=sns,
        )
        ax_main.scatter(
            subset["观测pTox"],
            subset["预测pTox"],
            s=style["scatter_size"],
            color=color,
            alpha=style["scatter_alpha"],
            edgecolors=style["scatter_edgecolor"],
            linewidths=style["scatter_edgewidth"],
            label=label,
        )
        ax_residual.scatter(
            subset["观测pTox"],
            subset["预测残差"],
            s=float(style["scatter_size"]) * 0.70,
            color=color,
            alpha=0.56,
            edgecolors=style["scatter_edgecolor"],
            linewidths=style["scatter_edgewidth"],
        )

    ax_main.plot(
        axis_limit,
        axis_limit,
        color=style["ideal_line_color"],
        linestyle=style["ideal_line_style"],
        linewidth=style["ideal_line_width"],
    )
    ax_main.set_xlim(axis_limit)
    ax_main.set_ylim(axis_limit)
    ax_main.grid(True, linestyle=style["grid_style"], alpha=style["grid_alpha"])
    ax_main.set_ylabel("Predicted pTox")
    _annotate_validation_metrics(ax_main, data)
    if show_legend:
        ax_main.legend(loc="lower right", frameon=False)

    ax_residual.axhline(0.0, color=style["ideal_line_color"], linewidth=1.2)
    ax_residual.grid(True, linestyle=style["grid_style"], alpha=style["grid_alpha"])
    ax_residual.set_xlabel("Observed pTox")
    ax_residual.set_ylabel("Residual")

    ax_top.set_xlim(axis_limit)
    ax_top.axis("off")
    ax_right.set_ylim(axis_limit)
    ax_right.axis("off")


def _prediction_plot_data(data: pd.DataFrame) -> pd.DataFrame:
    plot_data = _sample_for_plot(data, max_points=25000).copy()
    for column in ("观测pTox", "预测pTox", "预测残差"):
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
    if "数据集" not in plot_data.columns:
        plot_data["数据集"] = "验证集"
    finite = plot_data[["观测pTox", "预测pTox", "预测残差"]].notna().all(axis=1)
    return plot_data.loc[finite].copy()


def _split_plot_styles(data: pd.DataFrame) -> list[tuple[str, dict[str, str]]]:
    style = PREDICTION_DIAGNOSTIC_STYLE
    known = {
        "训练集": {"label": "Train", "color": str(style["train_color"])},
        "验证集": {"label": "Validation", "color": str(style["validation_color"])},
        "测试集": {"label": "Test", "color": str(style["validation_color"])},
    }
    output = [(split, known[split]) for split in known if split in set(data["数据集"])]
    for split in data["数据集"].dropna().astype(str).unique():
        if split not in known:
            output.append((split, {"label": split, "color": get_palette("journal", 1)[0]}))
    return output


def _axis_limit(true: pd.Series, pred: pd.Series) -> list[float]:
    lower = float(min(true.min(), pred.min()))
    upper = float(max(true.max(), pred.max()))
    margin = (upper - lower) * 0.05 if upper > lower else 0.5
    return [lower - margin, upper + margin]


def _draw_histogram(
    ax: Any,
    *,
    values: pd.Series,
    color: str,
    orientation: str,
    sns: Any | None,
) -> None:
    style = PREDICTION_DIAGNOSTIC_STYLE
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return
    bins = min(int(style["hist_bins"]), max(8, int(np.sqrt(len(clean))) + 1))
    if sns is not None:
        kwargs = {
            "ax": ax,
            "color": color,
            "kde": len(clean) >= 3,
            "stat": "density",
            "alpha": style["hist_alpha"],
            "bins": bins,
            "edgecolor": style["hist_edgecolor"],
            "linewidth": style["hist_edgewidth"],
        }
        if orientation == "x":
            sns.histplot(x=clean, **kwargs)
        else:
            sns.histplot(y=clean, **kwargs)
        return
    ax.hist(
        clean,
        bins=bins,
        density=True,
        orientation="vertical" if orientation == "x" else "horizontal",
        color=color,
        alpha=style["hist_alpha"],
        edgecolor=style["hist_edgecolor"],
        linewidth=style["hist_edgewidth"],
    )


def _annotate_validation_metrics(ax: Any, data: pd.DataFrame) -> None:
    metric_data = data.loc[data["数据集"] != "训练集"]
    if metric_data.empty:
        metric_data = data
    metrics = evaluate_regression(metric_data["观测pTox"], metric_data["预测pTox"])
    ax.text(
        0.05,
        0.90,
        f"R$^2$={metrics['R2']:.3f}\nRMSE={metrics['RMSE']:.3f}\nMAE={metrics['MAE']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#B2182B",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3.5},
    )


def _plot_stratified_prediction_residuals(
    data: pd.DataFrame,
    *,
    feature_set: str,
    model_name: str,
    output_dir: Path,
    formats: Iterable[str],
    plt: Any,
) -> list[Path]:
    plot_data = _prediction_plot_data(data)
    if "数据集" in plot_data.columns:
        plot_data = plot_data.loc[plot_data["数据集"] != "训练集"].copy()
    if plot_data.empty:
        return []

    saved: list[Path] = []
    for dimension_key, column, dimension_label in STRATIFICATION_SPECS:
        if column not in plot_data.columns:
            continue
        prepared = _prepare_stratified_plot_data(plot_data, column)
        if prepared.empty:
            continue
        colors = get_palette("journal", prepared["_plot_stratum"].nunique())
        color_map = dict(zip(sorted(prepared["_plot_stratum"].unique()), colors))

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
        axis_limit = _axis_limit(prepared["观测pTox"], prepared["预测pTox"])
        for stratum, group in prepared.groupby("_plot_stratum", sort=True):
            axes[0].scatter(
                group["观测pTox"],
                group["预测pTox"],
                s=24,
                alpha=0.72,
                color=color_map[stratum],
                edgecolors="white",
                linewidths=0.45,
                label=str(stratum),
            )
        axes[0].plot(axis_limit, axis_limit, color="black", linestyle="--", linewidth=1.3)
        axes[0].set_xlim(axis_limit)
        axes[0].set_ylim(axis_limit)
        axes[0].set_xlabel("Observed pTox")
        axes[0].set_ylabel("Predicted pTox")
        axes[0].grid(True, linestyle="--", alpha=0.28)

        ordered = sorted(prepared["_plot_stratum"].unique())
        residual_groups = [
            prepared.loc[prepared["_plot_stratum"] == stratum, "预测残差"].to_numpy(dtype=float)
            for stratum in ordered
        ]
        box = axes[1].boxplot(
            residual_groups,
            tick_labels=ordered,
            patch_artist=True,
            showfliers=False,
        )
        for patch, stratum in zip(box["boxes"], ordered):
            patch.set_facecolor(color_map[stratum])
            patch.set_alpha(0.55)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        axes[1].set_ylabel("Residual")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.28)
        axes[1].tick_params(axis="x", rotation=35)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="best", frameon=False, fontsize=9)
        saved.extend(
            save_figure(
                fig,
                output_dir
                / (
                    "全量基线_分层真实预测残差图_"
                    f"{_safe_filename(dimension_label)}_"
                    f"{_safe_filename(_feature_label(feature_set))}_"
                    f"{_safe_filename(_model_label(model_name))}"
                ),
                formats=formats,
                close=True,
            )
        )
    return saved


def _prepare_stratified_plot_data(
    data: pd.DataFrame,
    column: str,
    *,
    max_levels: int = 8,
    max_points: int = 25000,
) -> pd.DataFrame:
    clean = _sample_for_plot(data, max_points=max_points).copy()
    clean["_plot_stratum"] = clean[column].astype("string").fillna("缺失")
    counts = clean["_plot_stratum"].value_counts(dropna=False)
    keep = set(counts.head(max_levels).index)
    clean["_plot_stratum"] = clean["_plot_stratum"].where(clean["_plot_stratum"].isin(keep), "其他")
    return clean


def _compute_prediction_ad(
    *,
    reference_metadata: pd.DataFrame,
    query_metadata: pd.DataFrame,
    reference_features: pd.DataFrame,
    query_features: pd.DataFrame,
) -> pd.DataFrame:
    feature_ad = _compute_feature_range_ad(reference_features, query_features)
    species_ad = _compute_species_support_ad(reference_metadata, query_metadata)
    output = pd.concat([feature_ad, species_ad], axis=1)
    domain_columns = [
        column
        for column in ("特征空间AD内", "物种AD内")
        if column in output.columns
    ]
    if domain_columns:
        overall = output[domain_columns].apply(lambda row: all(bool(value) for value in row), axis=1)
        output["总体AD内"] = overall.astype(bool)
        output["总体AD"] = np.where(output["总体AD内"], "AD内", "AD外")
    else:
        output["总体AD内"] = False
        output["总体AD"] = "AD未评估"
    return output


def _compute_feature_range_ad(
    reference_features: pd.DataFrame,
    query_features: pd.DataFrame,
    *,
    tolerance: float = 0.05,
    min_score: float = 0.95,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    columns = [column for column in reference_features.columns if column in query_features.columns]
    if not columns:
        return pd.DataFrame(
            {
                "特征空间AD分数": np.nan,
                "特征空间AD内": False,
                "特征空间AD评价特征数": 0,
            },
            index=query_features.index,
        )

    reference = reference_features[columns].apply(pd.to_numeric, errors="coerce")
    lower = reference.min(axis=0, skipna=True).to_numpy(dtype=float)
    upper = reference.max(axis=0, skipna=True).to_numpy(dtype=float)
    finite_bounds = np.isfinite(lower) & np.isfinite(upper)
    span = upper - lower
    margin = np.where(span > 0, span * tolerance, np.abs(lower) * tolerance)
    lower = lower - margin
    upper = upper + margin
    reference_count = int(finite_bounds.sum())
    scores = np.full(len(query_features), np.nan, dtype=float)
    evaluated_counts = np.zeros(len(query_features), dtype=int)
    in_range_counts = np.zeros(len(query_features), dtype=int)

    for start in range(0, len(query_features), chunk_size):
        stop = min(start + chunk_size, len(query_features))
        chunk = query_features.iloc[start:stop][columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(chunk) & finite_bounds
        in_range = finite & (chunk >= lower) & (chunk <= upper)
        evaluated = finite.sum(axis=1)
        in_counts = in_range.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            available_fraction = evaluated / reference_count if reference_count else np.nan
            in_range_fraction = in_counts / evaluated
            chunk_scores = available_fraction * in_range_fraction
        scores[start:stop] = chunk_scores
        evaluated_counts[start:stop] = evaluated
        in_range_counts[start:stop] = in_counts

    return pd.DataFrame(
        {
            "特征空间AD分数": scores,
            "特征空间AD内": np.isfinite(scores) & (scores >= min_score),
            "特征空间AD评价特征数": evaluated_counts,
            "特征空间AD范围内特征数": in_range_counts,
            "特征空间AD参考特征数": reference_count,
        },
        index=query_features.index,
    )


def _compute_species_support_ad(
    reference_metadata: pd.DataFrame,
    query_metadata: pd.DataFrame,
    *,
    min_score: float = 0.5,
) -> pd.DataFrame:
    columns = [column for column in AD_TAXONOMY_COLUMNS if column in reference_metadata.columns or column in query_metadata.columns]
    support_sets = {
        column: _normalized_values(reference_metadata[column])
        for column in columns
        if column in reference_metadata.columns
    }
    supported_columns = [column for column in columns if support_sets.get(column)]
    medium_support = (
        _normalized_values(reference_metadata["primary_medium"])
        if "primary_medium" in reference_metadata.columns
        else set()
    )

    records: list[dict[str, Any]] = []
    for _idx, row in query_metadata.iterrows():
        supported = 0
        evaluated = 0
        for column in supported_columns:
            value = _clean_label(row.get(column))
            if value is None:
                continue
            evaluated += 1
            if value in support_sets[column]:
                supported += 1
        taxonomy_score = supported / len(supported_columns) if supported_columns else np.nan
        medium_value = _clean_label(row.get("primary_medium")) if "primary_medium" in query_metadata.columns else None
        medium_score = 1.0 if medium_support and medium_value in medium_support else 0.0 if medium_support else np.nan
        components = [value for value in (taxonomy_score, medium_score) if np.isfinite(value)]
        score = float(np.mean(components)) if components else np.nan
        records.append(
            {
                "物种AD分数": score,
                "物种AD内": bool(np.isfinite(score) and score >= min_score),
                "物种AD分类支持数": supported,
                "物种AD分类评价数": evaluated,
                "物种AD分类参考层级数": len(supported_columns),
                "物种AD介质支持": bool(medium_support and medium_value in medium_support),
            }
        )
    return pd.DataFrame.from_records(records, index=query_metadata.index)


def _normalized_values(series: pd.Series) -> set[str]:
    return {label for value in series if (label := _clean_label(value)) is not None}


def _clean_label(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "missing", "na"}:
        return None
    return text.lower()


def _filter_scope(modeling: pd.DataFrame, scope: str) -> pd.DataFrame:
    normalized = scope.strip().lower()
    if normalized == "main_water_task":
        _require_columns(modeling, ["is_main_water_task"])
        mask = modeling["is_main_water_task"].fillna(False)
    elif normalized == "transfer_model_ready":
        _require_columns(modeling, ["is_transfer_model_ready"])
        mask = modeling["is_transfer_model_ready"].fillna(False)
    else:
        raise ValueError(f"Unsupported scope: {scope}")

    mask = mask & modeling["target_ptox"].notna()
    filtered = modeling.loc[mask].copy()
    if filtered.empty:
        raise ValueError(f"No rows available for baseline scope={scope!r}.")
    return filtered


def _write_split_artifacts(
    data: pd.DataFrame,
    evaluation_config: Mapping[str, Any],
    *,
    output_dir: Path,
    use_config_paths: bool,
) -> None:
    evaluation = dict(evaluation_config.get("evaluation", evaluation_config))
    category_path = (
        Path(str(evaluation["category_assignment_table"]))
        if use_config_paths and evaluation.get("category_assignment_table")
        else output_dir / "chemical_category_assignments.csv"
    )
    split_path = (
        Path(str(evaluation["split_table"]))
        if use_config_paths and evaluation.get("split_table")
        else output_dir / "data_splits.parquet"
    )
    report_path = (
        Path(str(evaluation["report_json"]))
        if use_config_paths and evaluation.get("report_json")
        else output_dir / "split_report.json"
    )
    if category_path:
        category_columns = [
            "chemical_id",
            "chemical_category",
            "category_confidence",
            "category_evidence",
            "category_source",
        ]
        categories = data[[column for column in category_columns if column in data.columns]].drop_duplicates(
            subset=["chemical_id"]
        )
        _ensure_parent(category_path)
        categories.to_csv(category_path, index=False, encoding="utf-8-sig")
    if split_path:
        split_columns = [
            "record_id",
            "chemical_id",
            "chemical_category",
            "split",
            "split_strategy",
            "holdout_category_flag",
        ]
        splits = data[[column for column in split_columns if column in data.columns]].copy()
        _ensure_parent(split_path)
        splits.to_parquet(split_path, index=False)
    if report_path:
        _ensure_parent(report_path)
        report = {
            "row_count": int(len(data)),
            "chemical_count": int(data["chemical_id"].nunique()),
            "split_counts": _value_counts(data.get("split")),
            "category_counts": _value_counts(data.get("chemical_category")),
        }
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


def _dataset_summary(data: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(data)),
        "chemical_count": int(data["chemical_id"].nunique()),
        "target_ptox_non_missing": int(data["target_ptox"].notna().sum()),
        "target_unit_family_counts": _value_counts(data.get("target_unit_family")),
        "split_counts": _value_counts(data.get("split")),
        "category_counts": _value_counts(data.get("chemical_category")),
        "endpoint_family_counts": _value_counts(data.get("endpoint_family")),
    }


def _compact_console_summary(result: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    model_metrics: dict[str, Any] = {}
    for feature_name, feature_result in result["feature_sets"].items():
        model_metrics[feature_name] = {
            model_name: model_result["metrics"]
            for model_name, model_result in feature_result["models"].items()
            if model_result["status"] == "trained"
        }
    return {
        "report_path": str(report_path),
        "dataset": result.get("dataset", {}),
        "model_metrics": model_metrics,
    }


def _baseline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("baseline_ml", config.get("baseline", {})))


def _random_seed(config: Mapping[str, Any], baseline_config: Mapping[str, Any]) -> int:
    if "random_state" in baseline_config:
        return int(baseline_config["random_state"])
    experiment = config.get("experiment")
    if isinstance(experiment, Mapping) and "seed" in experiment:
        return int(experiment["seed"])
    return 20260524


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return dict(data)


def _require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_join_key(data: pd.DataFrame, column: str) -> pd.DataFrame:
    output = data.copy()
    output[column] = output[column].astype("string").str.strip()
    return output


def _sample_for_plot(data: pd.DataFrame, *, max_points: int) -> pd.DataFrame:
    if len(data) <= max_points:
        return data
    return data.sample(n=max_points, random_state=20260524).sort_index()


def _feature_label(value: object) -> str:
    labels = {
        "standard": "标准特征",
        "fixed_descriptor_groups": "固定描述符分组",
    }
    return labels.get(str(value), str(value))


def _model_label(value: object) -> str:
    labels = {
        "pls": "PLS",
        "elasticnet": "ElasticNet",
        "random_forest": "随机森林",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "svr": "SVR",
    }
    return labels.get(str(value), str(value))


def _safe_filename(value: str) -> str:
    forbidden = '<>:"/\\|?*'
    output = "".join("_" if char in forbidden else char for char in value)
    return output.strip().replace(" ", "_")


def _value_counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _jsonable(value: Any) -> Any:
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
