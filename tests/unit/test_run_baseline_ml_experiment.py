from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.io.parquet import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_baseline_ml_experiment import (  # noqa: E402
    _compute_prediction_ad,
    _prediction_frame,
    _safe_filename,
    build_stratified_metrics_table,
    export_baseline_figures,
    load_baseline_dataset,
)


def require_parquet_engine() -> None:
    try:
        get_engine("auto")
    except ImportError:
        pytest.skip("pyarrow or fastparquet is required for parquet-backed runner tests.")


def test_load_baseline_dataset_normalizes_chemical_id_join_key(tmp_path: Path) -> None:
    require_parquet_engine()
    modeling = pd.DataFrame(
        {
            "chemical_id": [101, 101, 202],
            "target_ptox": [1.0, 1.2, 2.0],
            "target_unit_family": ["water_mg_l", "water_mg_l", "water_mg_l"],
            "is_main_water_task": [True, True, True],
        }
    )
    chemicals = pd.DataFrame(
        {
            "chemical_id": ["101", "202"],
            "rdkit_descriptor_mol_logp": [0.5, 1.5],
        }
    )
    modeling_path = tmp_path / "modeling.parquet"
    chemicals_path = tmp_path / "chemicals.parquet"
    modeling.to_parquet(modeling_path, index=False)
    chemicals.to_parquet(chemicals_path, index=False)

    merged = load_baseline_dataset(
        modeling_table=modeling_path,
        chemical_features=chemicals_path,
        scope="main_water_task",
        max_rows=None,
        random_seed=7,
    )

    assert list(merged["chemical_id"]) == ["101", "101", "202"]
    assert list(merged["rdkit_descriptor_mol_logp"]) == [0.5, 0.5, 1.5]


def test_prediction_frame_records_dataset_label() -> None:
    data = pd.DataFrame({"record_id": [1, 2], "chemical_id": ["A", "B"]})
    y_true = pd.Series([1.0, 2.0], index=[0, 1])

    frame = _prediction_frame(
        data,
        y_true=y_true,
        y_pred=[1.1, 1.8],
        feature_set="standard",
        model_name="random_forest",
        dataset_label="训练集",
    )

    assert list(frame.columns[:3]) == ["特征集", "模型", "数据集"]
    assert frame["数据集"].tolist() == ["训练集", "训练集"]
    assert frame["预测残差"].round(6).tolist() == [0.1, -0.2]


def test_compute_prediction_ad_adds_overall_domain_labels() -> None:
    reference_metadata = pd.DataFrame(
        {
            "taxonomy_class": ["Actinopterygii", "Actinopterygii"],
            "species_ecotox_group": ["fish", "fish"],
            "primary_medium": ["aquatic", "aquatic"],
        }
    )
    query_metadata = pd.DataFrame(
        {
            "taxonomy_class": ["Actinopterygii", "Aves"],
            "species_ecotox_group": ["fish", "bird"],
            "primary_medium": ["aquatic", "terrestrial"],
        }
    )
    reference_features = pd.DataFrame({"x1": [0.0, 1.0], "x2": [2.0, 4.0]})
    query_features = pd.DataFrame({"x1": [0.5, 8.0], "x2": [3.0, 9.0]})

    ad = _compute_prediction_ad(
        reference_metadata=reference_metadata,
        query_metadata=query_metadata,
        reference_features=reference_features,
        query_features=query_features,
    )

    assert ad["总体AD"].tolist() == ["AD内", "AD外"]
    assert ad["特征空间AD内"].tolist() == [True, False]
    assert ad["物种AD内"].tolist() == [True, False]


def test_build_stratified_metrics_table_outputs_requested_dimensions() -> None:
    predictions = pd.DataFrame(
        {
            "特征集": ["standard"] * 8,
            "模型": ["random_forest"] * 8,
            "数据集": ["验证集"] * 8,
            "chemical_id": list("AABBCCDD"),
            "endpoint_family": ["LC", "LC", "EC", "EC", "LC", "LC", "EC", "EC"],
            "chemical_class_l2": ["cat1", "cat1", "cat1", "cat1", "cat2", "cat2", "cat2", "cat2"],
            "taxon_group_l2": ["fish", "fish", "algae", "algae", "fish", "fish", "algae", "algae"],
            "总体AD": ["AD内", "AD内", "AD外", "AD外", "AD内", "AD内", "AD外", "AD外"],
            "观测pTox": [1.0, 1.1, 2.0, 2.1, 1.2, 1.3, 2.2, 2.3],
            "预测pTox": [1.0, 1.2, 2.1, 2.0, 1.1, 1.4, 2.1, 2.4],
            "预测残差": [0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1],
            "绝对误差": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )

    table = build_stratified_metrics_table(predictions)

    assert {"endpoint", "chemical_category", "taxon", "applicability_domain"}.issubset(
        set(table["分层维度"])
    )
    assert table["样本数"].min() >= 2


def test_safe_filename_keeps_chinese_purpose_text() -> None:
    assert _safe_filename("单模型综合诊断图: 真实/预测") == "单模型综合诊断图__真实_预测"


def test_export_baseline_figures_writes_single_and_mosaic_diagnostics(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    data = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "validation"],
            "chemical_id": ["A", "B", "C", "D"],
            "endpoint_family": ["LC", "EC", "LC", "EC"],
        }
    )
    metrics = pd.DataFrame(
        {
            "特征集": ["standard"],
            "模型": ["random_forest"],
            "状态": ["trained"],
            "R2": [0.85],
            "RMSE": [0.2],
            "MAE": [0.1],
        }
    )
    predictions = pd.DataFrame(
        {
            "特征集": ["standard", "standard", "standard", "standard"],
            "模型": ["random_forest", "random_forest", "random_forest", "random_forest"],
            "数据集": ["训练集", "训练集", "验证集", "验证集"],
            "观测pTox": [1.0, 2.0, 1.2, 1.8],
            "预测pTox": [1.1, 1.9, 1.3, 1.7],
            "预测残差": [0.1, -0.1, 0.1, -0.1],
        }
    )

    saved = export_baseline_figures(
        data=data,
        metrics_table=metrics,
        predictions=predictions,
        output_dir=tmp_path,
        formats=["png"],
    )
    saved_names = {path.name for path in saved}

    assert any("单模型综合诊断图_真实预测散点_边缘分布_残差分布" in name for name in saved_names)
    assert any("全部模型综合诊断拼图_真实预测散点_边缘分布_残差分布" in name for name in saved_names)
    assert all(path.exists() and path.stat().st_size > 0 for path in saved)
