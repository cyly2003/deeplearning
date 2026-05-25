from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qsar_dl.training.baseline_ml import (  # noqa: E402
    METRIC_KEYS,
    build_feature_matrix,
    evaluate_regression,
    run_baseline_experiment,
    train_regressor,
)


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        pytest.skip(f"scikit-learn stack unavailable in this interpreter: {exc}")


@pytest.fixture()
def small_modeling_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chemical_id": [f"chem_{idx}" for idx in range(8)],
            "MolLogP": [0.1, 0.4, 0.8, 1.2, 1.7, 2.1, 2.5, 3.0],
            "MolMR": [1.0, 1.1, 1.5, 1.8, 2.0, 2.4, 2.8, 3.1],
            "TPSA": [10.0, 12.0, 14.0, 17.0, 19.0, 21.0, 23.0, 25.0],
            "duration_h": [24.0, 24.0, 48.0, 48.0, 72.0, 72.0, 96.0, 96.0],
            "primary_medium": ["water", "water", "water", "soil", "soil", "soil", "water", "soil"],
            "split": ["train", "train", "train", "train", "train", "train", "test", "test"],
            "target_ptox": [1.0, 1.3, 1.6, 2.0, 2.3, 2.6, 2.9, 3.2],
        }
    )


@pytest.fixture()
def descriptor_group_dict() -> dict:
    return {
        "descriptor_source": "rdkit",
        "standardization": {
            "method": "robust_zscore",
            "missing_strategy": "train_median_with_mask",
        },
        "groups": {
            "partition": {
                "description": "Partition descriptors.",
                "initial_group_weight": 1.0,
                "bias_init": 0.0,
                "descriptors": {
                    "MolLogP": {"role": "core", "initial_weight": 1.0},
                    "MolMR": {"role": "auxiliary", "initial_weight": 0.5},
                },
            },
            "polarity": {
                "description": "Polarity descriptors.",
                "initial_group_weight": 1.0,
                "bias_init": 0.0,
                "descriptors": {
                    "TPSA": {"role": "core", "initial_weight": 1.0},
                    "NumHDonors": {"role": "core", "initial_weight": 1.0},
                },
            },
        },
    }


def test_build_standard_feature_matrix_encodes_context_and_skips_missing(
    small_modeling_df: pd.DataFrame,
) -> None:
    matrix = build_feature_matrix(
        small_modeling_df,
        feature_set="standard",
        feature_columns=["MolLogP", "TPSA"],
        context_columns=["primary_medium", "organism_lifestage"],
    )

    assert list(matrix.y) == list(small_modeling_df["target_ptox"])
    assert {"MolLogP", "TPSA"}.issubset(matrix.feature_columns)
    assert "primary_medium_soil" in matrix.feature_columns
    assert "primary_medium_water" in matrix.feature_columns
    assert matrix.skipped_context_columns == ["organism_lifestage"]
    assert matrix.dropped_rows == 0


def test_build_fixed_descriptor_group_feature_matrix_reports_missing_descriptors(
    small_modeling_df: pd.DataFrame,
    descriptor_group_dict: dict,
) -> None:
    matrix = build_feature_matrix(
        small_modeling_df,
        feature_set="fixed_descriptor_groups",
        descriptor_group_dict=descriptor_group_dict,
        context_columns=["primary_medium", "missing_context"],
    )

    assert "desc_group_partition" in matrix.feature_columns
    assert "desc_group_partition_missing_rate" in matrix.feature_columns
    assert "desc_group_polarity" in matrix.feature_columns
    assert "NumHDonors" in matrix.missing_descriptor_columns
    assert matrix.skipped_context_columns == ["missing_context"]
    assert matrix.X.shape[0] == len(small_modeling_df)


def test_train_regressor_and_evaluate_regression_return_required_metrics(
    small_modeling_df: pd.DataFrame,
) -> None:
    require_sklearn()
    matrix = build_feature_matrix(
        small_modeling_df,
        feature_set="standard",
        feature_columns=["MolLogP", "MolMR", "TPSA"],
        context_columns=[],
    )
    trained = train_regressor(
        "elasticnet",
        matrix.X.iloc[:6],
        matrix.y.iloc[:6],
        random_state=7,
    )

    predictions = trained.estimator.predict(matrix.X.iloc[6:])
    metrics = evaluate_regression(matrix.y.iloc[6:], predictions)

    assert trained.status == "trained"
    assert set(metrics) == set(METRIC_KEYS)
    assert all(key in metrics for key in ("R2", "RMSE", "MAE", "MAPE"))
    assert metrics["RMSE"] >= 0.0


def test_optional_lightgbm_dependency_is_skipped_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    small_modeling_df: pd.DataFrame,
) -> None:
    require_sklearn()
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "lightgbm":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    matrix = build_feature_matrix(
        small_modeling_df,
        feature_set="standard",
        feature_columns=["MolLogP", "MolMR"],
        context_columns=[],
    )

    result = train_regressor("lightgbm", matrix.X.iloc[:6], matrix.y.iloc[:6])

    assert result.status == "skipped"
    assert result.estimator is None
    assert "lightgbm" in result.skip_reason


def test_run_baseline_experiment_trains_and_reports_skipped_optional_model(
    monkeypatch: pytest.MonkeyPatch,
    small_modeling_df: pd.DataFrame,
    descriptor_group_dict: dict,
) -> None:
    require_sklearn()
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "lightgbm":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    config = {
        "target": {"column": "target_ptox"},
        "experiment": {"seed": 7},
        "baseline_ml": {
            "models": ["pls", "random_forest", "lightgbm"],
            "split_column": "split",
            "context_columns": ["primary_medium", "organism_lifestage"],
            "feature_sets": [
                {
                    "name": "standard",
                    "type": "standard",
                    "feature_columns": ["MolLogP", "MolMR", "TPSA"],
                },
                {
                    "name": "fixed_groups",
                    "type": "fixed_descriptor_groups",
                    "descriptor_group_dict": descriptor_group_dict,
                },
            ],
            "model_params": {
                "random_forest": {"n_estimators": 5, "random_state": 7},
            },
        },
    }

    result = run_baseline_experiment(small_modeling_df, config)

    standard = result["feature_sets"]["standard"]
    fixed_groups = result["feature_sets"]["fixed_groups"]
    assert standard["train_rows"] == 6
    assert standard["test_rows"] == 2
    assert standard["skipped_context_columns"] == ["organism_lifestage"]
    assert fixed_groups["missing_descriptor_columns"] == ["NumHDonors"]
    assert standard["models"]["pls"]["status"] == "trained"
    assert standard["models"]["random_forest"]["status"] == "trained"
    assert standard["models"]["lightgbm"]["status"] == "skipped"
    assert set(standard["models"]["pls"]["metrics"]) == set(METRIC_KEYS)
    assert math.isfinite(standard["models"]["random_forest"]["metrics"]["RMSE"])
