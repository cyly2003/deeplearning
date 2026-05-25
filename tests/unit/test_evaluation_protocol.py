from __future__ import annotations

import math

import pandas as pd
import pytest

from qsar_dl.evaluation import (
    assign_chemical_categories,
    build_category_holdout_splits,
    regression_metrics,
)


def test_assign_chemical_categories_recognizes_protocol_classes() -> None:
    table = pd.DataFrame(
        [
            {"chemical_id": "pah", "chemical_name": "Benzo[a]pyrene"},
            {"chemical_id": "tph", "chemical_name": "Total petroleum hydrocarbons C10-C40"},
            {"chemical_id": "chloro", "chemical_name": "Trichloroethylene"},
            {"chemical_id": "pfas", "chemical_name": "Perfluorooctanoic acid"},
            {"chemical_id": "pesticide", "chemical_name": "Atrazine herbicide"},
            {"chemical_id": "ppcp", "chemical_name": "Ibuprofen pharmaceutical"},
            {"chemical_id": "phenol", "chemical_name": "Bisphenol A"},
            {"chemical_id": "op", "chemical_name": "Chlorpyrifos organophosphate"},
            {"chemical_id": "metal", "chemical_name": "Cadmium chloride"},
            {"chemical_id": "surf", "chemical_name": "Sodium dodecyl sulfate surfactant"},
            {"chemical_id": "dye", "chemical_name": "Methylene blue dye"},
            {"chemical_id": "voc", "chemical_name": "Toluene VOC solvent"},
        ]
    )

    assigned = assign_chemical_categories(table)

    observed = dict(zip(assigned["chemical_id"], assigned["chemical_category"]))
    assert observed == {
        "pah": "PAHs",
        "tph": "TPHs",
        "chloro": "chlorinated_organics",
        "pfas": "PFAS",
        "pesticide": "pesticides",
        "ppcp": "pharmaceuticals_personal_care_products",
        "phenol": "phenols",
        "op": "organophosphates",
        "metal": "metals_metalloids",
        "surf": "surfactants",
        "dye": "dyes",
        "voc": "solvents_vocs",
    }
    assert assigned["category_confidence"].min() >= 0.85
    assert assigned["category_evidence"].str.len().min() > 0


def test_assign_chemical_categories_preserves_existing_category_and_marks_unknown() -> None:
    table = pd.DataFrame(
        [
            {
                "chemical_id": "manual",
                "chemical_name": "Uninformative label",
                "ecotox_group": "PFAS",
            },
            {"chemical_id": "unknown", "chemical_name": "Unspecified mixture"},
        ]
    )

    assigned = assign_chemical_categories(table)

    manual = assigned.loc[assigned["chemical_id"] == "manual"].iloc[0]
    unknown = assigned.loc[assigned["chemical_id"] == "unknown"].iloc[0]
    assert manual["chemical_category"] == "PFAS"
    assert manual["category_confidence"] == 1.0
    assert manual["category_source"] == "existing"
    assert unknown["chemical_category"] == "other_unknown"
    assert unknown["category_confidence"] < 0.5


def test_category_holdout_split_does_not_leak_holdout_category_or_chemical_id() -> None:
    modeling = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 5, 6],
            "chemical_id": ["pfas-1", "pfas-1", "pfas-2", "phenol-1", "phenol-2", "voc-1"],
            "target_ptox": [5.0, 5.2, 4.8, 3.0, 3.3, 2.1],
            "modeling_split_group": [None] * 6,
        }
    )
    categories = pd.DataFrame(
        {
            "chemical_id": ["pfas-1", "pfas-2", "phenol-1", "phenol-2", "voc-1"],
            "chemical_category": ["PFAS", "PFAS", "phenols", "phenols", "solvents_vocs"],
            "category_confidence": [1.0] * 5,
            "category_evidence": ["test fixture"] * 5,
        }
    )

    splits = build_category_holdout_splits(
        modeling,
        categories,
        {
            "evaluation": {
                "holdout_categories": ["PFAS"],
                "validation_fraction_within_train_categories": 0.5,
                "random_seed": 7,
            }
        },
    )

    assert set(splits.loc[splits["chemical_category"] == "PFAS", "split"]) == {"test"}
    non_holdout = splits.loc[splits["chemical_category"] != "PFAS"]
    assert "test" not in set(non_holdout["split"])
    assert set(non_holdout["chemical_category"]) == {"phenols", "solvents_vocs"}

    split_counts_by_chemical = splits.groupby("chemical_id")["split"].nunique()
    assert split_counts_by_chemical.max() == 1
    assert set(splits.loc[splits["split"] != "test", "chemical_category"]).isdisjoint({"PFAS"})
    assert splits.loc[splits["chemical_id"] == "pfas-1", "modeling_split_group"].eq("PFAS").all()


def test_regression_metrics_reports_core_values_and_filters_invalid_rows() -> None:
    metrics = regression_metrics(
        [1.0, 2.0, 4.0, math.nan],
        [1.1, 1.9, 4.2, 10.0],
        prediction_interval_low=[0.8, 1.5, 3.5, 0.0],
        prediction_interval_high=[1.2, 2.5, 4.5, 1.0],
        interval_confidence=0.90,
    )

    assert metrics["n"] == 3
    assert metrics["rmse"] == pytest.approx(math.sqrt((0.1**2 + 0.1**2 + 0.2**2) / 3))
    assert metrics["mae"] == pytest.approx((0.1 + 0.1 + 0.2) / 3)
    assert metrics["mape"] == pytest.approx((0.1 / 1.0 + 0.1 / 2.0 + 0.2 / 4.0) / 3)
    assert metrics["r2"] > 0.98
    assert metrics["spearman_rho"] == pytest.approx(1.0)
    assert metrics["interval_coverage"] == pytest.approx(1.0)
    assert metrics["calibration_error"] == pytest.approx(0.10)
