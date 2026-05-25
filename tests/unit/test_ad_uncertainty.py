from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qsar_dl.applicability_domain import (  # noqa: E402
    compute_chemical_ad,
    compute_rule_ad,
    compute_species_ad,
    summarize_ad,
)
from qsar_dl.uncertainty import (  # noqa: E402
    calibrate_intervals,
    ensemble_predict,
    summarize_uncertainty,
)


def test_chemical_ad_combines_descriptor_range_and_tanimoto() -> None:
    reference = pd.DataFrame(
        {
            "mw": [100.0, 200.0],
            "logkow": [1.0, 3.0],
            "fp_0": [1, 0],
            "fp_1": [0, 1],
            "fp_2": [1, 1],
        }
    )
    query = pd.DataFrame(
        {
            "mw": [150.0, 250.0],
            "logkow": [2.0, 2.5],
            "fp_0": [1, 0],
            "fp_1": [0, 0],
            "fp_2": [1, 1],
        }
    )

    result = compute_chemical_ad(
        reference,
        query,
        descriptor_columns=["mw", "logkow"],
        fingerprint_columns=["fp_0", "fp_1", "fp_2"],
        tanimoto_threshold=0.6,
    )

    assert bool(result.loc[0, "descriptor_ad_in_domain"]) is True
    assert result.loc[0, "fingerprint_max_tanimoto"] == pytest.approx(1.0)
    assert bool(result.loc[0, "chemical_ad_in_domain"]) is True
    assert result.loc[1, "descriptor_out_of_range_count"] == 1
    assert result.loc[1, "descriptor_ad_score"] == pytest.approx(0.5)
    assert result.loc[1, "fingerprint_max_tanimoto"] == pytest.approx(0.5)
    assert bool(result.loc[1, "chemical_ad_in_domain"]) is False


def test_species_ad_scores_taxonomy_and_primary_medium_support() -> None:
    reference = pd.DataFrame(
        {
            "phylum_division": ["Arthropoda", "Annelida"],
            "genus": ["Daphnia", "Eisenia"],
            "primary_medium": ["aquatic", "soil"],
        }
    )
    query = pd.DataFrame(
        {
            "phylum_division": ["Arthropoda", "Chordata"],
            "genus": ["Daphnia", "Pimephales"],
            "primary_medium": ["aquatic", "air"],
        }
    )

    result = compute_species_ad(
        reference,
        query,
        taxonomy_columns=["phylum_division", "genus"],
        min_support_score=0.75,
    )

    assert bool(result.loc[0, "species_ad_in_domain"]) is True
    assert result.loc[0, "taxonomy_supported_count"] == 2
    assert bool(result.loc[1, "species_ad_in_domain"]) is False
    assert result.loc[1, "species_ad_score"] == pytest.approx(0.0)
    assert "primary_medium=air" in result.loc[1, "taxonomy_unsupported_fields"]


def test_rule_ad_summarizes_missing_and_applicable_coverage() -> None:
    rule_table = pd.DataFrame(
        {
            "rule_solubility_applicable": [True, False],
            "rule_solubility_missing_inputs": ["", "water_solubility_mg_l"],
            "rule_duration_applicable": [False, True],
            "rule_duration_missing_inputs": ["duration_h", ""],
        }
    )

    result = compute_rule_ad(
        rule_table,
        rule_names=["solubility", "duration"],
        min_applicable_fraction=0.5,
        max_missing_fraction=0.5,
    )

    assert bool(result.loc[0, "rule_ad_in_domain"]) is True
    assert result.loc[0, "rule_applicable_fraction"] == pytest.approx(0.5)
    assert result.loc[0, "rule_missing_fraction"] == pytest.approx(0.5)
    assert result.loc[1, "rule_missing_names"] == "solubility"


def test_summarize_ad_reports_domain_counts_and_scores() -> None:
    table = pd.DataFrame(
        {
            "chemical_ad_in_domain": [True, False, True],
            "species_ad_in_domain": [True, True, False],
            "chemical_ad_score": [1.0, 0.25, 0.8],
        }
    )

    summary = summarize_ad(table)

    assert summary["n_rows"] == 3
    assert summary["domain_columns"]["chemical_ad_in_domain"]["in_domain_count"] == 2
    assert summary["overall_in_domain_count"] == 1
    assert summary["score_columns"]["chemical_ad_score"]["mean"] == pytest.approx(
        (1.0 + 0.25 + 0.8) / 3
    )


def test_ensemble_predict_and_uncertainty_summary_use_precomputed_members() -> None:
    member_predictions = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.2, 1.8, 2.9],
            [0.8, 2.2, 3.1],
        ]
    )

    direct_summary = summarize_uncertainty(member_predictions)
    ensemble_summary = ensemble_predict(member_predictions)

    assert direct_summary["prediction_mean"].to_list() == pytest.approx([1.0, 2.0, 3.0])
    assert ensemble_summary["prediction_q05"].iloc[0] == pytest.approx(0.82)
    assert ensemble_summary["prediction_q95"].iloc[2] == pytest.approx(3.09)
    assert all(ensemble_summary["prediction_n_members"] == 3)


def test_calibrate_intervals_returns_conformal_offset_and_report() -> None:
    y_true = np.array([1.0, 2.0, 4.0])
    predictions = np.array([1.0, 2.0, 3.0])

    calibrated, report = calibrate_intervals(y_true, predictions, coverage=0.8)

    assert report["target_coverage"] == pytest.approx(0.8)
    assert report["n_calibration"] == 3
    assert report["calibration_offset"] == pytest.approx(1.0)
    assert calibrated["interval_lower"].to_list() == pytest.approx([0.0, 1.0, 2.0])
    assert calibrated["interval_upper"].to_list() == pytest.approx([2.0, 3.0, 4.0])
    assert report["empirical_coverage_after"] == pytest.approx(1.0)
