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

from qsar_dl.features import chemical  # noqa: E402


def small_config() -> dict:
    return {
        "fingerprints": {
            "enabled": True,
            "radius": 2,
            "n_bits": 8,
            "prefix": "morgan_fp_",
        },
        "output": {"write": False},
    }


def test_module_imports_without_requiring_rdkit() -> None:
    assert callable(chemical.normalize_smiles)
    assert callable(chemical.compute_rdkit_descriptors)
    assert callable(chemical.compute_fingerprints)
    assert callable(chemical.build_chemical_features)


def test_normalize_smiles_strips_when_rdkit_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chemical, "_load_rdkit", lambda: None)

    assert chemical.normalize_smiles(" CCO ") == "CCO"
    assert chemical.normalize_smiles("") is None


def test_rdkit_descriptor_missing_flags_when_rdkit_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chemical, "_load_rdkit", lambda: None)

    result = chemical.compute_rdkit_descriptors("CCO")

    assert result["rdkit_available"] is False
    assert result["structure_lookup_status"] == "rdkit_unavailable"
    assert result["descriptor_missing_flag"] is True
    assert np.isnan(result["molecular_weight_rdkit_g_mol"])


def test_fingerprint_returns_zero_vector_when_rdkit_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chemical, "_load_rdkit", lambda: None)

    fingerprint = chemical.compute_fingerprints("CCO", radius=2, n_bits=16)

    assert fingerprint.shape == (16,)
    assert fingerprint.dtype == np.int8
    assert int(fingerprint.sum()) == 0


def test_build_chemical_features_preserves_rows_and_missing_masks_without_rdkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chemical, "_load_rdkit", lambda: None)
    input_table = pd.DataFrame(
        {
            "chemical_id": ["chem-1", "chem-2"],
            "casrn": ["64-17-5", "bad-cas"],
            "smiles": ["CCO", "not-a-smiles"],
        }
    )

    features = chemical.build_chemical_features(input_table, small_config())

    assert features.shape[0] == 2
    assert set(features["chemical_id"]) == {"chem-1", "chem-2"}
    assert features["descriptor_missing_flag"].tolist() == [True, True]
    assert features["fingerprint_missing_flag"].tolist() == [True, True]
    assert "rdkit_descriptors" in features.loc[0, "chemical_missing_fields"]
    assert all(column in features for column in ["morgan_fp_0000", "morgan_fp_0007"])
    assert int(features.filter(regex=r"^morgan_fp_").to_numpy().sum()) == 0


def test_build_chemical_features_uses_external_physchem_and_flags_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chemical, "_load_rdkit", lambda: None)
    input_table = pd.DataFrame(
        {
            "chemical_id": ["chem-1"],
            "smiles": ["CCO"],
            "molecular_weight_g_mol": [46.069],
            "logd": [0.1],
            "water_solubility": [1_000_000],
            "koc": [2.5],
        }
    )

    features = chemical.build_chemical_features(input_table, small_config())
    row = features.iloc[0]

    assert row["molecular_weight_g_mol"] == pytest.approx(46.069)
    assert row["molecular_weight_source"] == "molecular_weight_g_mol"
    assert bool(row["logd_missing_flag"]) is False
    assert bool(row["water_solubility_missing_flag"]) is False
    assert bool(row["koc_missing_flag"]) is False
    assert bool(row["henry_law_constant_missing_flag"]) is True
    assert "henry_law_constant" in row["chemical_missing_fields"]


def test_build_chemical_features_deduplicates_by_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chemical, "_load_rdkit", lambda: None)
    input_table = pd.DataFrame(
        {
            "chemical_id": ["chem-1", "chem-1", "chem-2"],
            "smiles": ["CCO", "CCO", "O"],
        }
    )

    features = chemical.build_chemical_features(input_table, small_config())

    assert features["chemical_id"].tolist() == ["chem-1", "chem-2"]


def test_config_file_can_be_loaded() -> None:
    config_path = PROJECT_ROOT / "configs" / "features" / "chemical_rdkit_morgan.yaml"
    input_table = pd.DataFrame({"chemical_id": ["chem-1"], "smiles": ["CCO"]})

    features = chemical.build_chemical_features(input_table, config_path)

    assert features.shape[0] == 1
    assert "morgan_fp_0000" in features.columns


def test_valid_rdkit_descriptor_values_when_rdkit_is_installed() -> None:
    if chemical._load_rdkit() is None:
        pytest.skip("RDKit is not installed in this environment.")

    result = chemical.compute_rdkit_descriptors("CCO")

    assert result["structure_lookup_status"] == "ok"
    assert result["descriptor_missing_flag"] is False
    assert result["normalized_smiles"] == "CCO"
    assert result["molecular_weight_rdkit_g_mol"] == pytest.approx(46.069, rel=1e-3)


def test_invalid_smiles_is_flagged_when_rdkit_is_installed() -> None:
    if chemical._load_rdkit() is None:
        pytest.skip("RDKit is not installed in this environment.")

    result = chemical.compute_rdkit_descriptors("not-a-smiles")

    assert result["structure_lookup_status"] == "invalid_smiles"
    assert result["smiles_parse_error_flag"] is True
    assert result["descriptor_missing_flag"] is True
