from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qsar_dl.data import (  # noqa: E402
    build_modeling_table,
    derive_concentration,
    derive_duration,
    load_clean_sqlite,
    parse_endpoint,
    standardize_target_units,
)


def test_derive_concentration_prefers_mean() -> None:
    result = derive_concentration(
        {
            "conc1_mean": "12.5",
            "conc1_min": "1",
            "conc1_max": "20",
            "conc1_unit": "mg/L",
            "num_doses_mean": "5",
        }
    )

    assert result["conc_value"] == 12.5
    assert result["conc_derivation_method"] == "mean"
    assert result["conc_unit"] == "mg/L"


def test_derive_concentration_range_odd_num_doses() -> None:
    result = derive_concentration(
        {
            "conc1_min": "0",
            "conc1_max": "8",
            "conc1_unit": "mg/L",
            "num_doses_mean": "5",
        }
    )

    assert result["conc_value"] == 4.0
    assert result["num_doses_used"] == 5.0
    assert result["conc_derivation_method"] == "direct_range_midpoint"


def test_derive_concentration_range_even_num_doses() -> None:
    result = derive_concentration(
        {
            "conc1_min": "0",
            "conc1_max": "9",
            "conc1_unit": "mg/L",
            "num_doses_mean": "4",
        }
    )

    assert result["conc_value"] == 4.5
    assert result["num_doses_used"] == 4.0


def test_derive_concentration_swaps_reversed_range_and_defaults_num_doses() -> None:
    result = derive_concentration(
        {
            "conc1_min": "10",
            "conc1_max": "2",
            "conc1_unit": "mg/L",
            "num_doses_mean": "1",
        }
    )

    assert result["conc_value"] == 6.0
    assert "conc_min_gt_max_swapped" in result["qa_flags"]
    assert "num_doses_invalid_defaulted" in result["qa_flags"]
    assert result["num_doses_used"] == 4.0


def test_derive_concentration_uses_num_doses_min_max_mean() -> None:
    result = derive_concentration(
        {
            "conc1_min": "1",
            "conc1_max": "5",
            "conc1_unit": "mg/L",
            "num_doses_min": "3",
            "num_doses_max": "5",
        }
    )

    assert result["num_doses_used"] == 4.0
    assert result["conc_value"] == 3.0


def test_derive_concentration_records_censored_operator() -> None:
    result = derive_concentration(
        {
            "conc1_mean_op": "<",
            "conc1_mean": "0.1",
            "conc1_unit": "mg/L",
        }
    )

    assert result["conc_value"] == 0.1
    assert "conc1_mean_censored_lt" in result["qa_flags"]


def test_derive_concentration_prefers_standardized_value_and_keeps_original_unit() -> None:
    result = derive_concentration(
        {
            "conc1_mean": "1000",
            "conc1_unit": "ug/L",
            "conc1_mean_standardized": 1.0,
            "conc1_standard_unit": "mg/L",
            "conc1_standardization_status": "standardized",
        }
    )

    assert result["conc_value"] == 1.0
    assert result["conc_unit"] == "mg/L"
    assert result["conc_unit_original"] == "ug/L"


def test_derive_duration_exposure_mean_days_to_hours() -> None:
    result = derive_duration(
        {
            "exposure_duration_mean": "2",
            "exposure_duration_unit": "d",
        }
    )

    assert result["duration_h"] == 48.0
    assert result["duration_derivation_method"] == "exposure_mean"
    assert result["duration_missing_flag"] is False


def test_derive_duration_prefers_standardized_hours() -> None:
    result = derive_duration(
        {
            "exposure_duration_mean": "2",
            "exposure_duration_unit": "d",
            "exposure_duration_mean_h": 48.0,
        }
    )

    assert result["duration_h"] == 48.0
    assert result["duration_derivation_method"] == "exposure_mean"


def test_derive_duration_observation_mean_fallback() -> None:
    result = derive_duration(
        {
            "obs_duration_mean": "90",
            "obs_duration_unit": "min",
        }
    )

    assert result["duration_h"] == 1.5
    assert result["duration_derivation_method"] == "observation_mean"


def test_derive_duration_range_grid_midpoint() -> None:
    result = derive_duration(
        {
            "exposure_duration_min": "1",
            "exposure_duration_max": "3",
            "exposure_duration_unit": "d",
            "num_doses_mean": "3",
        }
    )

    assert result["duration_h"] == 48.0
    assert result["duration_derivation_method"] == "exposure_range_grid_mid"


def test_derive_duration_missing_manual_review() -> None:
    result = derive_duration({"exposure_duration_mean": None})

    assert result["duration_h"] is None
    assert result["duration_derivation_method"] == "missing_manual_review"
    assert result["duration_missing_flag"] is True


@pytest.mark.parametrize(
    ("raw_endpoint", "family", "level"),
    [
        ("LC50", "LC", 50.0),
        ("EC10", "EC", 10.0),
        ("LOEC", "LOEC", None),
        ("NOEC", None, None),
    ],
)
def test_parse_endpoint(raw_endpoint: str, family: str | None, level: float | None) -> None:
    result = parse_endpoint(raw_endpoint)

    assert result["endpoint_family"] == family
    assert result["effect_level"] == level


def test_standardize_target_units_mg_l_to_ptox() -> None:
    result = standardize_target_units(
        {
            "conc_value": 10.0,
            "conc_unit": "mg/L",
            "molecular_weight_g_mol": 100.0,
        }
    )

    assert result["target_unit_family"] == "water_mg_l"
    assert result["target_mg_l"] == 10.0
    assert result["target_mol_l"] == 0.0001
    assert result["target_ptox"] == 4.0


def test_standardize_target_units_ug_l() -> None:
    result = standardize_target_units(
        {
            "conc_value": 1000.0,
            "conc_unit": "ug/L",
            "molecular_weight_g_mol": 100.0,
        }
    )

    assert result["target_mg_l"] == 1.0
    assert result["target_mol_l"] == 0.00001


def test_standardize_target_units_oral_daily() -> None:
    result = standardize_target_units(
        {
            "conc_value": 5.0,
            "conc_unit": "mg/kg/d",
            "primary_medium": "terrestrial",
        }
    )

    assert result["target_unit_family"] == "oral_mg_kg_d"
    assert result["target_mg_kg_d"] == 5.0
    assert result["target_ptox"] is None


def test_standardize_target_units_mass_per_mass_to_mg_kg() -> None:
    result = standardize_target_units(
        {
            "conc_value": 1000.0,
            "conc_unit": "ug/kg",
            "primary_medium": "soil",
        }
    )

    assert result["target_unit_family"] == "soil_mg_kg"
    assert result["target_mg_kg"] == 1.0
    assert result["target_ptox"] is None


def test_standardize_target_units_unknown_unit() -> None:
    result = standardize_target_units(
        {
            "conc_value": 5.0,
            "conc_unit": "widgets",
            "molecular_weight_g_mol": 100.0,
        }
    )

    assert result["target_unit_family"] == "other"
    assert result["target_ptox"] is None


def test_load_clean_sqlite_missing_path_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError, match="Clean ECOTOX SQLite not found"):
        load_clean_sqlite(missing)


def test_build_modeling_table_missing_sqlite_has_clear_error(tmp_path: Path) -> None:
    config = tmp_path / "ecotox_clean_sqlite.yaml"
    config.write_text(
        "\n".join(
            [
                "project:",
                f"  root: {tmp_path.as_posix()}",
                "data:",
                "  clean_sqlite: outputs/databases/missing.sqlite",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Clean ECOTOX SQLite not found"):
        build_modeling_table(config)


def test_load_clean_sqlite_augments_primary_medium_from_species(tmp_path: Path) -> None:
    database = tmp_path / "clean.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE ecotox_toxicity_joined (
                result_id INTEGER,
                test_id INTEGER,
                species_number INTEGER,
                endpoint TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO ecotox_toxicity_joined VALUES (1, 10, 99, 'LC50')"
        )
        conn.execute(
            """
            CREATE TABLE species (
                species_number INTEGER,
                primary_medium TEXT
            )
            """
        )
        conn.execute("INSERT INTO species VALUES (99, 'aquatic')")

    loaded = load_clean_sqlite(database)

    assert loaded["joined"].loc[0, "primary_medium"] == "aquatic"


def test_load_clean_sqlite_augments_curated_categories(tmp_path: Path) -> None:
    database = tmp_path / "clean.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE ecotox_toxicity_joined (
                result_id INTEGER,
                cas_number TEXT,
                species_number INTEGER,
                endpoint TEXT
            )
            """
        )
        conn.execute("INSERT INTO ecotox_toxicity_joined VALUES (1, '50-00-0', 99, 'LC50')")
        conn.execute(
            """
            CREATE TABLE chemical_category_curated (
                cas_number TEXT,
                chemical_class_l2 TEXT
            )
            """
        )
        conn.execute("INSERT INTO chemical_category_curated VALUES ('50-00-0', 'metal_metalloid')")
        conn.execute(
            """
            CREATE TABLE species_category_curated (
                species_number INTEGER,
                taxon_group_l2 TEXT,
                is_standard_test_species INTEGER
            )
            """
        )
        conn.execute("INSERT INTO species_category_curated VALUES (99, 'crustacean', 1)")

    loaded = load_clean_sqlite(database)

    assert loaded["joined"].loc[0, "chemical_class_l2"] == "metal_metalloid"
    assert loaded["joined"].loc[0, "taxon_group_l2"] == "crustacean"


def test_build_modeling_table_writes_report_and_transfer_candidates(tmp_path: Path) -> None:
    database = tmp_path / "clean.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE ecotox_toxicity_joined (
                result_id INTEGER,
                test_id INTEGER,
                reference_number INTEGER,
                cas_number TEXT,
                dtxsid TEXT,
                smiles TEXT,
                molecular_weight_g_mol REAL,
                species_number INTEGER,
                latin_name TEXT,
                kingdom TEXT,
                phylum_division TEXT,
                class TEXT,
                tax_order TEXT,
                family TEXT,
                genus TEXT,
                species_ecotox_group TEXT,
                primary_medium TEXT,
                organism_lifestage TEXT,
                endpoint TEXT,
                effect TEXT,
                measurement TEXT,
                trend TEXT,
                endpoint_comments TEXT,
                response_site_comments TEXT,
                conc1_type TEXT,
                conc1_mean REAL,
                conc1_unit TEXT,
                exposure_duration_mean REAL,
                exposure_duration_unit TEXT,
                num_doses_mean REAL
            )
            """
        )
        rows = [
            (
                1,
                10,
                100,
                "50-00-0",
                "DTXSID1",
                "C",
                100.0,
                200,
                "Daphnia magna",
                "Animalia",
                "Arthropoda",
                "Branchiopoda",
                "Cladocera",
                "Daphniidae",
                "Daphnia",
                "Invertebrates",
                "aquatic",
                "adult",
                "LC50",
                "mortality",
                "mortality",
                None,
                None,
                None,
                "active",
                10.0,
                "mg/L",
                2.0,
                "d",
                5.0,
            ),
            (
                2,
                11,
                101,
                "60-00-0",
                "DTXSID2",
                "CC",
                120.0,
                201,
                "Eisenia fetida",
                "Animalia",
                "Annelida",
                "Clitellata",
                "Haplotaxida",
                "Lumbricidae",
                "Eisenia",
                "Worms",
                "soil",
                "adult",
                "EC10",
                "growth",
                "growth",
                None,
                None,
                None,
                "active",
                20.0,
                "mg/kg",
                48.0,
                "h",
                4.0,
            ),
            (
                3,
                12,
                102,
                "70-00-0",
                "DTXSID3",
                None,
                None,
                202,
                "Pimephales promelas",
                "Animalia",
                "Chordata",
                "Actinopterygii",
                "Cyprinodontiformes",
                "Cyprinidae",
                "Pimephales",
                "Fish",
                "aquatic",
                "juvenile",
                "NOEC",
                "growth",
                "growth",
                None,
                None,
                None,
                "active",
                5.0,
                "ug/L",
                None,
                None,
                None,
            ),
        ]
        conn.executemany(
            "INSERT INTO ecotox_toxicity_joined VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    config = tmp_path / "ecotox_clean_sqlite.yaml"
    output_table = tmp_path / "modeling_toxicity_long.parquet"
    output_csv = tmp_path / "modeling_toxicity_long.csv"
    report_json = tmp_path / "modeling_table_build_report.json"
    config.write_text(
        "\n".join(
            [
                "project:",
                f"  root: {tmp_path.as_posix()}",
                "data:",
                f"  clean_sqlite: {database.as_posix()}",
                "  joined_view: ecotox_toxicity_joined",
                f"  output_table: {output_table.as_posix()}",
                f"  output_csv: {output_csv.as_posix()}",
                f"  report_json: {report_json.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    table = build_modeling_table(config)
    report = json.loads(report_json.read_text(encoding="utf-8"))

    assert len(table) == 3
    assert table.loc[table["result_id"] == 1, "is_main_water_task"].item() is True
    assert table.loc[table["result_id"] == 2, "is_transfer_candidate"].item() is True
    assert table.loc[table["result_id"] == 2, "is_transfer_model_ready"].item() is False
    assert table.loc[table["result_id"] == 1, "duration_h"].item() == 48.0
    assert math.isclose(
        table.loc[table["result_id"] == 1, "target_ptox"].item(), 4.0
    )
    assert output_csv.exists()
    assert report["total_rows"] == 3
    assert report["main_water_task_rows"] == 1
    assert report["transfer_candidate_rows"] == 1
    assert report["transfer_model_ready_rows"] == 0
    assert report["missing_smiles_rows"] == 1
    assert report["missing_mw_rows"] == 1
    assert "missing_or_invalid_target" in report["not_modelable_reason_counts"]
