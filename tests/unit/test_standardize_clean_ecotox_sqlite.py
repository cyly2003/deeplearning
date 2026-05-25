from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from standardize_clean_ecotox_sqlite import standardize_clean_database  # noqa: E402


def test_standardize_clean_database_adds_mw_duration_and_concentration_fields(
    tmp_path: Path,
) -> None:
    database = tmp_path / "clean.sqlite"
    report_json = tmp_path / "standardization_report.json"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE chemicals (
                cas_number TEXT PRIMARY KEY,
                chemical_name TEXT,
                ecotox_group TEXT,
                dtxsid TEXT,
                smiles TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO chemicals VALUES ('50-00-0', 'formaldehyde', 'Other', 'DTXSID1', 'C=O')"
        )
        conn.execute(
            """
            CREATE TABLE results (
                result_id INTEGER PRIMARY KEY,
                test_id INTEGER,
                obs_duration_mean REAL,
                obs_duration_unit TEXT,
                conc1_type TEXT,
                conc1_mean REAL,
                conc1_unit TEXT,
                conc1_min REAL,
                conc1_max REAL,
                endpoint TEXT,
                endpoint_comments TEXT,
                trend TEXT,
                effect TEXT,
                measurement TEXT,
                response_site_comments TEXT,
                conc1_mean_op TEXT,
                conc1_min_op TEXT,
                conc1_max_op TEXT,
                obs_duration_mean_op TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO results VALUES (1, 10, 2, 'd', 'active', 1000, 'ug/L', 100, 2000, 'LC50', NULL, NULL, 'mortality', 'mortality', NULL, NULL, NULL, NULL, NULL)"
        )
        conn.execute(
            """
            CREATE TABLE tests (
                test_id INTEGER PRIMARY KEY,
                reference_number INTEGER,
                test_cas TEXT,
                species_number INTEGER,
                exposure_duration_mean REAL,
                exposure_duration_min REAL,
                exposure_duration_max REAL,
                exposure_duration_unit TEXT,
                organism_habitat TEXT,
                organism_lifestage TEXT,
                media_type TEXT,
                num_doses_mean REAL,
                num_doses_min REAL,
                num_doses_max REAL,
                test_purity_mean_op TEXT,
                test_purity_mean REAL,
                test_purity_min_op TEXT,
                test_purity_min REAL,
                test_purity_max_op TEXT,
                test_purity_max REAL,
                exposure_duration_mean_op TEXT,
                exposure_duration_min_op TEXT,
                exposure_duration_max_op TEXT,
                num_doses_mean_op TEXT,
                num_doses_min_op TEXT,
                num_doses_max_op TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO tests VALUES (10, 100, '50-00-0', 200, 1, 1, 3, 'wk', 'Water', 'adult', 'freshwater', 5, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
        )
        conn.execute(
            """
            CREATE TABLE species (
                species_number INTEGER PRIMARY KEY,
                common_name TEXT,
                latin_name TEXT,
                kingdom TEXT,
                phylum_division TEXT,
                class TEXT,
                tax_order TEXT,
                family TEXT,
                genus TEXT,
                species TEXT,
                ecotox_group TEXT,
                ncbi_taxid TEXT,
                primary_medium TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO species VALUES (200, 'water flea', 'Daphnia magna', 'Animalia', 'Arthropoda', 'Branchiopoda', 'Cladocera', 'Daphniidae', 'Daphnia', 'magna', 'Invertebrates', '666', 'aquatic')"
        )
        conn.execute(
            """
            CREATE TABLE "references" (
                reference_number INTEGER PRIMARY KEY,
                reference_type TEXT,
                author TEXT,
                title TEXT,
                source TEXT,
                publication_year INTEGER,
                doi TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO \"references\" VALUES (100, 'journal', 'Author', 'Title', 'Source', 2024, '10.1/test')"
        )

    report = standardize_clean_database(database, report_json)

    with sqlite3.connect(database) as conn:
        chem = conn.execute(
            "SELECT molecular_weight_g_mol, molecular_weight_status FROM chemicals"
        ).fetchone()
        result = conn.execute(
            """
            SELECT obs_duration_mean_h, conc1_mean_standardized,
                   conc1_standard_unit, conc1_unit_family
            FROM results
            """
        ).fetchone()
        test = conn.execute(
            "SELECT exposure_duration_mean_h, exposure_duration_min_h, exposure_duration_max_h FROM tests"
        ).fetchone()
        joined = conn.execute(
            "SELECT primary_medium, molecular_weight_g_mol, conc1_mean_standardized FROM ecotox_toxicity_joined"
        ).fetchone()

    assert chem[0] is not None
    assert chem[1] == "rdkit_smiles"
    assert result == (48.0, 1.0, "mg/L", "water_mg_l")
    assert test == (168.0, 168.0, 504.0)
    assert joined[0] == "aquatic"
    assert joined[1] is not None
    assert joined[2] == 1.0
    assert report["joined_view"]["includes_primary_medium"] is True
    assert json.loads(report_json.read_text(encoding="utf-8"))["results"]["row_count"] == 1
