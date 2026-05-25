from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from curate_ecotox_categories import curate_database  # noqa: E402


def test_curate_database_splits_species_flags_from_taxon_groups(tmp_path: Path) -> None:
    database = tmp_path / "clean.sqlite"
    report = tmp_path / "report.json"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE chemicals (
                cas_number TEXT,
                chemical_name TEXT,
                ecotox_group TEXT,
                dtxsid TEXT,
                smiles TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE species (
                species_number INTEGER,
                common_name TEXT,
                latin_name TEXT,
                kingdom TEXT,
                phylum_division TEXT,
                class TEXT,
                tax_order TEXT,
                family TEXT,
                genus TEXT,
                species TEXT,
                ecotox_group TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ecotox_toxicity_joined (
                result_id INTEGER,
                cas_number TEXT,
                species_number INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO chemicals VALUES ('50-00-0','Cadmium chloride','Metals','DTX1','[Cd+2].[Cl-].[Cl-]')"
        )
        conn.execute(
            "INSERT INTO species VALUES (1,'water flea','Daphnia magna','Animalia','Arthropoda','Branchiopoda','Diplostraca','Daphniidae','Daphnia','magna','Crustaceans;Standard Test Species')"
        )
        conn.execute("INSERT INTO ecotox_toxicity_joined VALUES (10,'50-00-0',1)")

    result = curate_database(database, report)

    with sqlite3.connect(database) as conn:
        chem = pd.read_sql_query("SELECT * FROM chemical_category_curated", conn)
        species = pd.read_sql_query("SELECT * FROM species_category_curated", conn)
        joined = pd.read_sql_query("SELECT * FROM ecotox_toxicity_joined_curated", conn)

    assert result["chemical_rows"] == 1
    assert chem.loc[0, "chemical_class_l2"] == "metal_metalloid"
    assert species.loc[0, "taxon_group_l2"] == "crustacean"
    assert bool(species.loc[0, "is_standard_test_species"]) is True
    assert joined.loc[0, "taxon_group_l2"] == "crustacean"
