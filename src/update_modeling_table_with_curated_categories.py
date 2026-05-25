"""Merge curated ECOTOX categories into the modeling long table."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATABASE = Path("outputs/databases/ecotox_clean.sqlite")
DEFAULT_MODELING_TABLE = Path("outputs/tables/modeling_toxicity_long.parquet")
DEFAULT_MODELING_CSV = Path("outputs/tables/modeling_toxicity_long.csv")
DEFAULT_REPORT_JSON = Path("outputs/reports/modeling_table_curated_category_merge_report.json")


CHEMICAL_COLUMNS = [
    "cas_number",
    "chemical_class_l1",
    "chemical_class_l2",
    "chemical_class_l3",
    "use_source_class",
    "structure_flags",
    "chemical_class_confidence",
    "chemical_class_source",
    "chemical_class_evidence",
]
SPECIES_COLUMNS = [
    "species_number",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "is_standard_test_species",
    "is_us_invasive_species",
    "is_us_threatened_endangered",
    "taxon_group_confidence",
    "taxon_group_source",
    "taxon_group_evidence",
]


def merge_curated_categories(
    *,
    database_path: Path,
    modeling_table_path: Path,
    modeling_csv_path: Path | None,
    report_json_path: Path,
) -> dict[str, Any]:
    if not database_path.exists():
        raise FileNotFoundError(f"Clean ECOTOX SQLite not found: {database_path}")
    if not modeling_table_path.exists():
        raise FileNotFoundError(f"Modeling table not found: {modeling_table_path}")

    modeling = pd.read_parquet(modeling_table_path)
    with sqlite3.connect(database_path) as conn:
        chemicals = pd.read_sql_query(
            "SELECT " + ", ".join(CHEMICAL_COLUMNS) + " FROM chemical_category_curated",
            conn,
        )
        species = pd.read_sql_query(
            "SELECT " + ", ".join(SPECIES_COLUMNS) + " FROM species_category_curated",
            conn,
        )

    output = modeling.drop(
        columns=[column for column in [*CHEMICAL_COLUMNS[1:], *SPECIES_COLUMNS[1:]] if column in modeling.columns]
    )
    chemicals["cas_number"] = chemicals["cas_number"].astype("string")
    species["species_number"] = species["species_number"].astype("string")
    output["chemical_id"] = output["chemical_id"].astype("string")
    output["species_number"] = output["species_number"].astype("string")
    output = output.merge(chemicals, left_on="chemical_id", right_on="cas_number", how="left")
    if "cas_number" in output.columns:
        output = output.drop(columns=["cas_number"])
    output = output.merge(species, on="species_number", how="left")

    modeling_table_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(modeling_table_path, index=False)
    if modeling_csv_path is not None:
        modeling_csv_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(modeling_csv_path, index=False, encoding="utf-8-sig")

    report = {
        "database": str(database_path),
        "modeling_table": str(modeling_table_path),
        "modeling_csv": str(modeling_csv_path) if modeling_csv_path else None,
        "rows": int(len(output)),
        "chemical_class_l2_counts": _counts(output, "chemical_class_l2"),
        "taxon_group_l2_counts": _counts(output, "taxon_group_l2"),
        "missing_chemical_class_l2_rows": int(output["chemical_class_l2"].isna().sum()),
        "missing_taxon_group_l2_rows": int(output["taxon_group_l2"].isna().sum()),
    }
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].fillna("missing").value_counts().items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge curated categories into modeling table.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--modeling-table", type=Path, default=DEFAULT_MODELING_TABLE)
    parser.add_argument("--modeling-csv", type=Path, default=DEFAULT_MODELING_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge_curated_categories(
        database_path=args.database,
        modeling_table_path=args.modeling_table,
        modeling_csv_path=None if args.no_csv else args.modeling_csv,
        report_json_path=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
