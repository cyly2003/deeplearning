"""Data-contract pipeline for the clean ECOTOX SQLite database."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "clean_sqlite": "outputs/databases/ecotox_clean.sqlite",
        "joined_view": "ecotox_toxicity_joined",
        "output_table": "outputs/tables/modeling_toxicity_long.parquet",
        "output_csv": "outputs/tables/modeling_toxicity_long.csv",
        "report_json": "outputs/reports/modeling_table_build_report.json",
    },
    "target": {
        "main_medium_field": "primary_medium",
        "main_medium_values": ["aquatic"],
        "main_unit_family": "water_mg_l",
        "target_column": "target_ptox",
        "endpoints": ["LC", "EC", "LOEC"],
    },
    "derivation": {
        "default_num_doses": 4,
        "use_dose_grid_midpoint": True,
    },
}

REQUIRED_OUTPUT_COLUMNS = [
    "record_id",
    "result_id",
    "test_id",
    "reference_number",
    "chemical_id",
    "casrn",
    "dtxsid",
    "smiles",
    "species_id",
    "species_number",
    "scientific_name",
    "taxonomy_kingdom",
    "taxonomy_phylum",
    "taxonomy_class",
    "taxonomy_order",
    "taxonomy_family",
    "taxonomy_genus",
    "species_ecotox_group",
    "primary_medium",
    "organism_lifestage",
    "endpoint_raw",
    "endpoint_family",
    "effect_level",
    "effect",
    "measurement",
    "trend",
    "endpoint_comments",
    "response_site_comments",
    "conc_value",
    "conc_unit",
    "conc_derivation_method",
    "conc1_type",
    "duration_h",
    "duration_derivation_method",
    "duration_missing_flag",
    "num_doses_used",
    "target_mg_l",
    "target_mol_l",
    "target_ptox",
    "target_unit_family",
    "modeling_split_group",
    "qa_flags",
    "is_main_water_task",
    "is_transfer_candidate",
]


def load_clean_sqlite(database_path: Path) -> Mapping[str, pd.DataFrame]:
    """Load clean ECOTOX tables and the joined toxicity frame from SQLite."""
    database_path = Path(database_path)
    if not database_path.exists():
        raise FileNotFoundError(
            "Clean ECOTOX SQLite not found: "
            f"{database_path}. Build it first or update data.clean_sqlite."
        )

    tables: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(database_path) as conn:
        names = _sqlite_objects(conn)
        if "ecotox_toxicity_joined" in names:
            tables["joined"] = pd.read_sql_query(
                f"SELECT * FROM {_quote_identifier('ecotox_toxicity_joined')}", conn
            )
        else:
            tables["joined"] = _build_joined_from_tables(conn, names)

        for table_name in ("results", "tests", "chemicals", "species", "references"):
            if table_name in names:
                tables[table_name] = pd.read_sql_query(
                    f"SELECT * FROM {_quote_identifier(table_name)}", conn
                )

        tables["joined"] = _augment_joined_frame(conn, tables["joined"], names)

    return tables


def derive_concentration(row: Mapping[str, object]) -> dict[str, object]:
    """Return concentration value, unit, dose count, method, and QA flags."""
    qa_flags: list[str] = []
    conc_unit = _clean_text(row.get("conc1_unit"))
    qa_flags.extend(_operator_qa(row, "conc1_mean_op", "conc1_mean"))

    mean_value = _to_float(row.get("conc1_mean"))
    if mean_value is not None:
        return {
            "conc_value": mean_value,
            "conc_unit": conc_unit,
            "num_doses_used": _derive_num_doses(row, qa_flags),
            "conc_derivation_method": "mean",
            "qa_flags": qa_flags,
        }

    min_value = _to_float(row.get("conc1_min"))
    max_value = _to_float(row.get("conc1_max"))
    qa_flags.extend(_operator_qa(row, "conc1_min_op", "conc1_min"))
    qa_flags.extend(_operator_qa(row, "conc1_max_op", "conc1_max"))

    if min_value is not None and max_value is not None:
        if min_value > max_value:
            qa_flags.append("conc_min_gt_max_swapped")
            min_value, max_value = max_value, min_value
        num_doses = _derive_num_doses(row, qa_flags)
        return {
            "conc_value": _dose_grid_midpoint(min_value, max_value, int(num_doses)),
            "conc_unit": conc_unit,
            "num_doses_used": float(num_doses),
            "conc_derivation_method": "direct_range_midpoint",
            "qa_flags": qa_flags,
        }

    return {
        "conc_value": None,
        "conc_unit": conc_unit,
        "num_doses_used": _derive_num_doses(row, qa_flags),
        "conc_derivation_method": "missing",
        "qa_flags": qa_flags,
    }


def derive_duration(row: Mapping[str, object]) -> dict[str, object]:
    """Return exposure duration in hours, derivation method, missing flag, QA flags."""
    qa_flags: list[str] = []
    candidates = [
        (
            "exposure_mean",
            row.get("exposure_duration_mean"),
            row.get("exposure_duration_unit"),
            "exposure_duration_mean_op",
        ),
        (
            "observation_mean",
            row.get("obs_duration_mean"),
            row.get("obs_duration_unit"),
            "obs_duration_mean_op",
        ),
    ]
    for method, value, unit, op_key in candidates:
        numeric = _to_float(value)
        if numeric is None:
            continue
        qa_flags.extend(_operator_qa(row, op_key, op_key.removesuffix("_op")))
        duration_h = standardize_duration(numeric, unit)
        if duration_h is not None:
            return {
                "duration_h": duration_h,
                "duration_derivation_method": method,
                "duration_missing_flag": False,
                "qa_flags": qa_flags,
            }
        qa_flags.append(f"{method}_unit_unrecognized")

    range_candidates = [
        (
            "exposure_range_grid_mid",
            row.get("exposure_duration_min"),
            row.get("exposure_duration_max"),
            row.get("exposure_duration_unit"),
            "exposure_duration_min_op",
            "exposure_duration_max_op",
        ),
        (
            "observation_range_grid_mid",
            row.get("obs_duration_min"),
            row.get("obs_duration_max"),
            row.get("obs_duration_unit"),
            "obs_duration_min_op",
            "obs_duration_max_op",
        ),
    ]
    for method, min_raw, max_raw, unit, min_op, max_op in range_candidates:
        min_value = _to_float(min_raw)
        max_value = _to_float(max_raw)
        if min_value is None or max_value is None:
            continue
        qa_flags.extend(_operator_qa(row, min_op, min_op.removesuffix("_op")))
        qa_flags.extend(_operator_qa(row, max_op, max_op.removesuffix("_op")))
        if min_value > max_value:
            qa_flags.append(f"{method}_min_gt_max_swapped")
            min_value, max_value = max_value, min_value
        num_doses = _derive_num_doses(row, qa_flags)
        midpoint = _dose_grid_midpoint(min_value, max_value, int(num_doses))
        duration_h = standardize_duration(midpoint, unit)
        if duration_h is not None:
            return {
                "duration_h": duration_h,
                "duration_derivation_method": method,
                "duration_missing_flag": False,
                "qa_flags": qa_flags,
            }
        qa_flags.append(f"{method}_unit_unrecognized")

    return {
        "duration_h": None,
        "duration_derivation_method": "missing_manual_review",
        "duration_missing_flag": True,
        "qa_flags": qa_flags,
    }


def parse_endpoint(endpoint: str, effect: str | None = None) -> dict[str, object]:
    """Return endpoint family and numeric effect level for LC/EC/LOEC endpoints."""
    del effect
    endpoint_text = _clean_text(endpoint).upper()
    if not endpoint_text:
        return {"endpoint_family": None, "effect_level": None}

    match = re.match(r"^\s*(LOEC|LC|EC)\s*[-_/]?\s*(\d+(?:\.\d+)?)?", endpoint_text)
    if not match:
        return {"endpoint_family": None, "effect_level": None}

    return {
        "endpoint_family": match.group(1),
        "effect_level": _to_float(match.group(2)),
    }


def standardize_duration(value: object, unit: object) -> float | None:
    """Convert a duration value to hours."""
    numeric = _to_float(value)
    unit_key = _normalize_unit(unit)
    if numeric is None or not unit_key:
        return None

    unit_factors = {
        "s": 1 / 3600,
        "sec": 1 / 3600,
        "second": 1 / 3600,
        "seconds": 1 / 3600,
        "min": 1 / 60,
        "minute": 1 / 60,
        "minutes": 1 / 60,
        "h": 1,
        "hr": 1,
        "hrs": 1,
        "hour": 1,
        "hours": 1,
        "d": 24,
        "day": 24,
        "days": 24,
        "wk": 24 * 7,
        "wks": 24 * 7,
        "week": 24 * 7,
        "weeks": 24 * 7,
        "mo": 24 * 30,
        "month": 24 * 30,
        "months": 24 * 30,
        "yr": 24 * 365,
        "year": 24 * 365,
        "years": 24 * 365,
    }
    factor = unit_factors.get(unit_key)
    if factor is None:
        return None
    return numeric * factor


def standardize_target_units(row: Mapping[str, object]) -> dict[str, object]:
    """Return target values in mg/L, mol/L, pTox, and a unit-family label."""
    value = _to_float(row.get("conc_value"))
    unit = _normalize_unit(row.get("conc_unit"))
    molecular_weight = _get_molecular_weight(row)
    medium_hint = " ".join(
        _clean_text(row.get(key)).lower()
        for key in ("primary_medium", "media_type", "organism_habitat")
        if _clean_text(row.get(key))
    )

    target_mg_l: float | None = None
    target_mol_l: float | None = None
    target_unit_family = "other"

    if value is None or value <= 0 or not unit:
        return {
            "target_mg_l": None,
            "target_mol_l": None,
            "target_ptox": None,
            "target_unit_family": target_unit_family,
        }

    water_factor = _water_mass_per_volume_factor_to_mg_l(unit)
    mol_factor = _molar_factor_to_mol_l(unit)
    if water_factor is not None:
        target_unit_family = "water_mg_l"
        target_mg_l = value * water_factor
        if molecular_weight is not None and molecular_weight > 0:
            target_mol_l = target_mg_l / (molecular_weight * 1000.0)
    elif mol_factor is not None:
        target_unit_family = "water_mg_l"
        target_mol_l = value * mol_factor
        if molecular_weight is not None and molecular_weight > 0:
            target_mg_l = target_mol_l * molecular_weight * 1000.0
    elif _is_oral_daily_unit(unit):
        target_unit_family = "oral_mg_kg_d"
    elif _is_mass_per_mass_unit(unit):
        target_unit_family = (
            "sediment_mg_kg" if "sediment" in medium_hint else "soil_mg_kg"
        )

    target_ptox = None
    if target_mol_l is not None and target_mol_l > 0:
        target_ptox = -math.log10(target_mol_l)

    return {
        "target_mg_l": target_mg_l,
        "target_mol_l": target_mol_l,
        "target_ptox": target_ptox,
        "target_unit_family": target_unit_family,
    }


def build_modeling_table(config_path: Path) -> pd.DataFrame:
    """Build the canonical modeling long table and write configured outputs."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    project_root = _project_root_from_config(config_path, config)
    data_config = config["data"]
    target_config = config["target"]

    database_path = _resolve_path(project_root, data_config["clean_sqlite"])
    if not database_path.exists():
        raise FileNotFoundError(
            "Clean ECOTOX SQLite not found: "
            f"{database_path}. Build outputs/databases/ecotox_clean.sqlite "
            f"or update data.clean_sqlite in {config_path}."
        )

    joined = _load_joined_frame(database_path, data_config.get("joined_view"))
    records = joined.to_dict(orient="records")
    output_records: list[dict[str, Any]] = []
    target_endpoints = {str(value).upper() for value in target_config["endpoints"]}
    main_medium_values = {
        str(value).strip().lower() for value in target_config["main_medium_values"]
    }

    for source_row in records:
        conc = derive_concentration(source_row)
        duration = derive_duration(source_row)
        endpoint = parse_endpoint(
            _clean_text(source_row.get("endpoint")),
            _clean_text(source_row.get("effect")),
        )
        target_row = {**source_row, **conc}
        target = standardize_target_units(target_row)
        qa_flags = _merge_flags(
            conc.get("qa_flags"),
            duration.get("qa_flags"),
            _source_quality_flags(source_row, endpoint, target),
        )

        medium = _clean_text(source_row.get(target_config["main_medium_field"]))
        endpoint_family = endpoint["endpoint_family"]
        target_value = target.get(target_config["target_column"])
        is_main = (
            medium.lower() in main_medium_values
            and target["target_unit_family"] == target_config["main_unit_family"]
            and endpoint_family in target_endpoints
            and _to_float(target_value) is not None
        )
        is_transfer = _is_transfer_candidate(
            medium=medium,
            target_unit_family=str(target["target_unit_family"]),
            is_main=is_main,
        )
        not_modelable_reasons = _not_modelable_reasons(
            medium=medium,
            main_medium_values=main_medium_values,
            endpoint_family=endpoint_family,
            target_endpoints=target_endpoints,
            target=target,
            source_row=source_row,
            is_main=is_main,
        )

        output_records.append(
            {
                "record_id": _record_id(source_row),
                "result_id": source_row.get("result_id"),
                "test_id": source_row.get("test_id"),
                "reference_number": source_row.get("reference_number"),
                "chemical_id": _first_present(
                    source_row, ["cas_number", "casrn", "test_cas", "dtxsid"]
                ),
                "casrn": _first_present(source_row, ["cas_number", "casrn", "test_cas"]),
                "dtxsid": source_row.get("dtxsid"),
                "smiles": source_row.get("smiles"),
                "molecular_weight_g_mol": _get_molecular_weight(source_row),
                "species_id": source_row.get("species_number"),
                "species_number": source_row.get("species_number"),
                "scientific_name": _first_present(
                    source_row, ["latin_name", "scientific_name"]
                ),
                "taxonomy_kingdom": source_row.get("kingdom"),
                "taxonomy_phylum": source_row.get("phylum_division"),
                "taxonomy_class": source_row.get("class"),
                "taxonomy_order": source_row.get("tax_order"),
                "taxonomy_family": source_row.get("family"),
                "taxonomy_genus": source_row.get("genus"),
                "species_ecotox_group": source_row.get("species_ecotox_group"),
                "primary_medium": source_row.get("primary_medium"),
                "organism_lifestage": source_row.get("organism_lifestage"),
                "endpoint_raw": source_row.get("endpoint"),
                "endpoint_family": endpoint_family,
                "effect_level": endpoint["effect_level"],
                "effect": source_row.get("effect"),
                "measurement": source_row.get("measurement"),
                "trend": source_row.get("trend"),
                "endpoint_comments": source_row.get("endpoint_comments"),
                "response_site_comments": source_row.get("response_site_comments"),
                "conc_value": conc["conc_value"],
                "conc_unit": conc["conc_unit"],
                "conc_derivation_method": conc["conc_derivation_method"],
                "conc1_type": source_row.get("conc1_type"),
                "duration_h": duration["duration_h"],
                "duration_derivation_method": duration["duration_derivation_method"],
                "duration_missing_flag": duration["duration_missing_flag"],
                "num_doses_used": conc["num_doses_used"],
                "target_mg_l": target["target_mg_l"],
                "target_mol_l": target["target_mol_l"],
                "target_ptox": target["target_ptox"],
                "target_unit_family": target["target_unit_family"],
                "modeling_split_group": None,
                "qa_flags": ";".join(qa_flags),
                "not_modelable_reasons": ";".join(not_modelable_reasons),
                "is_main_water_task": is_main,
                "is_transfer_candidate": is_transfer,
            }
        )

    modeling_table = pd.DataFrame(output_records)
    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in modeling_table.columns:
            modeling_table[column] = None
    modeling_table = modeling_table[
        REQUIRED_OUTPUT_COLUMNS
        + [column for column in modeling_table.columns if column not in REQUIRED_OUTPUT_COLUMNS]
    ]

    output_table = _resolve_path(project_root, data_config["output_table"])
    output_csv = _resolve_path(project_root, data_config.get("output_csv", ""))
    report_json = _resolve_path(project_root, data_config["report_json"])
    parquet_status = _write_outputs(modeling_table, output_table, output_csv)
    _write_report(modeling_table, report_json, database_path, parquet_status)
    return modeling_table


def _load_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Data config must be a mapping: {config_path}")
        _deep_update(config, loaded)
    return config


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _project_root_from_config(config_path: Path, config: Mapping[str, Any]) -> Path:
    project_root = config.get("project", {}).get("root") if isinstance(config.get("project"), Mapping) else None
    if project_root:
        return Path(project_root).expanduser().resolve()
    return _find_project_root(config_path.parent)


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "docs" / "DATA_CONTRACT.md").exists():
            return candidate
    return current


def _resolve_path(project_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _sqlite_objects(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _load_joined_frame(database_path: Path, joined_view: object) -> pd.DataFrame:
    with sqlite3.connect(database_path) as conn:
        names = _sqlite_objects(conn)
        view_name = _clean_text(joined_view) or "ecotox_toxicity_joined"
        if view_name in names:
            frame = pd.read_sql_query(f"SELECT * FROM {_quote_identifier(view_name)}", conn)
        else:
            frame = _build_joined_from_tables(conn, names)
        return _augment_joined_frame(conn, frame, names)


def _build_joined_from_tables(conn: sqlite3.Connection, names: set[str]) -> pd.DataFrame:
    required = {"results", "tests", "chemicals", "species", "references"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "Clean SQLite must contain ecotox_toxicity_joined or normalized tables. "
            f"Missing tables: {', '.join(missing)}"
        )

    results = pd.read_sql_query("SELECT * FROM results", conn)
    tests = pd.read_sql_query("SELECT * FROM tests", conn)
    chemicals = pd.read_sql_query("SELECT * FROM chemicals", conn)
    species = pd.read_sql_query("SELECT * FROM species", conn)
    references = pd.read_sql_query('SELECT * FROM "references"', conn)

    frame = results.merge(tests, on="test_id", how="left", suffixes=("", "_test"))
    if "test_cas" in frame.columns and "cas_number" in chemicals.columns:
        frame = frame.merge(
            chemicals, left_on="test_cas", right_on="cas_number", how="left"
        )
    if "species_number" in frame.columns and "species_number" in species.columns:
        frame = frame.merge(species, on="species_number", how="left", suffixes=("", "_species"))
    if "reference_number" in frame.columns and "reference_number" in references.columns:
        frame = frame.merge(
            references, on="reference_number", how="left", suffixes=("", "_reference")
        )
    return frame


def _augment_joined_frame(
    conn: sqlite3.Connection, frame: pd.DataFrame, names: set[str]
) -> pd.DataFrame:
    augmented = frame.copy()
    if "species" in names and "species_number" in augmented.columns:
        species = pd.read_sql_query("SELECT * FROM species", conn)
        if "species_number" in species.columns:
            extra_species_columns = [
                column
                for column in species.columns
                if column not in augmented.columns or column == "species_number"
            ]
            if len(extra_species_columns) > 1:
                augmented = augmented.merge(
                    species[extra_species_columns],
                    on="species_number",
                    how="left",
                    suffixes=("", "_species_extra"),
                )

    if "chemicals" in names:
        chemicals = pd.read_sql_query("SELECT * FROM chemicals", conn)
        if "cas_number" in chemicals.columns:
            left_key = "cas_number" if "cas_number" in augmented.columns else "test_cas"
            if left_key in augmented.columns:
                extra_chemical_columns = [
                    column
                    for column in chemicals.columns
                    if column not in augmented.columns or column == "cas_number"
                ]
                if len(extra_chemical_columns) > 1:
                    augmented = augmented.merge(
                        chemicals[extra_chemical_columns],
                        left_on=left_key,
                        right_on="cas_number",
                        how="left",
                        suffixes=("", "_chemical_extra"),
                    )
                    if "cas_number_chemical_extra" in augmented.columns:
                        augmented = augmented.drop(columns=["cas_number_chemical_extra"])
    return augmented


def _write_outputs(
    modeling_table: pd.DataFrame, output_table: Path, output_csv: Path
) -> dict[str, Any]:
    output_table.parent.mkdir(parents=True, exist_ok=True)
    if str(output_csv):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        modeling_table.to_csv(output_csv, index=False, encoding="utf-8-sig")

    status: dict[str, Any] = {
        "parquet_path": str(output_table),
        "parquet_written": False,
        "parquet_error": None,
        "csv_path": str(output_csv) if str(output_csv) else None,
    }
    if output_table.suffix.lower() == ".parquet":
        try:
            modeling_table.to_parquet(output_table, index=False)
            status["parquet_written"] = True
        except Exception as exc:  # pragma: no cover - depends on optional engines.
            status["parquet_error"] = f"{type(exc).__name__}: {exc}"
            if not str(output_csv):
                fallback_csv = output_table.with_suffix(".csv")
                modeling_table.to_csv(fallback_csv, index=False, encoding="utf-8-sig")
                status["csv_path"] = str(fallback_csv)
    else:
        modeling_table.to_csv(output_table, index=False, encoding="utf-8-sig")
        status["csv_path"] = str(output_table)
    return status


def _write_report(
    modeling_table: pd.DataFrame,
    report_json: Path,
    database_path: Path,
    parquet_status: Mapping[str, Any],
) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    reason_counts: Counter[str] = Counter()
    for value in modeling_table.get("not_modelable_reasons", pd.Series(dtype=str)).fillna(""):
        for reason in str(value).split(";"):
            if reason:
                reason_counts[reason] += 1

    report = {
        "source_sqlite": str(database_path),
        "total_rows": int(len(modeling_table)),
        "main_water_task_rows": int(modeling_table["is_main_water_task"].sum()),
        "transfer_candidate_rows": int(modeling_table["is_transfer_candidate"].sum()),
        "missing_smiles_rows": int(modeling_table["smiles"].isna().sum()),
        "missing_mw_rows": int(modeling_table["molecular_weight_g_mol"].isna().sum()),
        "conc_derivation_method_counts": _value_counts(modeling_table, "conc_derivation_method"),
        "duration_derivation_method_counts": _value_counts(
            modeling_table, "duration_derivation_method"
        ),
        "not_modelable_reason_counts": dict(sorted(reason_counts.items())),
        "outputs": dict(parquet_status),
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("missing").value_counts().sort_index().items()
    }


def _source_quality_flags(
    source_row: Mapping[str, object],
    endpoint: Mapping[str, object],
    target: Mapping[str, object],
) -> list[str]:
    flags: list[str] = []
    if endpoint.get("endpoint_family") is None:
        flags.append("unsupported_endpoint")
    if not _clean_text(source_row.get("smiles")):
        flags.append("missing_smiles")
    if _get_molecular_weight(source_row) is None:
        flags.append("missing_mw")
    if target.get("target_unit_family") == "other":
        flags.append("target_unit_family_other")
    if _to_float(target.get("target_ptox")) is None and target.get("target_unit_family") == "water_mg_l":
        flags.append("target_ptox_missing")
    return flags


def _not_modelable_reasons(
    *,
    medium: str,
    main_medium_values: set[str],
    endpoint_family: object,
    target_endpoints: set[str],
    target: Mapping[str, object],
    source_row: Mapping[str, object],
    is_main: bool,
) -> list[str]:
    if is_main:
        return []

    reasons: list[str] = []
    if medium.lower() not in main_medium_values:
        reasons.append("non_main_medium")
    if endpoint_family not in target_endpoints:
        reasons.append("unsupported_endpoint")
    if target.get("target_unit_family") != "water_mg_l":
        reasons.append("non_main_unit_family")
    if _to_float(target.get("target_ptox")) is None:
        reasons.append("missing_or_invalid_target")
    if not _clean_text(source_row.get("smiles")):
        reasons.append("missing_smiles")
    if _get_molecular_weight(source_row) is None:
        reasons.append("missing_mw")
    return reasons


def _is_transfer_candidate(
    *, medium: str, target_unit_family: str, is_main: bool
) -> bool:
    if is_main or target_unit_family == "oral_mg_kg_d":
        return False
    medium_key = medium.strip().lower()
    return target_unit_family in {"soil_mg_kg", "sediment_mg_kg"} or medium_key in {
        "soil",
        "sediment",
        "terrestrial",
    }


def _record_id(row: Mapping[str, object]) -> str:
    result_id = row.get("result_id")
    if not _is_missing(result_id):
        return str(result_id)
    payload = "|".join(
        _clean_text(row.get(key))
        for key in ("test_id", "endpoint", "conc1_mean", "conc1_min", "conc1_max")
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _first_present(row: Mapping[str, object], keys: list[str]) -> object:
    for key in keys:
        value = row.get(key)
        if not _is_missing(value) and _clean_text(value):
            return value
    return None


def _derive_num_doses(row: Mapping[str, object], qa_flags: list[str]) -> float:
    mean_value = _to_float(row.get("num_doses_mean"))
    if mean_value is not None:
        return float(_valid_num_doses(mean_value, qa_flags))

    min_value = _to_float(row.get("num_doses_min"))
    max_value = _to_float(row.get("num_doses_max"))
    if min_value is not None and max_value is not None:
        return float(_valid_num_doses((min_value + max_value) / 2.0, qa_flags))

    qa_flags.append("num_doses_defaulted")
    return 4.0


def _valid_num_doses(value: float, qa_flags: list[str]) -> int:
    if value < 2:
        qa_flags.append("num_doses_invalid_defaulted")
        return 4
    rounded = int(round(value))
    if not math.isclose(value, rounded):
        qa_flags.append("num_doses_rounded")
    if rounded < 2:
        qa_flags.append("num_doses_invalid_defaulted")
        return 4
    return rounded


def _dose_grid_midpoint(min_value: float, max_value: float, num_doses: int) -> float:
    if num_doses < 2:
        num_doses = 4
    if num_doses % 2 == 1:
        middle_index = num_doses // 2
        step = (max_value - min_value) / (num_doses - 1)
        return min_value + step * middle_index
    left_index = num_doses // 2 - 1
    right_index = num_doses // 2
    step = (max_value - min_value) / (num_doses - 1)
    return ((min_value + step * left_index) + (min_value + step * right_index)) / 2.0


def _operator_qa(
    row: Mapping[str, object], operator_key: str, value_key: str
) -> list[str]:
    operator = _clean_text(row.get(operator_key))
    if not operator or ("<" not in operator and ">" not in operator):
        return []
    direction = "lt" if "<" in operator else "gt"
    return [f"{value_key}_censored_{direction}"]


def _merge_flags(*flag_groups: object) -> list[str]:
    merged: list[str] = []
    for group in flag_groups:
        if group is None:
            continue
        if isinstance(group, str):
            candidates = [value for value in group.split(";") if value]
        else:
            candidates = list(group)
        for flag in candidates:
            flag_text = _clean_text(flag)
            if flag_text and flag_text not in merged:
                merged.append(flag_text)
    return merged


def _get_molecular_weight(row: Mapping[str, object]) -> float | None:
    for key in (
        "molecular_weight_rdkit_g_mol",
        "molecular_weight_g_mol",
        "molecular_weight",
        "mol_weight",
        "mw",
    ):
        value = _to_float(row.get(key))
        if value is not None and value > 0:
            return value

    smiles = _clean_text(row.get("smiles"))
    if not smiles:
        return None
    try:  # Optional: RDKit may be supplied by the chemistry feature environment.
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return float(Descriptors.MolWt(molecule))
    except Exception:
        return None


def _water_mass_per_volume_factor_to_mg_l(unit: str) -> float | None:
    compact = unit.replace(" ", "")
    factors = {
        "mg/l": 1.0,
        "mg/liter": 1.0,
        "mg/litre": 1.0,
        "mgl-1": 1.0,
        "mg/l-1": 1.0,
        "ug/l": 0.001,
        "ug/liter": 0.001,
        "ug/litre": 0.001,
        "ugl-1": 0.001,
        "ug/l-1": 0.001,
        "ng/l": 0.000001,
        "ngl-1": 0.000001,
        "g/l": 1000.0,
        "gl-1": 1000.0,
    }
    return factors.get(compact)


def _molar_factor_to_mol_l(unit: str) -> float | None:
    compact = unit.replace(" ", "")
    factors = {
        "mol/l": 1.0,
        "m/l": 1.0,
        "moll-1": 1.0,
        "mmol/l": 0.001,
        "umol/l": 0.000001,
        "umoll-1": 0.000001,
        "nmol/l": 0.000000001,
    }
    return factors.get(compact)


def _is_mass_per_mass_unit(unit: str) -> bool:
    compact = unit.replace(" ", "")
    return compact in {
        "mg/kg",
        "mgkg-1",
        "ug/kg",
        "ugkg-1",
        "ng/kg",
        "ngkg-1",
        "g/kg",
        "gkg-1",
    }


def _is_oral_daily_unit(unit: str) -> bool:
    compact = unit.replace(" ", "")
    return compact in {
        "mg/kg/d",
        "mg/kg/day",
        "mgkg-1d-1",
        "mgkg-1day-1",
        "ug/kg/d",
        "ug/kg/day",
        "ugkg-1d-1",
        "ugkg-1day-1",
    }


def _normalize_unit(unit: object) -> str:
    text = _clean_text(unit).lower()
    if not text:
        return ""
    replacements = {
        "micro": "u",
        "µ": "u",
        "μ": "u",
        " per ": "/",
        " litre": " liter",
        "litre": "liter",
        "−": "-",
        "–": "-",
        "·": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _to_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        numeric = float(match.group(0))
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except TypeError:
        return False
    if isinstance(result, (list, tuple)):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
