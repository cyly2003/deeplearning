"""Chemical feature builders for QSAR modeling.

The module keeps RDKit imports lazy so environments without RDKit can still
load the package and emit explicit missing/error masks.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PHYSICOCHEMICAL_FIELDS = (
    "molecular_weight_g_mol",
    "logkow",
    "logd",
    "water_solubility",
    "henry_law_constant",
    "koc",
)

RDKIT_DESCRIPTOR_FIELDS = (
    "molecular_weight_rdkit_g_mol",
    "rdkit_descriptor_mol_wt",
    "rdkit_descriptor_exact_mol_wt",
    "rdkit_descriptor_mol_logp",
    "rdkit_descriptor_tpsa",
    "rdkit_descriptor_h_bond_donors",
    "rdkit_descriptor_h_bond_acceptors",
    "rdkit_descriptor_rotatable_bonds",
    "rdkit_descriptor_ring_count",
    "rdkit_descriptor_heavy_atom_count",
    "rdkit_descriptor_fraction_csp3",
    "rdkit_descriptor_formal_charge",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "smiles_column": "smiles",
    "identifier_columns": ["chemical_id", "casrn", "cas_number", "dtxsid"],
    "deduplicate": True,
    "deduplicate_columns": ["chemical_id", "casrn", "cas_number", "dtxsid", "smiles"],
    "rdkit": {
        "normalize_smiles": True,
    },
    "fingerprints": {
        "enabled": True,
        "type": "morgan",
        "radius": 2,
        "n_bits": 2048,
        "prefix": "morgan_fp_",
        "use_chirality": False,
    },
    "physchem": {
        "prefer_external": True,
        "molecular_weight_priority": ["rdkit", "external"],
        "external_columns": {
            "molecular_weight_g_mol": [
                "molecular_weight_g_mol",
                "molecular_weight",
                "mw",
                "average_mass",
                "AVERAGE_MASS",
            ],
            "logkow": ["logkow", "log_kow", "logKow", "logp", "logP"],
            "logd": ["logd", "log_d", "logD"],
            "water_solubility": [
                "water_solubility",
                "water_solubility_mg_l",
                "solubility",
                "aqueous_solubility",
            ],
            "henry_law_constant": [
                "henry_law_constant",
                "henry_constant",
                "henry",
            ],
            "koc": [
                "koc",
                "KOC",
                "organic_carbon_partition_coefficient",
            ],
        },
    },
    "output": {
        "write": False,
        "features_table": "outputs/features/chemical_features.parquet",
    },
}


@lru_cache(maxsize=1)
def _load_rdkit() -> dict[str, Any] | None:
    """Return RDKit modules, or None when RDKit is unavailable."""

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
    except Exception:
        return None
    return {
        "Chem": Chem,
        "DataStructs": DataStructs,
        "AllChem": AllChem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "rdMolDescriptors": rdMolDescriptors,
    }


def normalize_smiles(smiles: str | None) -> str | None:
    """Return canonical SMILES when RDKit is available, otherwise stripped input.

    Invalid SMILES return None when RDKit can parse/validate structures. If RDKit
    is unavailable, the function cannot validate chemistry and returns the
    cleaned non-empty string so downstream code can preserve lineage.
    """

    cleaned = _clean_text(smiles)
    if cleaned is None:
        return None

    rdkit = _load_rdkit()
    if rdkit is None:
        return cleaned

    try:
        mol = rdkit["Chem"].MolFromSmiles(cleaned)
        if mol is None:
            return None
        return rdkit["Chem"].MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def compute_rdkit_descriptors(smiles: str | None) -> dict[str, object]:
    """Compute a compact RDKit descriptor set with explicit status flags."""

    cleaned = _clean_text(smiles)
    rdkit = _load_rdkit()
    if cleaned is None:
        return _missing_descriptor_record("missing_smiles", rdkit_available=rdkit is not None)
    if rdkit is None:
        return _missing_descriptor_record("rdkit_unavailable", rdkit_available=False)

    try:
        mol = rdkit["Chem"].MolFromSmiles(cleaned)
    except Exception:
        return _missing_descriptor_record("smiles_parse_error", rdkit_available=True)
    if mol is None:
        return _missing_descriptor_record("invalid_smiles", rdkit_available=True)

    try:
        normalized = rdkit["Chem"].MolToSmiles(mol, canonical=True)
        mol_wt = float(rdkit["Descriptors"].MolWt(mol))
        exact_mol_wt = float(rdkit["Descriptors"].ExactMolWt(mol))
        mol_logp = float(rdkit["Crippen"].MolLogP(mol))
        tpsa = float(rdkit["rdMolDescriptors"].CalcTPSA(mol))
        hbd = float(rdkit["rdMolDescriptors"].CalcNumHBD(mol))
        hba = float(rdkit["rdMolDescriptors"].CalcNumHBA(mol))
        rotatable = float(rdkit["rdMolDescriptors"].CalcNumRotatableBonds(mol))
        rings = float(rdkit["rdMolDescriptors"].CalcNumRings(mol))
        heavy_atoms = float(mol.GetNumHeavyAtoms())
        fraction_csp3 = float(rdkit["rdMolDescriptors"].CalcFractionCSP3(mol))
        formal_charge = float(rdkit["Chem"].GetFormalCharge(mol))
    except Exception:
        record = _missing_descriptor_record("descriptor_error", rdkit_available=True)
        record["descriptor_error_flag"] = True
        record["descriptor_missing_flag"] = True
        return record

    return {
        "rdkit_available": True,
        "normalized_smiles": normalized,
        "structure_lookup_status": "ok",
        "smiles_missing_flag": False,
        "smiles_parse_error_flag": False,
        "descriptor_missing_flag": False,
        "descriptor_error_flag": False,
        "molecular_weight_rdkit_g_mol": mol_wt,
        "rdkit_descriptor_mol_wt": mol_wt,
        "rdkit_descriptor_exact_mol_wt": exact_mol_wt,
        "rdkit_descriptor_mol_logp": mol_logp,
        "rdkit_descriptor_tpsa": tpsa,
        "rdkit_descriptor_h_bond_donors": hbd,
        "rdkit_descriptor_h_bond_acceptors": hba,
        "rdkit_descriptor_rotatable_bonds": rotatable,
        "rdkit_descriptor_ring_count": rings,
        "rdkit_descriptor_heavy_atom_count": heavy_atoms,
        "rdkit_descriptor_fraction_csp3": fraction_csp3,
        "rdkit_descriptor_formal_charge": formal_charge,
        "logkow": mol_logp,
    }


def compute_fingerprints(
    smiles: str | None,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
) -> np.ndarray:
    """Return a Morgan fingerprint bit vector.

    Failures return an all-zero vector with the requested length. The richer
    missing/error flags are added by build_chemical_features.
    """

    fingerprint, _status = _compute_fingerprint_with_status(
        smiles=smiles,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )
    return fingerprint


def build_chemical_features(
    chemical_table: pd.DataFrame,
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Build one chemical feature row per configured chemical identifier."""

    if not isinstance(chemical_table, pd.DataFrame):
        raise TypeError("chemical_table must be a pandas DataFrame.")

    cfg = _load_config(config)
    table = chemical_table.copy()
    smiles_column = str(cfg["smiles_column"])
    if smiles_column not in table.columns:
        table[smiles_column] = None
    table = _deduplicate_chemical_table(table, cfg)

    records: list[dict[str, object]] = []
    fingerprint_cfg = cfg["fingerprints"]
    n_bits = int(fingerprint_cfg["n_bits"])
    radius = int(fingerprint_cfg["radius"])
    use_chirality = bool(fingerprint_cfg.get("use_chirality", False))
    prefix = str(fingerprint_cfg.get("prefix", "morgan_fp_"))
    include_fingerprints = bool(fingerprint_cfg.get("enabled", True))

    for row_index, row in table.iterrows():
        smiles = row.get(smiles_column)
        descriptors = compute_rdkit_descriptors(smiles)

        feature_row: dict[str, object] = {}
        for column in cfg["identifier_columns"]:
            if column in table.columns:
                feature_row[column] = row.get(column)
        feature_row["chemical_id"] = _chemical_id_from_row(row, cfg, row_index)
        feature_row["smiles"] = _clean_text(smiles)
        feature_row.update(descriptors)

        _merge_physicochemical_fields(feature_row, row, cfg)

        if include_fingerprints:
            fingerprint, fingerprint_status = _compute_fingerprint_with_status(
                smiles=smiles,
                radius=radius,
                n_bits=n_bits,
                use_chirality=use_chirality,
            )
            feature_row.update(fingerprint_status)
            width = max(4, len(str(n_bits - 1)))
            for bit_index, bit_value in enumerate(fingerprint):
                feature_row[f"{prefix}{bit_index:0{width}d}"] = int(bit_value)
        else:
            feature_row.update(
                {
                    "fingerprint_status": "disabled",
                    "fingerprint_missing_flag": False,
                    "fingerprint_error_flag": False,
                }
            )

        feature_row["chemical_missing_fields"] = _missing_field_mask(feature_row)
        records.append(feature_row)

    features = pd.DataFrame.from_records(records)
    _write_output_if_requested(features, cfg)
    return features


def _missing_descriptor_record(status: str, rdkit_available: bool) -> dict[str, object]:
    parse_error_statuses = {"invalid_smiles", "smiles_parse_error"}
    record: dict[str, object] = {
        "rdkit_available": rdkit_available,
        "normalized_smiles": None,
        "structure_lookup_status": status,
        "smiles_missing_flag": status == "missing_smiles",
        "smiles_parse_error_flag": status in parse_error_statuses,
        "descriptor_missing_flag": True,
        "descriptor_error_flag": status == "descriptor_error",
        "logkow": np.nan,
    }
    for field in RDKIT_DESCRIPTOR_FIELDS:
        record[field] = np.nan
    return record


def _compute_fingerprint_with_status(
    smiles: str | None,
    radius: int,
    n_bits: int,
    use_chirality: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")

    empty = np.zeros(n_bits, dtype=np.int8)
    cleaned = _clean_text(smiles)
    rdkit = _load_rdkit()
    if cleaned is None:
        return empty, _fingerprint_status("missing_smiles")
    if rdkit is None:
        return empty, _fingerprint_status("rdkit_unavailable")

    try:
        mol = rdkit["Chem"].MolFromSmiles(cleaned)
    except Exception:
        return empty, _fingerprint_status("smiles_parse_error", error=True)
    if mol is None:
        return empty, _fingerprint_status("invalid_smiles")

    try:
        bit_vector = rdkit["AllChem"].GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
            useChirality=use_chirality,
        )
        fingerprint = np.zeros(n_bits, dtype=np.int8)
        on_bits = list(bit_vector.GetOnBits())
        if on_bits:
            fingerprint[on_bits] = 1
        return fingerprint, _fingerprint_status("ok", missing=False)
    except Exception:
        return empty, _fingerprint_status("fingerprint_error", error=True)


def _fingerprint_status(
    status: str,
    *,
    missing: bool = True,
    error: bool = False,
) -> dict[str, object]:
    return {
        "fingerprint_status": status,
        "fingerprint_missing_flag": missing,
        "fingerprint_error_flag": error,
    }


def _merge_physicochemical_fields(
    feature_row: dict[str, object],
    source_row: pd.Series,
    cfg: Mapping[str, Any],
) -> None:
    physchem_cfg = cfg["physchem"]
    external_columns = physchem_cfg["external_columns"]
    prefer_external = bool(physchem_cfg.get("prefer_external", True))

    rdkit_mw = _to_float(feature_row.get("molecular_weight_rdkit_g_mol"))
    external_mw, external_mw_source = _first_numeric_value(
        source_row,
        external_columns["molecular_weight_g_mol"],
    )
    mw_priority = tuple(physchem_cfg.get("molecular_weight_priority", ["rdkit", "external"]))
    mw_value = np.nan
    mw_source = "missing"
    for source_name in mw_priority:
        if source_name == "rdkit" and _is_finite_number(rdkit_mw):
            mw_value = rdkit_mw
            mw_source = "rdkit"
            break
        if source_name == "external" and _is_finite_number(external_mw):
            mw_value = external_mw
            mw_source = str(external_mw_source)
            break

    feature_row["molecular_weight_g_mol"] = mw_value
    feature_row["molecular_weight_source"] = mw_source

    _merge_physchem_value(
        feature_row,
        source_row,
        field="logkow",
        candidate_columns=external_columns["logkow"],
        fallback_value=_to_float(feature_row.get("rdkit_descriptor_mol_logp")),
        fallback_source="rdkit_mol_logp",
        prefer_external=prefer_external,
    )
    for field in ("logd", "water_solubility", "henry_law_constant", "koc"):
        _merge_physchem_value(
            feature_row,
            source_row,
            field=field,
            candidate_columns=external_columns[field],
            fallback_value=np.nan,
            fallback_source="missing",
            prefer_external=prefer_external,
        )

    for field in PHYSICOCHEMICAL_FIELDS:
        feature_row[f"{field}_missing_flag"] = not _is_finite_number(feature_row.get(field))


def _merge_physchem_value(
    feature_row: dict[str, object],
    source_row: pd.Series,
    *,
    field: str,
    candidate_columns: Sequence[str],
    fallback_value: float,
    fallback_source: str,
    prefer_external: bool,
) -> None:
    external_value, external_source = _first_numeric_value(source_row, candidate_columns)
    fallback_is_valid = _is_finite_number(fallback_value)
    external_is_valid = _is_finite_number(external_value)

    if prefer_external and external_is_valid:
        feature_row[field] = external_value
        feature_row[f"{field}_source"] = str(external_source)
        return
    if fallback_is_valid:
        feature_row[field] = float(fallback_value)
        feature_row[f"{field}_source"] = fallback_source
        return
    if external_is_valid:
        feature_row[field] = external_value
        feature_row[f"{field}_source"] = str(external_source)
        return

    feature_row[field] = np.nan
    feature_row[f"{field}_source"] = "missing"


def _first_numeric_value(
    row: pd.Series,
    candidate_columns: Sequence[str],
) -> tuple[float, str | None]:
    for column in candidate_columns:
        if column not in row.index:
            continue
        value = _to_float(row.get(column))
        if _is_finite_number(value):
            return value, column
    return np.nan, None


def _missing_field_mask(feature_row: Mapping[str, object]) -> str:
    missing_fields = [
        field
        for field in PHYSICOCHEMICAL_FIELDS
        if bool(feature_row.get(f"{field}_missing_flag", True))
    ]
    if bool(feature_row.get("descriptor_missing_flag", False)):
        missing_fields.append("rdkit_descriptors")
    if bool(feature_row.get("fingerprint_missing_flag", False)):
        missing_fields.append("morgan_fingerprint")
    return ";".join(missing_fields)


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is None:
        return cfg
    if isinstance(config, (str, Path)):
        loaded = _read_yaml(Path(config))
    elif isinstance(config, Mapping):
        loaded = dict(config)
    else:
        raise TypeError("config must be a mapping, path, string, or None.")

    if "chemical_features" in loaded:
        loaded = dict(loaded["chemical_features"])
    return _deep_merge(cfg, loaded)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load YAML feature configs.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return dict(data)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _deduplicate_chemical_table(table: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if not bool(cfg.get("deduplicate", True)):
        return table.reset_index(drop=True)

    candidate_columns = [
        column
        for column in cfg.get("deduplicate_columns", [])
        if column in table.columns
    ]
    if not candidate_columns:
        return table.reset_index(drop=True)

    keys = [
        _first_nonempty(row, candidate_columns) or f"row_{idx}"
        for idx, row in table.iterrows()
    ]
    deduplicated = table.assign(_chemical_feature_key=keys)
    deduplicated = deduplicated.drop_duplicates("_chemical_feature_key", keep="first")
    return deduplicated.drop(columns="_chemical_feature_key").reset_index(drop=True)


def _chemical_id_from_row(row: pd.Series, cfg: Mapping[str, Any], row_index: Any) -> str:
    columns = [column for column in cfg.get("identifier_columns", []) if column in row.index]
    value = _first_nonempty(row, columns)
    if value is not None:
        return value
    smiles = _clean_text(row.get(str(cfg["smiles_column"])))
    if smiles is not None:
        return smiles
    return f"row_{row_index}"


def _first_nonempty(row: pd.Series, columns: Sequence[str]) -> str | None:
    for column in columns:
        value = _clean_text(row.get(column))
        if value is not None:
            return value
    return None


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    return text


def _to_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned == "":
            return np.nan
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _write_output_if_requested(features: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    output_cfg = cfg.get("output", {})
    if not bool(output_cfg.get("write", False)):
        return

    output_path = Path(str(output_cfg["features_table"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        features.to_csv(output_path, index=False)
    else:
        features.to_parquet(output_path, index=False)
