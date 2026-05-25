"""Species context feature builders for multi-species QSAR modeling."""

from __future__ import annotations

from copy import deepcopy
from hashlib import blake2b
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd


PRIMARY_MEDIA = ("aquatic", "sediment", "soil", "terrestrial", "unknown")
LIFESTAGE_CATEGORIES = (
    "adult",
    "juvenile",
    "larva",
    "egg_embryo",
    "seedling",
    "seed",
    "mixed",
    "other",
    "unknown",
)
TAXONOMY_LEVELS = ("kingdom", "phylum", "class", "order", "family", "genus")

MISSING_TOKENS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "nr",
    "not reported",
    "not available",
    "unknown",
    "unspecified",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "identifier_columns": ["species_id", "species_number"],
    "species_number_column": "species_number",
    "scientific_name_columns": ["scientific_name", "latin_name"],
    "missing_category": "unknown",
    "include_taxonomy": True,
    "include_eco_group": True,
    "include_primary_medium": True,
    "include_lifestage": True,
    "include_potential_routes": False,
    "deduplicate": False,
    "deduplicate_columns": ["species_id", "species_number", "scientific_name", "latin_name"],
    "taxonomy": {
        "hash_buckets": 4096,
        "levels": {
            "kingdom": ["taxonomy_kingdom", "kingdom"],
            "phylum": ["taxonomy_phylum", "phylum_division", "phylum"],
            "class": ["taxonomy_class", "class", "class_name"],
            "order": ["taxonomy_order", "tax_order", "order"],
            "family": ["taxonomy_family", "family"],
            "genus": ["taxonomy_genus", "genus"],
        },
    },
    "eco_group_columns": ["eco_group", "species_ecotox_group", "ecotox_group"],
    "eco_group_hash_buckets": 512,
    "primary_medium_field": "primary_medium",
    "primary_medium": {
        "categories": list(PRIMARY_MEDIA),
        "main_water_values": ["aquatic"],
    },
    "lifestage_field": "organism_lifestage",
    "lifestage": {
        "categories": list(LIFESTAGE_CATEGORIES),
    },
    "output": {
        "write": False,
        "features_table": "outputs/features/species_features.parquet",
    },
}


def normalize_species_name(name: object) -> str:
    """Return a stable, lowercase scientific-name key or ``unknown``."""

    cleaned = _clean_text(name)
    if cleaned is None:
        return "unknown"
    normalized = cleaned.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized or "unknown"


def encode_primary_medium(
    value: object,
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, object]:
    """Encode the lifecycle-dominant species medium with masks and task flags."""

    cfg = _load_config(config)
    cleaned = _clean_text(value, treat_unknown_as_missing=False)
    source_missing = cleaned is None
    medium = _canonical_primary_medium(cleaned)
    categories = tuple(str(item) for item in cfg["primary_medium"]["categories"])
    if medium not in categories:
        medium = str(cfg.get("missing_category", "unknown"))

    main_water_values = {
        _normalize_category(item) or "unknown"
        for item in cfg["primary_medium"].get("main_water_values", ["aquatic"])
    }
    is_main_water_species = medium in main_water_values
    unknown_flag = medium == "unknown"
    exclusion_reason = ""
    if not is_main_water_species:
        exclusion_reason = (
            "primary_medium_unknown" if unknown_flag else "primary_medium_not_aquatic"
        )

    encoded: dict[str, object] = {
        "primary_medium": medium,
        "habitat_medium": medium,
        "primary_medium_code": categories.index(medium) if medium in categories else 0,
        "primary_medium_missing_flag": source_missing,
        "primary_medium_unknown_flag": unknown_flag,
        "primary_medium_valid_flag": not unknown_flag,
        "is_main_water_species": is_main_water_species,
        "exclude_from_main_water_task": not is_main_water_species,
        "primary_medium_exclude_from_main_water_task": not is_main_water_species,
        "main_water_task_exclusion_reason": exclusion_reason,
    }
    for category in categories:
        encoded[f"primary_medium_{category}"] = int(medium == category)
    return encoded


def encode_lifestage(
    value: object,
    config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, object]:
    """Encode organism lifestage with explicit unknown/missing masks."""

    cfg = _load_config(config)
    cleaned = _clean_text(value, treat_unknown_as_missing=False)
    source_missing = cleaned is None
    lifestage = _canonical_lifestage(cleaned)
    categories = tuple(str(item) for item in cfg["lifestage"]["categories"])
    if lifestage not in categories:
        lifestage = "unknown"

    encoded: dict[str, object] = {
        "organism_lifestage": cleaned or "unknown",
        "organism_lifestage_normalized": lifestage,
        "lifestage": lifestage,
        "lifestage_code": categories.index(lifestage) if lifestage in categories else 0,
        "lifestage_missing_flag": source_missing,
        "lifestage_unknown_flag": lifestage == "unknown",
    }
    for category in categories:
        encoded[f"lifestage_{category}"] = int(lifestage == category)
    return encoded


def build_taxonomy_features(
    species_table: pd.DataFrame,
    taxonomy_ref: pd.DataFrame | Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Build normalized taxonomy features and masks from species metadata."""

    if not isinstance(species_table, pd.DataFrame):
        raise TypeError("species_table must be a pandas DataFrame.")
    if isinstance(taxonomy_ref, Mapping) and config is None:
        config = taxonomy_ref

    cfg = _load_config(config)
    table = _merge_taxonomy_reference(species_table, taxonomy_ref, cfg)
    level_columns = cfg["taxonomy"]["levels"]
    hash_buckets = int(cfg["taxonomy"].get("hash_buckets", 4096))

    records: list[dict[str, object]] = []
    for _, row in table.iterrows():
        feature_row: dict[str, object] = {}
        missing_fields: list[str] = []
        path_parts: list[str] = []

        for level in TAXONOMY_LEVELS:
            candidates = level_columns.get(level, [])
            raw_value = _first_nonempty(row, candidates)
            normalized = _normalize_taxon(raw_value)
            output_column = f"taxonomy_{level}"
            is_missing = normalized is None
            if is_missing:
                normalized = "unknown"
                missing_fields.append(output_column)
            else:
                path_parts.append(f"{level}:{normalized}")

            feature_row[output_column] = normalized
            feature_row[f"{output_column}_missing_flag"] = is_missing
            feature_row[f"{output_column}_code"] = (
                0 if is_missing else _stable_code(f"{level}:{normalized}", hash_buckets)
            )

        feature_row["taxonomy_path"] = "|".join(path_parts) if path_parts else "unknown"
        feature_row["taxonomy_depth"] = len(path_parts)
        feature_row["taxonomy_missing_fields"] = ";".join(missing_fields)
        feature_row["taxonomy_missing_flag"] = bool(missing_fields)
        feature_row["taxonomy_all_missing_flag"] = len(path_parts) == 0
        feature_row["taxonomy_complete_flag"] = not missing_fields
        feature_row["taxonomy_source"] = "input_table" if path_parts else "missing"
        records.append(feature_row)

    return pd.DataFrame.from_records(records, index=species_table.index)


def build_species_features(
    species_table: pd.DataFrame,
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Build species context features from ECOTOX species/test metadata."""

    if not isinstance(species_table, pd.DataFrame):
        raise TypeError("species_table must be a pandas DataFrame.")

    cfg = _load_config(config)
    table = species_table.copy()
    table = _deduplicate_if_requested(table, cfg)
    taxonomy_features = (
        build_taxonomy_features(table, config=cfg)
        if bool(cfg.get("include_taxonomy", True))
        else pd.DataFrame(index=table.index)
    )

    records: list[dict[str, object]] = []
    for row_index, row in table.iterrows():
        feature_row: dict[str, object] = {}
        species_number = _first_nonempty(row, [str(cfg.get("species_number_column", ""))])
        species_id = _species_id_from_row(row, cfg, row_index)
        scientific_name = _first_nonempty(row, cfg.get("scientific_name_columns", []))

        for column in cfg.get("identifier_columns", []):
            if column in table.columns:
                feature_row[column] = row.get(column)
        feature_row["species_id"] = species_id
        feature_row["species_number"] = species_number
        feature_row["scientific_name"] = scientific_name or "unknown"
        feature_row["normalized_species_name"] = normalize_species_name(scientific_name)

        if bool(cfg.get("include_taxonomy", True)):
            feature_row.update(taxonomy_features.loc[row_index].to_dict())

        if bool(cfg.get("include_eco_group", True)):
            feature_row.update(_encode_eco_group(row, cfg))

        if bool(cfg.get("include_primary_medium", True)):
            medium_value = _first_nonempty(
                row,
                [str(cfg.get("primary_medium_field", ""))],
                treat_unknown_as_missing=False,
            )
            feature_row.update(encode_primary_medium(medium_value, cfg))

        if bool(cfg.get("include_lifestage", True)):
            lifestage_value = _first_nonempty(
                row,
                [str(cfg.get("lifestage_field", ""))],
                treat_unknown_as_missing=False,
            )
            feature_row.update(encode_lifestage(lifestage_value, cfg))

        feature_row["species_missing_fields"] = _species_missing_field_mask(feature_row, cfg)
        feature_row["species_feature_status"] = _species_feature_status(feature_row, cfg)
        records.append(feature_row)

    features = pd.DataFrame.from_records(records)
    _write_output_if_requested(features, cfg)
    return features


def _canonical_primary_medium(value: str | None) -> str:
    key = _normalize_category(value, treat_unknown_as_missing=False)
    if key is None:
        return "unknown"
    aliases = {
        "water": "aquatic",
        "freshwater": "aquatic",
        "marine": "aquatic",
        "brackish": "aquatic",
        "amphibious": "aquatic",
        "aquatic": "aquatic",
        "sediment": "sediment",
        "benthic": "sediment",
        "benthos": "sediment",
        "soil": "soil",
        "terrestrial_soil": "soil",
        "land": "terrestrial",
        "non_soil": "terrestrial",
        "nonsoil": "terrestrial",
        "terrestrial": "terrestrial",
        "unknown": "unknown",
    }
    return aliases.get(key, key if key in PRIMARY_MEDIA else "unknown")


def _canonical_lifestage(value: str | None) -> str:
    key = _clean_text(value, treat_unknown_as_missing=False)
    if key is None:
        return "unknown"
    text = key.lower()
    compact = _normalize_category(text) or ""
    if compact in {"unknown", "not_reported", "unspecified"}:
        return "unknown"

    hits: set[str] = set()
    if any(token in text for token in ("adult", "mature")):
        hits.add("adult")
    if any(
        token in text
        for token in (
            "juvenile",
            "immature",
            "subadult",
            "young",
            "neonate",
            "nymph",
            "instar",
            "fingerling",
            "fry",
        )
    ):
        hits.add("juvenile")
    if any(token in text for token in ("larva", "larvae", "larval", "naupli")):
        hits.add("larva")
    if any(token in text for token in ("egg", "embryo", "embryonic", "zygote")):
        hits.add("egg_embryo")
    if "seedling" in text or "plantlet" in text:
        hits.add("seedling")
    elif "seed" in text or "spore" in text:
        hits.add("seed")
    if any(token in text for token in ("mixed", "various", "multiple", "several", "all stages")):
        hits.add("mixed")
    if len(hits) > 1:
        return "mixed"
    if hits:
        return next(iter(hits))
    return "other"


def _encode_eco_group(row: pd.Series, cfg: Mapping[str, Any]) -> dict[str, object]:
    source_value = _first_nonempty(row, cfg.get("eco_group_columns", []))
    normalized = _normalize_category(source_value)
    missing = normalized is None
    eco_group = normalized or "unknown"
    hash_buckets = int(cfg.get("eco_group_hash_buckets", 512))
    return {
        "species_ecotox_group": source_value or "unknown",
        "eco_group": eco_group,
        "eco_group_code": 0 if missing else _stable_code(f"eco:{eco_group}", hash_buckets),
        "eco_group_missing_flag": missing,
        "eco_group_unknown_flag": eco_group == "unknown",
    }


def _merge_taxonomy_reference(
    species_table: pd.DataFrame,
    taxonomy_ref: pd.DataFrame | Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if not isinstance(taxonomy_ref, pd.DataFrame) or taxonomy_ref.empty:
        return species_table.copy()

    table = species_table.copy()
    ref = taxonomy_ref.copy()
    table_key = _normalized_name_series(table, cfg)
    ref_key = _normalized_name_series(ref, cfg)
    table = table.assign(_taxonomy_merge_key=table_key)
    ref = ref.assign(_taxonomy_merge_key=ref_key)
    ref = ref.drop_duplicates("_taxonomy_merge_key")
    merged = table.merge(ref, on="_taxonomy_merge_key", how="left", suffixes=("", "_ref"))

    for candidates in cfg["taxonomy"]["levels"].values():
        for column in candidates:
            ref_column = f"{column}_ref"
            if column in merged.columns and ref_column in merged.columns:
                merged[column] = merged[column].where(
                    merged[column].map(_clean_text).notna(), merged[ref_column]
                )
            elif column not in merged.columns and ref_column in merged.columns:
                merged[column] = merged[ref_column]
    drop_columns = [column for column in merged.columns if column.endswith("_ref")]
    drop_columns.append("_taxonomy_merge_key")
    return merged.drop(columns=drop_columns, errors="ignore")


def _normalized_name_series(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    columns = [column for column in cfg.get("scientific_name_columns", []) if column in frame]
    if not columns:
        return pd.Series(["unknown"] * len(frame), index=frame.index)
    return frame[columns[0]].map(normalize_species_name)


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

    if "species_features" in loaded:
        loaded = dict(loaded["species_features"])
    elif "species" in loaded:
        loaded = dict(loaded["species"])
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


def _deduplicate_if_requested(table: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if not bool(cfg.get("deduplicate", False)):
        return table.reset_index(drop=True)
    candidate_columns = [
        column for column in cfg.get("deduplicate_columns", []) if column in table.columns
    ]
    if not candidate_columns:
        return table.reset_index(drop=True)

    keys = [
        _first_nonempty(row, candidate_columns) or f"row_{idx}"
        for idx, row in table.iterrows()
    ]
    deduplicated = table.assign(_species_feature_key=keys)
    deduplicated = deduplicated.drop_duplicates("_species_feature_key", keep="first")
    return deduplicated.drop(columns="_species_feature_key").reset_index(drop=True)


def _species_id_from_row(row: pd.Series, cfg: Mapping[str, Any], row_index: object) -> str:
    value = _first_nonempty(row, cfg.get("identifier_columns", []))
    if value is not None:
        return str(value)
    scientific_name = _first_nonempty(row, cfg.get("scientific_name_columns", []))
    if scientific_name is not None:
        return normalize_species_name(scientific_name)
    return f"row_{row_index}"


def _first_nonempty(
    row: pd.Series,
    columns: Sequence[str],
    *,
    treat_unknown_as_missing: bool = True,
) -> str | None:
    for column in columns:
        if not column or column not in row.index:
            continue
        value = _clean_text(
            row.get(column),
            treat_unknown_as_missing=treat_unknown_as_missing,
        )
        if value is not None:
            return value
    return None


def _clean_text(value: object, *, treat_unknown_as_missing: bool = True) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    missing_tokens = MISSING_TOKENS
    if not treat_unknown_as_missing:
        missing_tokens = MISSING_TOKENS - {"unknown", "unspecified"}
    if not text or text.lower() in missing_tokens:
        return None
    return re.sub(r"\s+", " ", text)


def _normalize_category(
    value: object,
    *,
    treat_unknown_as_missing: bool = True,
) -> str | None:
    cleaned = _clean_text(value, treat_unknown_as_missing=treat_unknown_as_missing)
    if cleaned is None:
        return None
    normalized = cleaned.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or None


def _normalize_taxon(value: object) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    return re.sub(r"\s+", " ", cleaned).strip()


def _stable_code(value: str, buckets: int) -> int:
    if buckets <= 0:
        raise ValueError("hash bucket count must be a positive integer.")
    digest = blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % buckets + 1


def _species_missing_field_mask(
    feature_row: Mapping[str, object],
    cfg: Mapping[str, Any],
) -> str:
    missing_fields: list[str] = []
    if bool(cfg.get("include_taxonomy", True)):
        taxonomy_missing = str(feature_row.get("taxonomy_missing_fields", ""))
        if bool(feature_row.get("taxonomy_all_missing_flag", False)):
            missing_fields.append("taxonomy")
        elif taxonomy_missing:
            missing_fields.extend(taxonomy_missing.split(";"))
    if bool(cfg.get("include_eco_group", True)) and bool(
        feature_row.get("eco_group_missing_flag", False)
    ):
        missing_fields.append("eco_group")
    if bool(cfg.get("include_primary_medium", True)) and bool(
        feature_row.get("primary_medium_unknown_flag", False)
    ):
        missing_fields.append("primary_medium")
    if bool(cfg.get("include_lifestage", True)) and bool(
        feature_row.get("lifestage_unknown_flag", False)
    ):
        missing_fields.append("organism_lifestage")
    return ";".join(dict.fromkeys(missing_fields))


def _species_feature_status(feature_row: Mapping[str, object], cfg: Mapping[str, Any]) -> str:
    missing = str(feature_row.get("species_missing_fields", ""))
    if not missing:
        return "ok"

    expected_fields = []
    if bool(cfg.get("include_taxonomy", True)):
        expected_fields.append("taxonomy")
    if bool(cfg.get("include_eco_group", True)):
        expected_fields.append("eco_group")
    if bool(cfg.get("include_primary_medium", True)):
        expected_fields.append("primary_medium")
    if bool(cfg.get("include_lifestage", True)):
        expected_fields.append("organism_lifestage")

    missing_parts = set(missing.split(";"))
    if set(expected_fields).issubset(missing_parts):
        return "missing"
    return "partial"


def _write_output_if_requested(features: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    output_cfg = cfg.get("output", {})
    if not bool(output_cfg.get("write", False)):
        return

    output_path = Path(str(output_cfg["features_table"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        features.to_parquet(output_path, index=False)
