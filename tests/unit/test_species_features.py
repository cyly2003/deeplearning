from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qsar_dl.features.species import (  # noqa: E402
    build_species_features,
    build_taxonomy_features,
    encode_lifestage,
    encode_primary_medium,
    normalize_species_name,
)


def test_normalize_species_name_returns_stable_lowercase_key() -> None:
    assert normalize_species_name("  Danio   rerio ") == "danio rerio"
    assert normalize_species_name(None) == "unknown"


def test_encode_primary_medium_covers_all_project_media() -> None:
    media = ["aquatic", "sediment", "soil", "terrestrial", "unknown"]

    encoded = {medium: encode_primary_medium(medium) for medium in media}

    for medium in media:
        assert encoded[medium]["primary_medium"] == medium
        assert encoded[medium][f"primary_medium_{medium}"] == 1
    assert encoded["aquatic"]["exclude_from_main_water_task"] is False
    assert encoded["sediment"]["exclude_from_main_water_task"] is True
    assert encoded["soil"]["exclude_from_main_water_task"] is True
    assert encoded["terrestrial"]["exclude_from_main_water_task"] is True
    assert encoded["unknown"]["exclude_from_main_water_task"] is True
    assert encoded["unknown"]["main_water_task_exclusion_reason"] == "primary_medium_unknown"


def test_encode_primary_medium_missing_uses_unknown_mask() -> None:
    result = encode_primary_medium(None)

    assert result["primary_medium"] == "unknown"
    assert result["primary_medium_unknown"] == 1
    assert result["primary_medium_missing_flag"] is True
    assert result["primary_medium_unknown_flag"] is True
    assert result["exclude_from_main_water_task"] is True


def test_encode_primary_medium_explicit_unknown_is_not_source_missing() -> None:
    result = encode_primary_medium("unknown")

    assert result["primary_medium"] == "unknown"
    assert result["primary_medium_missing_flag"] is False
    assert result["primary_medium_unknown_flag"] is True


def test_encode_lifestage_missing_uses_unknown_mask() -> None:
    result = encode_lifestage(None)

    assert result["organism_lifestage"] == "unknown"
    assert result["lifestage"] == "unknown"
    assert result["lifestage_unknown"] == 1
    assert result["lifestage_missing_flag"] is True
    assert result["lifestage_unknown_flag"] is True


def test_build_taxonomy_features_marks_missing_levels() -> None:
    table = pd.DataFrame(
        {
            "species_number": [1],
            "latin_name": ["Mystery species"],
            "kingdom": [None],
            "phylum_division": [None],
            "class": [None],
            "tax_order": [None],
            "family": [None],
            "genus": [None],
        }
    )

    features = build_taxonomy_features(table)
    row = features.iloc[0]

    assert row["taxonomy_kingdom"] == "unknown"
    assert row["taxonomy_depth"] == 0
    assert bool(row["taxonomy_all_missing_flag"]) is True
    assert bool(row["taxonomy_missing_flag"]) is True
    assert "taxonomy_family" in row["taxonomy_missing_fields"]


def test_build_species_features_encodes_taxonomy_eco_medium_and_lifestage() -> None:
    table = pd.DataFrame(
        {
            "species_number": [101, 102, 103],
            "latin_name": ["Danio rerio", "Eisenia fetida", "Unknown taxon"],
            "kingdom": ["Animalia", "Animalia", None],
            "phylum_division": ["Chordata", "Annelida", None],
            "class": ["Actinopterygii", "Clitellata", None],
            "tax_order": ["Cypriniformes", "Haplotaxida", None],
            "family": ["Danionidae", "Lumbricidae", None],
            "genus": ["Danio", "Eisenia", None],
            "species_ecotox_group": ["Fish", "Worms", None],
            "primary_medium": ["aquatic", "soil", "unknown"],
            "organism_lifestage": ["adult", None, "larva"],
        }
    )

    features = build_species_features(table)

    fish = features.loc[features["species_number"] == "101"].iloc[0]
    worm = features.loc[features["species_number"] == "102"].iloc[0]
    unknown = features.loc[features["species_number"] == "103"].iloc[0]

    assert fish["normalized_species_name"] == "danio rerio"
    assert fish["taxonomy_class"] == "Actinopterygii"
    assert fish["eco_group"] == "fish"
    assert fish["primary_medium_aquatic"] == 1
    assert fish["lifestage_adult"] == 1
    assert fish["species_feature_status"] == "ok"

    assert worm["primary_medium_soil"] == 1
    assert bool(worm["exclude_from_main_water_task"]) is True
    assert bool(worm["lifestage_missing_flag"]) is True
    assert "organism_lifestage" in worm["species_missing_fields"]

    assert unknown["primary_medium_unknown"] == 1
    assert bool(unknown["primary_medium_missing_flag"]) is False
    assert bool(unknown["exclude_from_main_water_task"]) is True
    assert bool(unknown["taxonomy_all_missing_flag"]) is True
    assert "taxonomy" in unknown["species_missing_fields"]


def test_config_file_can_be_loaded() -> None:
    config_path = PROJECT_ROOT / "configs" / "features" / "species_context.yaml"
    input_table = pd.DataFrame(
        {
            "species_number": [1],
            "latin_name": ["Daphnia magna"],
            "class": ["Branchiopoda"],
            "primary_medium": ["aquatic"],
            "organism_lifestage": ["juvenile"],
        }
    )

    features = build_species_features(input_table, config_path)

    assert features.shape[0] == 1
    assert features.loc[0, "primary_medium_aquatic"] == 1
    assert features.loc[0, "lifestage_juvenile"] == 1
