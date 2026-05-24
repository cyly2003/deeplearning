from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qsar_dl.features.descriptor_groups import (  # noqa: E402
    build_fixed_group_features,
    load_descriptor_group_dictionary,
    validate_descriptor_coverage,
)


def _write_group_yaml(path: Path) -> None:
    path.write_text(
        """
descriptor_source: rdkit
standardization:
  method: robust_zscore
  missing_strategy: train_median_with_mask
groups:
  hydrophobicity_partition:
    description: Hydrophobicity and partition-related descriptors.
    initial_group_weight: 1.0
    bias_init: 0.0
    descriptors:
      MolLogP:
        role: core
        initial_weight: 1.0
      MolMR:
        role: auxiliary
        initial_weight: 0.5
  polarity_hbond:
    description: Polarity and hydrogen-bond descriptors.
    initial_group_weight: 1.0
    bias_init: 0.0
    descriptors:
      TPSA:
        role: core
        initial_weight: 1.0
      NumHDonors:
        role: core
        initial_weight: 1.0
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def group_dict(tmp_path: Path) -> dict:
    config_path = tmp_path / "descriptor_groups.yaml"
    _write_group_yaml(config_path)
    return load_descriptor_group_dictionary(config_path)


def test_load_descriptor_group_dictionary_validates_yaml(group_dict: dict) -> None:
    assert group_dict["descriptor_source"] == "rdkit"
    assert set(group_dict["groups"]) == {
        "hydrophobicity_partition",
        "polarity_hbond",
    }


def test_load_descriptor_group_dictionary_rejects_duplicate_descriptors(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_groups.yaml"
    config_path.write_text(
        """
descriptor_source: rdkit
standardization:
  method: robust_zscore
  missing_strategy: train_median_with_mask
groups:
  group_a:
    description: First group.
    initial_group_weight: 1.0
    bias_init: 0.0
    descriptors:
      MolLogP:
        role: core
        initial_weight: 1.0
  group_b:
    description: Second group.
    initial_group_weight: 1.0
    bias_init: 0.0
    descriptors:
      MolLogP:
        role: core
        initial_weight: 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="appears in both"):
        load_descriptor_group_dictionary(config_path)


def test_validate_descriptor_coverage_reports_grouped_ungrouped_and_missing(
    group_dict: dict,
) -> None:
    coverage = validate_descriptor_coverage(["MolLogP", "TPSA", "ExtraDesc"], group_dict)

    statuses = dict(zip(coverage["descriptor"], coverage["status"], strict=True))
    assert statuses["MolLogP"] == "grouped"
    assert statuses["TPSA"] == "grouped"
    assert statuses["ExtraDesc"] == "ungrouped"
    assert statuses["MolMR"] == "missing"
    assert statuses["NumHDonors"] == "missing"

    extra_row = coverage.loc[coverage["descriptor"] == "ExtraDesc"].iloc[0]
    assert bool(extra_row["present"]) is True
    assert bool(extra_row["grouped"]) is False


def test_build_fixed_group_features_exports_group_values_and_missing_masks(
    group_dict: dict,
) -> None:
    descriptor_df = pd.DataFrame(
        {
            "chemical_id": ["chem_a", "chem_b"],
            "MolLogP": [2.0, None],
            "MolMR": [4.0, 6.0],
            "TPSA": [10.0, None],
        }
    )

    group_features = build_fixed_group_features(descriptor_df, group_dict)

    assert list(group_features["chemical_id"]) == ["chem_a", "chem_b"]
    assert math.isclose(
        group_features.loc[0, "desc_group_hydrophobicity_partition"],
        (2.0 * 1.0 + 4.0 * 0.5) / 1.5,
    )
    assert group_features.loc[1, "desc_group_hydrophobicity_partition"] == 6.0
    assert group_features.loc[0, "desc_group_hydrophobicity_partition_coverage"] == 2
    assert group_features.loc[1, "desc_group_hydrophobicity_partition_coverage"] == 1
    assert group_features.loc[1, "desc_group_hydrophobicity_partition_missing_rate"] == 0.5

    assert group_features.loc[0, "desc_group_polarity_hbond"] == 10.0
    assert group_features.loc[0, "desc_group_polarity_hbond_coverage"] == 1
    assert group_features.loc[0, "desc_group_polarity_hbond_missing_rate"] == 0.5
    assert pd.isna(group_features.loc[1, "desc_group_polarity_hbond"])
    assert group_features.loc[1, "desc_group_polarity_hbond_missing_rate"] == 1.0


def test_descriptor_group_weighting_import_is_safe_for_feature_tests(
    group_dict: dict,
) -> None:
    from qsar_dl.models.descriptor_weighting import DescriptorGroupWeighting

    torch = pytest.importorskip("torch")
    layer = DescriptorGroupWeighting(group_dict, mode="fixed_group_weights")
    x = torch.tensor([[2.0, 4.0, 10.0, 1.0]], dtype=torch.float32)
    output = layer(x)

    assert output["group_embeddings"].shape == (1, 2)
    assert output["global_embedding"].shape == (1, 1)
    assert output["intragroup_weights"].shape == (2, 2)
    assert output["intergroup_weights"].shape == (2,)
    assert not layer.intragroup_logits[0].requires_grad
    assert not layer.intergroup_logits.requires_grad

    masked_x = torch.tensor([[float("nan"), 6.0, float("nan"), 1.0]], dtype=torch.float32)
    mask = torch.tensor([[False, True, False, True]])
    masked_output = layer(masked_x, mask=mask)
    assert torch.isfinite(masked_output["group_embeddings"]).all()
    assert torch.isfinite(masked_output["global_embedding"]).all()
