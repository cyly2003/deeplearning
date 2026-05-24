from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qsar_dl.config import (  # noqa: E402
    ConfigError,
    load_yaml,
    resolve_config,
    write_resolved_config,
)


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_resolve_config_merges_includes_overrides_and_resolves_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    _write_yaml(
        root / "configs" / "data.yaml",
        """
data:
  clean_sqlite: data/custom.sqlite
  modeling_table: outputs/tables/custom_modeling.parquet
target:
  unit_family: soil_mg_kg
""",
    )
    _write_yaml(
        root / "configs" / "runtime.yaml",
        """
runtime:
  device: cpu
  num_workers: 1
logging:
  level: DEBUG
""",
    )
    _write_yaml(
        root / "configs" / "chemical_features.yaml",
        """
chemical_features:
  fingerprint: morgan
""",
    )
    _write_yaml(
        root / "configs" / "species_features.yaml",
        """
species_features:
  include_taxonomy: true
""",
    )
    _write_yaml(
        root / "configs" / "descriptor_groups.yaml",
        """
descriptor_groups:
  rdkit_basic: true
""",
    )
    _write_yaml(
        root / "configs" / "rules.yaml",
        """
rules:
  enabled: true
""",
    )
    _write_yaml(
        root / "configs" / "evaluation.yaml",
        """
evaluation:
  strategy: chemical_category_holdout
""",
    )
    config_path = root / "configs" / "experiment.yaml"
    _write_yaml(
        config_path,
        """
experiment:
  id: residual_qsar_test
  seed: 42
  output_dir: outputs/models/${experiment.id}
includes:
  data: configs/data.yaml
  chemical_features: configs/chemical_features.yaml
  species_features: configs/species_features.yaml
  descriptor_groups: configs/descriptor_groups.yaml
  rules: configs/rules.yaml
  evaluation: configs/evaluation.yaml
  runtime: configs/runtime.yaml
model:
  name: neutral_residual_qsar
  target_column: target_ptox
""",
    )

    resolved = resolve_config(
        config_path,
        overrides={
            "runtime": {"num_workers": 2},
            "logging.level": "WARNING",
        },
    )

    assert "includes" not in resolved
    assert resolved["project"]["root"] == root.resolve().as_posix()
    assert resolved["experiment"]["seed"] == 42
    assert resolved["runtime"]["device"] == "cpu"
    assert resolved["runtime"]["num_workers"] == 2
    assert resolved["logging"]["level"] == "WARNING"
    assert resolved["chemical_features"]["fingerprint"] == "morgan"
    assert resolved["species_features"]["include_taxonomy"] is True
    assert resolved["descriptor_groups"]["rdkit_basic"] is True
    assert resolved["rules"]["enabled"] is True
    assert resolved["evaluation"]["strategy"] == "chemical_category_holdout"
    assert resolved["target"]["column"] == "target_ptox"
    assert resolved["target"]["unit_family"] == "soil_mg_kg"
    assert resolved["data"]["clean_sqlite"] == (root / "data" / "custom.sqlite").as_posix()
    assert resolved["data"]["modeling_table"] == (
        root / "outputs" / "tables" / "custom_modeling.parquet"
    ).as_posix()
    assert resolved["experiment"]["output_dir"] == (
        root / "outputs" / "models" / "residual_qsar_test"
    ).as_posix()
    assert "${" not in str(resolved)


def test_missing_include_reports_concrete_path(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    missing_path = root / "configs" / "missing.yaml"
    config_path = root / "configs" / "experiment.yaml"
    _write_yaml(
        config_path,
        """
experiment:
  id: missing_include_test
  seed: 1
  output_dir: outputs/models/missing_include_test
includes:
  data: configs/missing.yaml
""",
    )

    with pytest.raises(ConfigError) as exc_info:
        resolve_config(config_path)

    assert missing_path.as_posix() in str(exc_info.value).replace("\\", "/")


def test_write_resolved_config_round_trips_utf8(tmp_path: Path) -> None:
    config = {
        "project": {"root": "C:/Users/Lenovo/Documents/深度学习QSAR"},
        "experiment": {
            "id": "中文配置测试",
            "seed": 20260524,
            "output_dir": "C:/outputs/model",
        },
        "data": {
            "clean_sqlite": "C:/outputs/databases/ecotox_clean.sqlite",
            "modeling_table": "C:/outputs/tables/modeling_toxicity_long.parquet",
        },
        "target": {"column": "target_ptox", "unit_family": "water_mg_l"},
        "runtime": {"device": "auto", "num_workers": 0},
        "logging": {"level": "INFO", "save_resolved_config": True},
    }
    output_path = tmp_path / "config_resolved.yaml"

    write_resolved_config(config, output_path)

    assert load_yaml(output_path)["experiment"]["id"] == "中文配置测试"


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(path)
