# 配置系统接口规范

版本：v0.1
日期：2026-05-24

## 1. 定位

本项目采用“拆分配置 + resolved config 导出”。每个模块维护自己的配置文件，实验配置负责引用这些文件。运行时配置加载器合并所有配置，并在输出目录保存完整的 `config_resolved.yaml`。

目标：

- 方便多个子代理并行开发。
- 避免路径和参数散落在代码里。
- 保证每次实验可复现。

## 2. 目录结构

```text
configs/
├── data/
│   └── ecotox_clean_sqlite.yaml
├── features/
│   ├── chemical_rdkit_morgan.yaml
│   ├── species_context.yaml
│   └── descriptor_groups_rdkit.yaml
├── rules/
│   └── mechanistic_rules.yaml
├── experiments/
│   ├── baseline_ml.yaml
│   ├── baseline_deep.yaml
│   ├── main_residual_qsar.yaml
│   └── transfer_soil_sediment.yaml
├── evaluation/
│   └── chemical_category_holdout.yaml
└── runtime/
    ├── local.yaml
    └── gpu_server.yaml
```

## 3. 实验配置格式

实验入口只接收一个文件：

```powershell
python -m qsar_dl.cli train --config configs/experiments/main_residual_qsar.yaml
```

实验配置必须包含：

```yaml
experiment:
  id: main_residual_qsar_v001
  seed: 20260524
  output_dir: outputs/models/main_residual_qsar_v001
includes:
  data: configs/data/ecotox_clean_sqlite.yaml
  chemical_features: configs/features/chemical_rdkit_morgan.yaml
  species_features: configs/features/species_context.yaml
  descriptor_groups: configs/features/descriptor_groups_rdkit.yaml
  rules: configs/rules/mechanistic_rules.yaml
  evaluation: configs/evaluation/chemical_category_holdout.yaml
  runtime: configs/runtime/local.yaml
model:
  name: neutral_residual_qsar
  architecture: rule_aware_residual_mlp
  target_column: target_ptox
```

## 4. 合并规则

配置加载器必须按以下顺序合并：

```text
base defaults
<included configs>
experiment config
CLI overrides
```

后者覆盖前者。所有相对路径均相对项目根目录解析。

禁止：

- 在模块代码中硬编码数据路径。
- 让不同模块各自解析不同项目根目录。
- 运行后不保存 resolved config。

## 5. Python 接口

包路径：

```text
src/qsar_dl/config/
```

必须实现：

```python
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file with UTF-8 encoding."""


def resolve_config(config_path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load includes, merge configs, resolve project-relative paths, validate keys."""


def write_resolved_config(config: dict[str, Any], output_path: Path) -> None:
    """Write the exact runtime config to YAML."""


def get_project_root(start: Path | None = None) -> Path:
    """Return repository root containing .git or pyproject.toml."""
```

## 6. 必需公共键

所有 resolved config 必须含有：

```yaml
project:
  root: C:/Users/Lenovo/Documents/深度学习QSAR
experiment:
  id: ""
  seed: 20260524
  output_dir: ""
data:
  clean_sqlite: outputs/databases/ecotox_clean.sqlite
  modeling_table: outputs/tables/modeling_toxicity_long.parquet
target:
  column: target_ptox
  unit_family: water_mg_l
runtime:
  device: auto
  num_workers: 0
logging:
  level: INFO
  save_resolved_config: true
```

## 7. 路径修改说明

如果更换 clean SQLite：

```yaml
data:
  clean_sqlite: D:/path/to/ecotox_clean.sqlite
```

如果更换实验输出目录：

```yaml
experiment:
  output_dir: outputs/models/<new_experiment_id>
```

如果切换到服务器：

```yaml
includes:
  runtime: configs/runtime/gpu_server.yaml
```

## 8. 验收标准

- 任一实验运行后必须生成 `config_resolved.yaml`。
- `config_resolved.yaml` 中不得包含未展开的 `${...}`。
- 缺少 included config 时必须报出具体文件路径。
- 同一 seed 和同一 config 下，数据划分必须一致。
