# 数据契约规范

版本：v0.1
日期：2026-05-24

## 1. 定位

本文件定义从 `outputs/databases/ecotox_clean.sqlite` 到建模长表、特征表和质量审计表的接口。第一阶段后续管线直接读取该 clean SQLite，不回查 ECOTOX 原始全量表。clean SQLite 的构建说明见 `docs/ECOTOX_CLEAN_DATABASE.md`。

权威输入：

```text
outputs/databases/ecotox_clean.sqlite
```

核心关系：

```text
results.test_id -> tests.test_id
tests.test_cas -> chemicals.cas_number
tests.species_number -> species.species_number
tests.reference_number -> references.reference_number
```

推荐读取视图：

```text
ecotox_toxicity_joined
```

如需要更严格的数据 lineage，可直接读取 normalized tables 后在代码中 join。

## 2. 第一阶段数据范围

主论文任务聚焦水相毒性：

- 使用 `species.primary_medium` 判定水相记录。
- 无法判定为水相的记录不进入第一版主训练，进入 `excluded_or_transfer_candidates`。
- 土壤和沉积物记录保留为迁移学习候选集。
- 经口 `mg/kg/d` 记录暂不进入第一篇主任务，只标记保存。

终点范围：

- `LC`
- 同族 `EC`
- `LOEC`

终点数字作为独立字段 `effect_level`，例如 `LC50 -> endpoint_family=LC, effect_level=50`。

## 3. 建模长表 Schema

默认输出：

```text
outputs/tables/modeling_toxicity_long.parquet
outputs/tables/modeling_toxicity_long.csv
outputs/reports/modeling_table_build_report.json
```

必需字段：

| 字段 | 类型 | 来源/计算 | 说明 |
|---|---|---|---|
| `record_id` | string | `result_id` 或 hash | 建模记录唯一 ID |
| `result_id` | string/int | `results.result_id` | 原始结果 ID |
| `test_id` | string/int | `results.test_id` | 测试 ID |
| `reference_number` | string/int | `tests.reference_number` | 文献关联 |
| `chemical_id` | string | `cas_number` 优先 | 项目内部化合物 ID |
| `casrn` | string/null | `chemicals.cas_number` | CAS 号 |
| `dtxsid` | string/null | `chemicals.dtxsid` | CompTox ID |
| `smiles` | string/null | `chemicals.smiles` | 分子结构 |
| `species_id` | string | `species.species_number` | 项目内部物种 ID |
| `species_number` | int/string | `species.species_number` | ECOTOX 物种 ID |
| `scientific_name` | string | `species.latin_name` | 学名 |
| `taxonomy_kingdom` | string/null | `species.kingdom` | 分类学 |
| `taxonomy_phylum` | string/null | `species.phylum_division` | 分类学 |
| `taxonomy_class` | string/null | `species.class` | 分类学 |
| `taxonomy_order` | string/null | `species.tax_order` | 分类学 |
| `taxonomy_family` | string/null | `species.family` | 分类学 |
| `taxonomy_genus` | string/null | `species.genus` | 分类学 |
| `species_ecotox_group` | string/null | `species.ecotox_group` | 生态/类群 |
| `primary_medium` | string/null | `species.primary_medium` | 主生存介质 |
| `organism_lifestage` | string/null | `tests.organism_lifestage` | life stage 原始字段 |
| `endpoint_raw` | string | `results.endpoint` | 原始终点 |
| `endpoint_family` | category | parser | `LC/EC/LOEC` |
| `effect_level` | float/null | parser | LC50/EC10 等数字 |
| `effect` | string/null | `results.effect` | 效应 |
| `measurement` | string/null | `results.measurement` | 测量项 |
| `trend` | string/null | `results.trend` | 趋势 |
| `endpoint_comments` | string/null | `results.endpoint_comments` | 终点备注 |
| `response_site_comments` | string/null | `results.response_site_comments` | 响应部位备注 |
| `conc_value` | float/null | 规则派生 | 优先 `conc1_mean` |
| `conc_unit` | string/null | `conc1_unit` | 浓度单位 |
| `conc_derivation_method` | string | 规则派生 | `mean/direct_range_midpoint/missing` |
| `conc1_type` | string/null | `results.conc1_type` | 质量/不确定度特征 |
| `duration_h` | float/null | 规则派生 | 暴露时长，小时 |
| `duration_derivation_method` | string | 规则派生 | 时长来源 |
| `duration_missing_flag` | bool | 规则派生 | 是否缺失 |
| `num_doses_used` | float | 规则派生 | 区间派生使用的剂量组数 |
| `target_mg_l` | float/null | 单位标准化 | 水相主目标 |
| `target_mol_l` | float/null | MW 换算 | 可计算 pTox 时使用 |
| `target_ptox` | float/null | `-log10(target_mol_l)` | 主回归目标 |
| `target_unit_family` | category | 单位标准化 | `water_mg_l/soil_mg_kg/oral_mg_kg_d/other` |
| `modeling_split_group` | string/null | 化学类别模块 | 留出类别 |
| `qa_flags` | string/list | 审计 | 分号或 JSON list |
| `is_main_water_task` | bool | 判定规则 | 是否进入水相主训练 |
| `is_transfer_candidate` | bool | 判定规则 | 是否进入迁移候选 |

## 4. 浓度派生规则

优先级：

```text
conc1_mean
> dose-grid midpoint from conc1_min/conc1_max
> missing
```

剂量组数 `n`：

```text
n = num_doses_mean
if missing: n = mean(num_doses_min, num_doses_max)
if still missing: n = 4
```

区间派生：

```text
dose_points = linspace(conc1_min, conc1_max, n, include_endpoints=True)
if n is odd:
    conc_value = middle dose point
if n is even:
    conc_value = mean(two middle dose points)
```

要求：

- `n` 最小有效值为 2；小于 2 或无法转为数值时按 4 处理并加入 `qa_flags`。
- 若 `conc1_min > conc1_max`，交换前必须记录 `qa_flags=conc_min_gt_max_swapped`。
- 若 `conc1_mean_op`、`conc1_min_op`、`conc1_max_op` 含 `<` 或 `>`，保留运算符字段，并加入 censor/qualifier 标记。
- `conc1_type` 不筛选记录，只作为质量和不确定度特征。

## 5. 暴露时长派生规则

优先级：

```text
exposure_duration_mean + exposure_duration_unit
> obs_duration_mean + obs_duration_unit
> exposure_duration_min/max + exposure_duration_unit
> obs_duration_min/max + obs_duration_unit, if available
> missing and manual review
```

当前 clean SQLite 已保留 `exposure_duration_min/max`；若没有 `obs_duration_min/max` 字段，则该分支自动跳过。

区间派生使用与浓度相同的 `num_doses` 中间剂量点规则。所有时长统一为小时：

```text
duration_h = standardize_duration(value, unit)
```

输出 `duration_derivation_method`：

- `exposure_mean`
- `observation_mean`
- `exposure_range_grid_mid`
- `observation_range_grid_mid`
- `missing_manual_review`

## 6. 单位与目标值规则

主任务：

```text
water concentration -> mg/L -> mol/L -> pTox
target_ptox = -log10(target_mol_l)
```

单位族：

| 单位族 | 标准单位 | 第一阶段用途 |
|---|---|---|
| `water_mg_l` | `mg/L` | 主训练 |
| `soil_mg_kg` | `mg/kg` | 迁移学习候选 |
| `sediment_mg_kg` | `mg/kg` | 迁移学习候选 |
| `oral_mg_kg_d` | `mg/kg/d` | 暂存，不进主任务 |
| `other` | 原单位 | 标记，等待人工确认 |

MW 来源：

1. RDKit 从 SMILES 计算 `molecular_weight_rdkit_g_mol`。
2. 若 RDKit 失败，可使用化合物表或外部结构映射中的 `molecular_weight_g_mol`。
3. MW 不可用时，`target_mol_l` 和 `target_ptox` 为空，记录不进入 pTox 训练。

## 7. Python 接口

包路径：

```text
src/qsar_dl/data/
```

必须实现：

```python
from pathlib import Path
from typing import Mapping

import pandas as pd


def load_clean_sqlite(database_path: Path) -> Mapping[str, pd.DataFrame]:
    """Load clean ECOTOX tables or joined view from SQLite."""


def derive_concentration(row: Mapping[str, object]) -> dict[str, object]:
    """Return conc_value, conc_unit, num_doses_used, derivation method, QA flags."""


def derive_duration(row: Mapping[str, object]) -> dict[str, object]:
    """Return duration_h, derivation method, missing flag, QA flags."""


def parse_endpoint(endpoint: str, effect: str | None = None) -> dict[str, object]:
    """Return endpoint_family and effect_level."""


def standardize_target_units(row: Mapping[str, object]) -> dict[str, object]:
    """Return target_mg_l, target_mol_l, target_ptox, target_unit_family."""


def build_modeling_table(config_path: Path) -> pd.DataFrame:
    """Build the canonical modeling long table and write configured outputs."""
```

## 8. 配置键

默认配置位置：

```text
configs/data/ecotox_clean_sqlite.yaml
```

最小配置：

```yaml
data:
  clean_sqlite: outputs/databases/ecotox_clean.sqlite
  joined_view: ecotox_toxicity_joined
  output_table: outputs/tables/modeling_toxicity_long.parquet
  output_csv: outputs/tables/modeling_toxicity_long.csv
  report_json: outputs/reports/modeling_table_build_report.json
target:
  main_medium_field: primary_medium
  main_medium_values: [aquatic]
  main_unit_family: water_mg_l
  target_column: target_ptox
  endpoints: [LC, EC, LOEC]
derivation:
  default_num_doses: 4
  use_dose_grid_midpoint: true
```

修改路径时，只改 `data.clean_sqlite`、`data.output_table`、`data.report_json`，不要在代码里写死路径。

## 9. 验收标准

数据模块交付时必须满足：

- 能从 clean SQLite 构建 `modeling_toxicity_long.parquet`。
- 报告总行数、水相主任务行数、迁移候选行数、缺 SMILES 行数、缺 MW 行数。
- 报告每种 `conc_derivation_method` 和 `duration_derivation_method` 数量。
- 输出无法建模原因分布。
- 至少包含 10 条 fixture 测试，覆盖 mean、range、单位失败、duration 缺失、endpoint parser、非水相迁移候选。
