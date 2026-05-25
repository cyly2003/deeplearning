# ECOTOX Clean SQLite 构建说明

## 输出文件

- 数据库：`outputs/databases/ecotox_clean.sqlite`
- 构建报告：`outputs/reports/ecotox_clean_build_report.json`
- 构建脚本：`src/build_clean_ecotox_sqlite.py`
- 标准化脚本：`src/standardize_clean_ecotox_sqlite.py`
- 标准化报告：`outputs/reports/ecotox_clean_standardization_report.json`

`outputs/` 下的大文件默认被 `.gitignore` 忽略；脚本和说明文档会进入版本控制。

## 重建命令

```powershell
E:\TOOLS\anaconda\python.exe src\build_clean_ecotox_sqlite.py --overwrite
```

构建或更新 `species.primary_medium` 后，运行标准化写回：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe src\standardize_clean_ecotox_sqlite.py `
  --database outputs\databases\ecotox_clean.sqlite `
  --report-json outputs\reports\ecotox_clean_standardization_report.json
```

默认输入：

- 原始 SQLite：`G:\ECOTOX_data\ECOTOX_SQLite\ECOTOX_ASCII.sqlite`
- SMILES 字典：`G:\新数据\新建文件夹\outputs\tables\compound_toxicity_master.csv`

如需更换路径：

```powershell
E:\TOOLS\anaconda\python.exe src\build_clean_ecotox_sqlite.py `
  --source-sqlite "G:\ECOTOX_data\ECOTOX_SQLite\ECOTOX_ASCII.sqlite" `
  --smiles-dictionary "G:\新数据\新建文件夹\outputs\tables\compound_toxicity_master.csv" `
  --output-sqlite "outputs\databases\ecotox_clean.sqlite" `
  --report-json "outputs\reports\ecotox_clean_build_report.json" `
  --overwrite
```

## 表结构摘要

新数据库包含：

- `results`：保留 `result_id`、`test_id` 以及观测时长、浓度、终点、趋势、效应、测量、响应部位备注等字段。
- `tests`：保留 `test_id`、`reference_number`、`test_cas`、`species_number` 以及纯度、栖息地、生活史阶段、暴露时长、介质类型、剂量数量等字段。
- `chemicals`：完整保留原始 `chemicals` 字段，并新增 `smiles`、`molecular_weight_rdkit_g_mol`、`molecular_weight_g_mol`、`molecular_weight_source`、`molecular_weight_status`。
- `results`：保留原始浓度和观测时长字段，并新增 `obs_duration_mean_h`、`conc1_mean_standardized`、`conc1_min_standardized`、`conc1_max_standardized`、`conc1_standard_unit`、`conc1_unit_family`、`conc1_standardization_status`。
- `tests`：保留原始暴露时长字段，并新增 `exposure_duration_mean_h`、`exposure_duration_min_h`、`exposure_duration_max_h`、`exposure_duration_standardization_status`。
- `species`：完整复制原始 `species` 表。
- `references`：完整复制原始 `references` 表，便于通过 `reference_number` 回溯文献。
- `chemical_smiles_dictionary`：保留用于合并 SMILES 的去重字典记录，便于追踪来源。
- `ecotox_toxicity_joined`：分析便利视图，按 `results -> tests -> chemicals/species/references` 关联。

## 字段命名说明

ECOTOX 原始 `tests` 表中剂量字段实际命名为 `num_doses_*`，不是 `number_doses_*`。脚本保留原始字段名，避免后续与源库核对时产生歧义。

脚本同时保留关键数值字段对应的 `_op` 运算符字段，例如 `conc1_mean_op`、`test_purity_mean_op`、`exposure_duration_mean_op`。这些字段可能表示 `<`、`>` 等限定符，对毒性值解释和后续 QSAR 建模筛选很重要。

标准化字段说明：

- 时间统一为小时，字段后缀为 `_h`。
- 水相质量浓度统一为 `mg/L`。
- 水相摩尔浓度统一为 `mol/L`，后续建模长表会结合 MW 派生 `target_mol_l` 和 `target_ptox`。
- 土壤/沉积物质量浓度统一为 `mg/kg`，不直接换算为水相 `pTox`。
- 经口剂量统一为 `mg/kg/d`，暂不进入第一阶段主任务。
- 无法识别的单位保留原始值，`conc1_unit_family=other`，不进入主水相建模。

## 当前构建质量检查

最近一次构建报告显示：

- `results`：1,234,077 行
- `tests`：724,182 行
- `chemicals`：18,520 行
- `species`：29,598 行
- `references`：131,197 行
- 有 SMILES 的化合物：16,582 个
- 无 SMILES 的化合物：1,938 个
- `results -> tests`、`tests -> chemicals/species/references` 均无缺失关联

最近一次标准化报告显示：

- RDKit MW 可用化合物：16,580 个
- 缺 SMILES：1,938 个
- 无效 SMILES：2 个
- `results.conc1_*` 可标准化记录：551,804 行
- `results.obs_duration_mean` 可标准化记录：1,052,479 行
- `tests.exposure_duration_*` 可标准化记录：366,100 行
