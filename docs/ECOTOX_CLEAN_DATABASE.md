# ECOTOX Clean SQLite 构建说明

## 输出文件

- 数据库：`outputs/databases/ecotox_clean.sqlite`
- 构建报告：`outputs/reports/ecotox_clean_build_report.json`
- 构建脚本：`src/build_clean_ecotox_sqlite.py`

`outputs/` 下的大文件默认被 `.gitignore` 忽略；脚本和说明文档会进入版本控制。

## 重建命令

```powershell
E:\TOOLS\anaconda\python.exe src\build_clean_ecotox_sqlite.py --overwrite
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
- `chemicals`：完整保留原始 `chemicals` 字段，并新增 `smiles`。
- `species`：完整复制原始 `species` 表。
- `references`：完整复制原始 `references` 表，便于通过 `reference_number` 回溯文献。
- `chemical_smiles_dictionary`：保留用于合并 SMILES 的去重字典记录，便于追踪来源。
- `ecotox_toxicity_joined`：分析便利视图，按 `results -> tests -> chemicals/species/references` 关联。

## 字段命名说明

ECOTOX 原始 `tests` 表中剂量字段实际命名为 `num_doses_*`，不是 `number_doses_*`。脚本保留原始字段名，避免后续与源库核对时产生歧义。

脚本同时保留关键数值字段对应的 `_op` 运算符字段，例如 `conc1_mean_op`、`test_purity_mean_op`、`exposure_duration_mean_op`。这些字段可能表示 `<`、`>` 等限定符，对毒性值解释和后续 QSAR 建模筛选很重要。

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
