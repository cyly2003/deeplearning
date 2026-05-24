# 当前并行执行计划

日期：2026-05-24

## 当前约束

- 任务 E 物种上下文特征暂缓，原因是物种特征仍在构建中。
- 第一波并行任务不得依赖完整 species feature 模块。
- 允许读取 clean SQLite 中已有的 `species.primary_medium`，但不开发新的物种编码模块。
- 不修改 `src/build_clean_ecotox_sqlite.py` 和 `src/annotate_species_habitat.py`，这两个脚本保留为 clean database 准备层。

## 第一波并行任务

| 任务 | 状态 | 写入范围 |
|---|---|---|
| A 配置系统 | running | `src/qsar_dl/config/*`, `configs/*` |
| B 数据契约管线 | running | `src/qsar_dl/data/*`, `tests/unit/test_data_contract.py` |
| C 化合物特征 | running | `src/qsar_dl/features/chemical.py`, `configs/features/chemical_rdkit_morgan.yaml` |
| D 描述符分组加权 | running | `src/qsar_dl/features/descriptor_groups.py`, `src/qsar_dl/models/descriptor_weighting.py` |
| F 显式规则层骨架 | running | `src/qsar_dl/rules/*`, `configs/rules/mechanistic_rules.yaml` |
| L 可视化样式库 | running | `src/qsar_dl/visualization/*` |

## 占位配置

为保证主实验配置可解析，当前保留两个占位配置：

- `configs/features/species_context.yaml`：任务 E 暂缓，`species.enabled=false`。
- `configs/evaluation/chemical_category_holdout.yaml`：评估模块尚未实现，但固定化学类别留出配置键。

## 暂缓任务

| 任务 | 暂缓原因 | 恢复条件 |
|---|---|---|
| E 物种上下文特征 | taxonomy/eco group/primary_medium/lifestage 特征仍在构建 | 字段字典和编码规则确认 |
| G 传统 ML baseline | 依赖 B/C/D 输出稳定 | 数据、化学特征、分组特征接口合并后 |
| H 深度模型 | 依赖 B/C/D/F 输出稳定 | 第一波模块合并并通过 smoke test |
| I/J/K 评估、AD、不确定度、SSD | 依赖模型输出 | baseline 与主模型可运行后 |

## 合并检查

第一波合并前必须检查：

```powershell
git status --short --branch
git diff --check
E:\TOOLS\anaconda\python.exe -m pytest
```

如果依赖缺失导致全量测试不能运行，至少运行可导入检查：

```powershell
E:\TOOLS\anaconda\python.exe -c "import qsar_dl; print(qsar_dl.__version__)"
```
