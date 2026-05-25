# 当前并行执行计划

日期：2026-05-25

## 当前状态

第一波和第二波 worker 已全部返回，主线程已完成最终集成。当前阶段从并行骨架开发切换为真实数据管线验证。

已确认：

- clean SQLite 已存在：`outputs/databases/ecotox_clean.sqlite`
- `species.primary_medium` 已完成标注，分布为 `aquatic=9775`、`sediment=513`、`soil=5019`、`terrestrial=13277`、`unknown=1014`
- 已运行 `src/standardize_clean_ecotox_sqlite.py`，`ecotox_toxicity_joined` 视图直接包含 `primary_medium`、MW、标准化浓度和标准化时间字段
- `qsar-ph3` 环境全量单元测试通过：`94 passed`
- base Anaconda 环境可完整收集测试：`90 passed, 4 skipped`，跳过项来自 `sklearn/scipy/numpy` DLL 问题和缺少 parquet 引擎
- 已完成全量主水相 PLS/ElasticNet baseline 和 50,000 行 RandomForest smoke

详细进度见 `docs/PROJECT_PROGRESS.md`。

## 已完成任务

| 任务 | 状态 | 写入范围 |
|---|---|---|
| A 配置系统 | completed | `src/qsar_dl/config/*`, `configs/*` |
| B 数据契约管线 | completed-unit | `src/qsar_dl/data/*`, `tests/unit/test_data_contract.py` |
| C 化合物特征 | completed-unit | `src/qsar_dl/features/chemical.py`, `configs/features/chemical_rdkit_morgan.yaml` |
| D 描述符分组加权 | completed | `src/qsar_dl/features/descriptor_groups.py`, `src/qsar_dl/models/descriptor_weighting.py` |
| E 物种上下文特征 | completed | `src/qsar_dl/features/species.py`, `configs/features/species_context.yaml` |
| F 显式规则层骨架 | completed-unit | `src/qsar_dl/rules/*`, `configs/rules/mechanistic_rules.yaml` |
| G 传统 ML baseline | completed-unit | `src/qsar_dl/training/baseline_ml.py`, `configs/experiments/baseline_ml.yaml` |
| H 深度模型骨架 | completed-smoke | `src/qsar_dl/models/residual_qsar.py`, `src/qsar_dl/training/train_deep.py` |
| I 评估与化学类别留出 | completed-unit | `src/qsar_dl/evaluation/*`, `configs/evaluation/chemical_category_holdout.yaml` |
| J AD 与不确定度 | completed-unit | `src/qsar_dl/applicability_domain/*`, `src/qsar_dl/uncertainty/*` |
| K SSD 应用示范 | completed-unit | `src/qsar_dl/ssd/*` |
| L 可视化样式库 | completed-unit | `src/qsar_dl/visualization/*` |
| M 远程服务器运行 | completed-static | `scripts/*`, `docs/SERVER_RUNBOOK.md`, `configs/runtime/gpu_server.yaml` |

## 集成检查

推荐主检查：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m pytest
```

当前结果：

```text
94 passed
```

base 环境兼容检查：

```powershell
E:\TOOLS\anaconda\python.exe -m pytest
```

当前结果：

```text
90 passed, 4 skipped, 1 warning
```

说明：base 环境的跳过项来自 `sklearn/scipy/numpy` DLL 链接异常和缺少 parquet 引擎。`baseline_ml` 已做延迟导入处理，依赖异常不会影响模块导入和测试收集。

## 当前优先级

### P0：真实建模长表

已完成数据契约管线运行，从 clean SQLite 生成：

- `outputs/tables/modeling_toxicity_long.parquet`
- `outputs/tables/modeling_toxicity_long.csv`
- `outputs/reports/modeling_table_build_report.json`

审计重点：

- 主水相任务记录数
- 迁移候选记录数
- endpoint 分布
- 单位族分布
- 缺 SMILES、缺 MW、无法计算 `target_ptox` 的原因

当前结果：

- 主水相任务记录数：226,123
- 宽松迁移候选记录数：526,114
- 严格迁移可建模记录数：8,067
- parquet 和 CSV 均已写出

### P1：真实 baseline

已完成全量传统 ML baseline：

- 入口：`src/run_baseline_ml_experiment.py`
- 全量主水相数据：226,123 条记录，4,987 个化合物
- 输出：`outputs/models/baseline_ml_v001/baseline_metrics.json`
- 标准特征 PLS：R2=0.307，RMSE=1.661，MAE=1.363，MAPE=31.98%
- 标准特征 ElasticNet：R2=0.280，RMSE=1.693，MAE=1.394，MAPE=32.38%
- 标准特征 RandomForest：R2=0.519，RMSE=1.383，MAE=1.072，MAPE=22.76%
- 标准特征 XGBoost：R2=0.502，RMSE=1.408，MAE=1.126，MAPE=24.53%
- 标准特征 LightGBM：R2=0.556，RMSE=1.329，MAE=1.047，MAPE=23.07%
- 固定描述符分组 PLS：R2=0.303，RMSE=1.665，MAE=1.357，MAPE=31.97%
- 固定描述符分组 ElasticNet：R2=0.269，RMSE=1.706，MAE=1.407，MAPE=33.11%
- 固定描述符分组 RandomForest：R2=0.504，RMSE=1.405，MAE=1.084，MAPE=23.55%
- 固定描述符分组 XGBoost：R2=0.477，RMSE=1.443，MAE=1.160，MAPE=25.22%
- 固定描述符分组 LightGBM：R2=0.544，RMSE=1.347，MAE=1.065，MAPE=23.13%

图表和表格：

- `outputs/models/baseline_ml_v001/表格/全量基线_模型指标汇总.csv`
- `outputs/models/baseline_ml_v001/表格/全量基线_验证集预测结果.parquet`
- `outputs/models/baseline_ml_v001/图表/全量基线_数据筛选与切分流程.png`
- `outputs/models/baseline_ml_v001/图表/全量基线_模型性能对比.png`
- 每个模型/特征集均有中文命名的真实值-预测值图和残差分布图。

待继续：

- 分 endpoint、chemical category、primary medium、taxon、AD 的分层指标
- 论文级分层图、误差来源诊断图和外推类别表现图

### P2：深度模型真实数据 smoke

在 baseline 和特征表稳定后：

- 构建最小 tensor batch
- 验证 `ResidualQSARModel` 在真实派生特征上的 forward
- 暂不直接追求性能，先验证接口和预测分解

## 暂缓任务

| 任务 | 暂缓原因 | 恢复条件 |
|---|---|---|
| 潜在暴露途径特征 | 暴露途径辞典尚未制作 | 字段字典和编码规则确认 |
| chemical activity 规则 | 缺自由浓度和 mol/L 溶解度字段 | 标准化字段进入数据契约 |
| MoA excess toxicity 规则 | MoA/ToxCast/alert 输入未稳定 | 化合物机制特征表稳定 |
| TKTD/volatility/bioavailability 规则 | 参数和输入字段未校准 | 完成真实数据审计和规则输入覆盖率报告 |
