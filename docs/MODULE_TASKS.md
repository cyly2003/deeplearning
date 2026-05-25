# 并行开发任务规范

版本：v0.1
日期：2026-05-24

## 1. 合并原则

所有子任务遵循：

- 通过统一 Python 接口和配置键交互。
- 使用默认路径，但路径必须可通过配置修改。
- 不读取其他模块未声明的私有临时文件。
- 每个模块交付单元测试、fixture、report。
- 输出大文件写入 `outputs/`，不提交 Git。

## 2. 任务 A：配置系统

交付：

- `src/qsar_dl/config/`
- `configs/` 目录模板。
- `resolve_config` 和 `config_resolved.yaml` 导出。

依赖：无。

验收：

- 能加载 `configs/experiments/main_residual_qsar.yaml`。
- 缺少 include 时给出清晰错误。

## 3. 任务 B：数据契约管线

交付：

- `src/qsar_dl/data/`
- clean SQLite -> modeling long table。
- 数据审计报告。

依赖：任务 A。

主要文档：`DATA_CONTRACT.md`。

验收：

- 完成浓度派生、时长派生、endpoint parser、单位族标记。
- 输出水相主任务与迁移候选行数。

## 4. 任务 C：化合物特征

交付：

- `src/qsar_dl/features/chemical.py`
- RDKit 描述符。
- Morgan fingerprint。
- MW、logKow、logD、溶解度、Henry、KOC 等字段接口。

依赖：任务 A、B。

验收：

- 无效 SMILES 不崩溃。
- 输出缺失字段 mask。

## 5. 任务 D：描述符分组加权

交付：

- `src/qsar_dl/features/descriptor_groups.py`
- `src/qsar_dl/models/descriptor_weighting.py`
- RDKit 分组辞典模板。
- 固定权重组级特征导出。

依赖：任务 C。

主要文档：`DESCRIPTOR_GROUPING.md`。

验收：

- 可输出组级特征给传统 ML。
- 深度模块可导出组内/组间权重。

## 6. 任务 E：物种上下文特征

当前状态：已恢复。`species.primary_medium` 已完成标注，分布为
`aquatic=9775`、`sediment=513`、`soil=5019`、`terrestrial=13277`、`unknown=1014`。

交付：

- `src/qsar_dl/features/species.py`
- taxonomy、eco group、primary_medium、lifestage 编码。

依赖：任务 B。

验收：

- `primary_medium` 缺失或 unknown 的样本可标记并排除主任务。
- lifestage 缺失用 unknown + mask。

## 7. 任务 F：显式规则层

交付：

- `src/qsar_dl/rules/`
- 规则特征表。
- 规则覆盖率报告。

依赖：任务 B、C、E。

主要文档：`RULE_LAYER.md`。

验收：

- 每条规则有正常、缺输入、不适用测试。
- 输出解释文本和 missing input 标记。

## 8. 任务 G：传统 ML baseline

当前状态：已完成首轮真实数据运行。B/C/D/E 接口已稳定，入口脚本为
`src/run_baseline_ml_experiment.py`。

交付：

- `src/qsar_dl/training/baseline_ml.py`
- `src/run_baseline_ml_experiment.py`
- PLS、ElasticNet、SVR、RandomForest、XGBoost/LightGBM。

依赖：任务 B、C、D、E。

验收：

- 标准特征和固定分组特征各跑一套。
- 已输出全量 PLS/ElasticNet metrics。
- 已输出 50,000 行 RandomForest smoke metrics。
- predictions 和分层指标仍在下一步补齐。

## 9. 任务 H：深度模型

当前状态：已启动骨架开发。B/C/D/F 已有接口，E 输出按可选上下文接入。

交付：

- `src/qsar_dl/models/residual_qsar.py`
- `src/qsar_dl/training/train_deep.py`

依赖：任务 B、C、D、E、F。

主要文档：`MODEL_ARCHITECTURE.md`。

验收：

- chemical-only、without rules、with rules 都可跑。
- forward 输出预测分解。

## 10. 任务 I：评估与消融

交付：

- `src/qsar_dl/evaluation/`
- 化学类别留出。
- 消融报告。

依赖：任务 G、H。

主要文档：`EVALUATION_PROTOCOL.md`。

验收：

- 输出主表、分层指标、消融指标。
- 每个测试样本可追溯化学类别。

## 11. 任务 J：AD 与不确定度

交付：

- `src/qsar_dl/applicability_domain/`
- `src/qsar_dl/uncertainty/`

依赖：任务 H、I。

验收：

- 输出 AD 等级和 ensemble 区间。
- out-of-domain 样本单独汇总。

## 12. 任务 K：SSD 应用示范

交付：

- `src/qsar_dl/ssd/`
- SSD 拟合和绘图。

依赖：任务 H、J。

验收：

- 少于最小物种数时拒绝拟合并输出原因。
- 输出 HC5、HC10、bootstrap CI。

## 13. 任务 L：可视化样式库

交付：

- `src/qsar_dl/visualization/`

依赖：无。

验收：

- 中文不乱码。
- 导出 PNG/TIFF/PDF/SVG。
- 满足字体、字号、轴线和图例要求。

## 14. 任务 M：远程服务器运行

交付：

- `scripts/train_remote.ps1`
- `scripts/sync_to_server.ps1`
- `scripts/sync_from_server.ps1`
- `configs/runtime/gpu_server.yaml`

依赖：任务 A、H。

验收：

- dry-run 显示将执行的命令。
- 每次训练记录 git commit hash 和 resolved config。
