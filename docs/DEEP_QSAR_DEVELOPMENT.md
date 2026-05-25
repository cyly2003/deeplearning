# 深度学习 QSAR 开发记录

日期：2026-05-25

## 1. 当前目标

在传统机器学习 baseline 已完成后，深度学习部分先建立一个可复现、可扩展、可诊断的真实数据训练链路。当前阶段不急于追求超过 LightGBM，而是先确保：

- 真实 ECOTOX 建模长表可以进入 PyTorch 训练；
- 化合物结构信号可以作为主预测分支；
- 训练/验证 split 与传统 baseline 保持同一类 chemical-category holdout 逻辑；
- 输出指标、预测表和后续 residual decomposition 字段；
- 后续可以逐步加入 endpoint、duration、species/taxon 和 mechanistic-rule residual 分支。

## 2. 新增实现

核心训练模块：

- `src/qsar_dl/training/train_deep.py`

新增命令行入口：

- `src/run_deep_qsar_experiment.py`

默认配置：

- `configs/experiments/baseline_deep.yaml`

新增能力：

- 解析真实数据中的 RDKit 描述符和 Morgan fingerprint。
- 基于训练集对连续描述符进行 median imputation 和标准化。
- 基于训练集 `target_ptox` 进行目标标准化，指标计算时反变换回原始 pTox。
- 复用 chemical-category split，避免同一化合物跨训练/验证造成泄漏。
- 支持 PyTorch DataLoader、AdamW、早停、验证集指标和预测表导出。
- 保留 `y_chemical`、`y_context_residual`、`uncertainty` 等字段，方便后续解释 residual-QSAR 贡献。

## 3. 当前模型范围

本轮 deep baseline 是 chemical-only 版本：

- 输入：15 个结构/物化描述符 + 2048 位 Morgan fingerprint。
- 不启用 species context。
- 不启用 exposure duration。
- 不启用 mechanistic rule residual。
- 不启用 endpoint one-hot。

当前默认配置已改为全量消融：

| 参数 | 当前值 |
|---|---:|
| `max_rows` | null |
| `batch_size` | 1,024 |
| `max_epochs` | 8 |
| `learning_rate` | 0.001 |
| `weight_decay` | 0.0001 |
| `patience` | 3 |
| device | `auto` |

## 4. 全量消融结果

运行命令：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe src\run_deep_qsar_experiment.py `
  --device cpu `
  --output-dir outputs\models\deep_ablation_full_v001
```

输出目录：

- `outputs/models/deep_ablation_full_v001/`

数据范围：

| 指标 | 数量 |
|---|---:|
| 主水相任务记录数 | 226,123 |
| 训练集记录数 | 185,158 |
| 验证集记录数 | 40,965 |
| 化合物数 | 4,987 |

消融结果：

| 消融实验 | 验证 R2 | 验证 RMSE | 验证 MAE | 验证 MAPE |
|---|---:|---:|---:|---:|
| `chemical_only` | 0.288 | 1.489 | 1.150 | 32.60% |
| `chemical_endpoint_duration` | 0.350 | 1.423 | 1.096 | 33.24% |
| `chemical_species_context` | 0.413 | 1.351 | 1.026 | 32.32% |

## 5. 科研解释

当前 deep baseline 仍弱于 curated 传统 ML baseline 中的 `standard + LightGBM`，但全量消融已经验证了原始三层设计方向：

- chemical-only 是结构主效应基线。
- endpoint + duration 带来验证 R2 提升，说明实验上下文确实解释了多 endpoint 合并任务的系统差异。
- species/taxon context 进一步提升，说明物种上下文残差层对多物种 QSAR 有实际贡献。
- 训练和验证仍存在差距，后续需要加强正则化、AD 和外推类别评估。

当前结果的价值在于：三层 residual-QSAR 框架已经从概念设计推进到全量真实数据消融验证。

## 6. 下一步开发顺序

建议按消融顺序继续：

1. `chemical-only` 全量或服务器版训练，确认深度模型结构信号上限。
2. 加入 endpoint one-hot 和 duration branch，判断 LC/EC/LOEC 合并任务中实验上下文的贡献。
3. 加入 species/taxon context branch，重点观察 `amphibian`、`cyanobacteria` 和 `algae` 的误差是否下降。
4. 加入更严格 AD 输出，区分 chemical-space AD 内外表现。
5. 最后加入 mechanistic-rule residual，避免规则层在未校准前干扰主结构信号。

## 7. 注意事项

- 当前 deep baseline 不能直接作为论文主结果，只能作为模型开发起点。
- 与 LightGBM 比较时必须说明训练样本规模、特征分支和上下文是否一致。
- 后续如果启用 species/taxon context，应同步输出 residual decomposition，判断性能提升是否来自合理生态上下文，而不是数据泄漏或类群频次记忆。
