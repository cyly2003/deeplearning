# 全量 Deep Residual-QSAR 消融报告

日期：2026-05-25

## 1. 实验定位

本轮实验按原始 residual-QSAR 设计推进深度学习消融，而不是只做 chemical-only smoke。当前框架按三层展开：

1. 化合物结构主效应层：RDKit 描述符、物化性质缺失标记和 Morgan fingerprint。
2. 物种/实验上下文残差层：endpoint、exposure duration、`primary_medium`、lifestage、taxon group。
3. 机制/规则残差层：当前仍未启用，因为 mechanistic rules 尚未完成校准，不能作为正式机制证据解释。

## 2. 数据范围

输出目录：

- `outputs/models/deep_ablation_full_v001/`

全量主任务数据：

| 指标 | 数量 |
|---|---:|
| 主水相任务记录数 | 226,123 |
| 训练集记录数 | 185,158 |
| 验证集记录数 | 40,965 |
| 化合物数 | 4,987 |

该运行未使用 `max_rows` 采样。

## 3. 消融设置

| 消融实验 | 化合物结构 | Endpoint | Duration | Species/Taxon | 规则层 |
|---|---|---|---|---|---|
| `chemical_only` | 是 | 否 | 否 | 否 | 否 |
| `chemical_endpoint_duration` | 是 | 是 | 是 | 否 | 否 |
| `chemical_species_context` | 是 | 是 | 是 | 是 | 否 |

模型输入维度：

| 消融实验 | 描述符维度 | Morgan fingerprint | endpoint 维度 | 物种上下文维度 |
|---|---:|---:|---:|---:|
| `chemical_only` | 15 | 2,048 | 0 | 0 |
| `chemical_endpoint_duration` | 15 | 2,048 | 3 | 0 |
| `chemical_species_context` | 15 | 2,048 | 3 | 148 |

## 4. 全量验证结果

| 消融实验 | 训练 R2 | 验证 R2 | 验证 RMSE | 验证 MAE | 验证 MAPE |
|---|---:|---:|---:|---:|---:|
| `chemical_only` | 0.590 | 0.288 | 1.489 | 1.150 | 32.60% |
| `chemical_endpoint_duration` | 0.637 | 0.350 | 1.423 | 1.096 | 33.24% |
| `chemical_species_context` | 0.711 | 0.413 | 1.351 | 1.026 | 32.32% |

## 5. 结果解释

消融结果符合预期：

- 从 `chemical_only` 到 `chemical_endpoint_duration`，验证 R2 从 0.288 提高到 0.350，说明 endpoint 和暴露时长确实解释了 ECOTOX 多 endpoint 合并任务中的系统差异。
- 从 `chemical_endpoint_duration` 到 `chemical_species_context`，验证 R2 进一步提高到 0.413，说明 species/taxon context 对跨物种毒性差异有增益。
- 训练 R2 与验证 R2 差距仍较大，说明深度模型仍有过拟合或外推不足风险，后续需要正则化、early stopping、dropout、类别外推评估和 AD 约束共同优化。
- 当前最佳深度模型仍低于 curated 传统 ML baseline 的 `standard + LightGBM`，但三层 residual 方向已经显示了合理增益。

科研解释上，这说明化学结构是主信号，但单靠结构无法充分解释同一化合物在不同 endpoint、暴露时长和物种类群上的毒性差异。物种上下文残差层不是附加装饰，而是对多物种 QSAR 有实际贡献。

## 6. 输出图表

已输出 PNG 和 PDF：

- `深度消融_验证集指标对比`
- `深度消融_训练验证损失曲线`
- `深度消融_验证集真实预测散点`
- `深度消融_验证集残差分层_终点类型`
- `深度消融_验证集残差分层_化学类别`
- `深度消融_验证集残差分层_物种类群`

图表目录：

- `outputs/models/deep_ablation_full_v001/图表/`

表格目录：

- `outputs/models/deep_ablation_full_v001/表格/`

核心表格：

- `深度消融_模型指标汇总.csv`
- `深度消融_全部预测结果.parquet`

## 7. 下一步

1. 针对 `chemical_species_context` 做分层指标表，重点比较 endpoint、化学类别、物种类群上的误差变化。
2. 加强 AD：Morgan fingerprint Tanimoto、descriptor distance、chemical category holdout、species support。
3. 优化深度模型正则化，降低训练/验证差距。
4. 在规则层校准前，继续保持 mechanistic-rule residual 禁用，避免机制解释过度。
