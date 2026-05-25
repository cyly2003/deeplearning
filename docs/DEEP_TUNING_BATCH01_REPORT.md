# Deep Residual-QSAR 全量调参批次 01 报告

日期：2026-05-25

## 1. 实验定位

本轮实验在上一轮全量消融最佳结构 `chemical_species_context` 基础上进行超参数调优。模型结构保持三层 residual-QSAR 设计中的前两层启用：

1. 化合物结构主效应层：RDKit 描述符、物化性质缺失标记和 Morgan fingerprint。
2. 物种/实验上下文残差层：endpoint、exposure duration、`primary_medium`、lifestage、taxon group 和物种状态标记。
3. 机制/规则残差层：继续禁用，等待 mechanistic rules 完成校准后再进入正式消融。

本轮目标不是扩大样本抽样，而是在全量主水相任务上比较学习率、权重衰减、dropout 和 batch size 对泛化性能的影响。

## 2. 运行命令

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe src\run_deep_tuning_experiment.py `
  --device cpu `
  --output-dir outputs\models\deep_tuning_batch01
```

输出目录：

- `outputs/models/deep_tuning_batch01/`

数据范围：

| 指标 | 数量 |
|---|---:|
| 主水相任务记录数 | 226,123 |
| 训练集记录数 | 185,158 |
| 验证集记录数 | 40,965 |
| 化合物数 | 4,987 |

该运行未使用 `max_rows` 采样。

## 3. 调参候选

所有候选均启用 endpoint、duration 和 species/taxon context。

| 调参实验 | 学习率 | weight decay | dropout | batch size | max epochs | patience |
|---|---:|---:|---:|---:|---:|---:|
| `ctx_baseline_lr1e3_wd1e4_do10` | 0.0010 | 0.0001 | 0.10 | 1,024 | 8 | 3 |
| `ctx_lr3e4_wd1e3_do20` | 0.0003 | 0.0010 | 0.20 | 1,024 | 12 | 4 |
| `ctx_lr3e4_wd1e4_do20` | 0.0003 | 0.0001 | 0.20 | 1,024 | 12 | 4 |
| `ctx_lr1e3_wd1e3_do20` | 0.0010 | 0.0010 | 0.20 | 1,024 | 12 | 4 |
| `ctx_lr1e3_wd1e3_do30_bs512` | 0.0010 | 0.0010 | 0.30 | 512 | 12 | 4 |
| `ctx_lr2e3_wd1e3_do20` | 0.0020 | 0.0010 | 0.20 | 1,024 | 12 | 4 |

模型输入维度：

| 输入 | 维度 |
|---|---:|
| 描述符/物化性质 | 15 |
| Morgan fingerprint | 2,048 |
| endpoint one-hot | 3 |
| 物种上下文 | 230 |

## 4. 全量验证结果

| 调参实验 | 训练 R2 | 验证 R2 | 验证 RMSE | 验证 MAE | 验证 MAPE |
|---|---:|---:|---:|---:|---:|
| `ctx_baseline_lr1e3_wd1e4_do10` | 0.711 | 0.413 | 1.351 | 1.026 | 32.32% |
| `ctx_lr3e4_wd1e3_do20` | 0.649 | 0.386 | 1.383 | 1.061 | 32.84% |
| `ctx_lr3e4_wd1e4_do20` | 0.649 | 0.386 | 1.383 | 1.061 | 32.84% |
| `ctx_lr1e3_wd1e3_do20` | 0.698 | 0.434 | 1.327 | 1.011 | 32.85% |
| `ctx_lr1e3_wd1e3_do30_bs512` | 0.698 | 0.412 | 1.353 | 1.026 | 30.61% |
| `ctx_lr2e3_wd1e3_do20` | 0.725 | 0.414 | 1.350 | 1.032 | 30.57% |

当前最佳候选：

- `ctx_lr1e3_wd1e3_do20`
- 验证 R2：0.434
- 验证 RMSE：1.327 pTox
- 验证 MAE：1.011 pTox

相对上一轮全量消融最佳 `chemical_species_context`：

| 指标 | 消融最佳 | 调参最佳 | 变化 |
|---|---:|---:|---:|
| 验证 R2 | 0.413 | 0.434 | +0.021 |
| 验证 RMSE | 1.351 | 1.327 | -0.024 |
| 验证 MAE | 1.026 | 1.011 | -0.016 |

## 5. 结果解释

本轮结果说明，深度模型当前主要受正则化强度和训练动态影响：

- `weight_decay=1e-3`、`dropout=0.20` 在保持训练性能不过度下降的情况下提升了验证 R2，说明上一轮 baseline 存在一定过拟合。
- `lr=3e-4` 两组验证性能较低，训练 R2 也较低，说明在当前 8-12 epoch 的预算下学习率偏小，模型可能欠拟合。
- `dropout=0.30 + batch_size=512` 让 MAPE 下降，但 R2/RMSE 变差，说明其对相对误差有帮助，但总体方差解释能力不足。
- `lr=2e-3` 训练 R2 最高但验证 R2 未同步提升，提示更高学习率可能加重过拟合或收敛到不够稳定的解。

科研解释上，当前最优深度模型已经证明物种/实验上下文残差层可被调参进一步释放，但仍未超过 curated 传统 ML baseline 的 `standard + LightGBM`。这意味着下一步不应只盲目扩大网络，而应优先做分层误差诊断、适用域约束和 endpoint/taxon 类别不平衡处理。

## 6. 输出图表

已输出 PNG 和 PDF：

- `深度调参_验证集指标对比`
- `深度调参_训练验证损失曲线`
- `深度调参_验证集真实预测散点`
- `深度调参_验证集残差分层_终点类型`
- `深度调参_验证集残差分层_化学类别`
- `深度调参_验证集残差分层_物种类群`

图表目录：

- `outputs/models/deep_tuning_batch01/图表/`

表格目录：

- `outputs/models/deep_tuning_batch01/表格/`

核心表格：

- `深度调参_模型指标汇总.csv`
- `深度调参_全部预测结果.parquet`

## 7. 下一步

1. 固定 `ctx_lr1e3_wd1e3_do20` 作为当前深度模型候选主线。
2. 输出该候选的 endpoint、化学类别、物种类群和 AD 分层指标表。
3. 加入更严格的 chemical-space AD：Morgan Tanimoto、descriptor distance、类别外推标记。
4. 评估 LOEC 是否需要单独建模或作为慢性毒性子任务展示。
5. 在规则层校准前继续保持 mechanistic-rule residual 禁用，避免机制解释过度。
