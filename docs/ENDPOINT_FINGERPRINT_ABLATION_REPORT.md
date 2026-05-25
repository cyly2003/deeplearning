# Endpoint 语义重编码与指纹消融报告

日期：2026-05-25

## 1. 实验定位

本轮按新的 endpoint 解释逻辑重构任务：

- `LCx` 不再作为独立 endpoint family，而是并入 `ECx` 的 `mortality` 响应域。
- `ECx`、`LOEC`、`NOEC` 分开建模。
- `ECx/LCx` 后面的数字作为连续型效应水平特征，包括 `effect_percent`、`effect_fraction` 和 `effect_level_logit`。
- `LOEC` 和 `NOEC` 作为阈值型统计终点，不与 ECx 点估计混合训练。

本轮先用 LightGBM 做传统机器学习诊断，用于判断 endpoint 重编码和 Morgan fingerprint 维度是否影响泛化。

## 2. 运行命令

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe src\run_endpoint_fingerprint_ablation.py `
  --output-dir outputs\models\endpoint_fingerprint_ablation_v001
```

输出目录：

- `outputs/models/endpoint_fingerprint_ablation_v001/`

核心表格：

- `outputs/models/endpoint_fingerprint_ablation_v001/表格/endpoint_ABCD_全部任务模型指标汇总.csv`

## 3. 数据范围

| 任务 | 记录数 | 训练样本数 | 验证样本数 |
|---|---:|---:|---:|
| ECx + LCx mortality | 153,327 | 131,051 | 22,276 |
| LOEC | 80,977 | 68,860 | 12,117 |
| NOEC | 106,736 | 92,052 | 14,684 |

总记录数：341,040。

切分方式仍为 chemical-category holdout，同一化合物不会跨训练/验证。

## 4. A/B/C/D 特征组

| 实验 | 特征定义 | 用途 |
|---|---|---|
| A | RDKit/物化描述符 + endpoint/species/context，无 Morgan fingerprint | 检查低维可解释基线 |
| B | A + 前 512 位 Morgan fingerprint | 检查低维指纹增益 |
| C | A + 2048 位 Morgan fingerprint | 对照当前高维指纹方案 |
| D | A + 训练集频率过滤后的 Morgan fingerprint，最多 512 位 | 降低稀疏/常数位噪声 |

## 5. 全量结果

| 任务 | 最佳特征组 | 验证 R2 | 验证 RMSE | 验证 MAE | 解释 |
|---|---|---:|---:|---:|---|
| ECx + LCx mortality | C 2048 位指纹 | 0.535 | 1.259 | 0.960 | 指纹有小幅增益，但 B 与 C 差距很小 |
| LOEC | B 512 位指纹 | 0.381 | 1.426 | 1.104 | 2048 位指纹明显变差，存在高维噪声/过拟合风险 |
| NOEC | C 2048 位指纹 | 0.465 | 1.327 | 1.062 | 2048 位指纹有增益，但相对 A 的提升有限 |

完整指标：

| 任务 | 特征组 | R2 | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|---:|
| ECx | A descriptors no FP | 0.521 | 1.278 | 0.972 | 45.22% |
| ECx | B descriptors + 512 FP | 0.534 | 1.260 | 0.960 | 46.05% |
| ECx | C descriptors + 2048 FP | 0.535 | 1.259 | 0.960 | 44.68% |
| ECx | D descriptors + filtered FP | 0.530 | 1.265 | 0.960 | 44.00% |
| LOEC | A descriptors no FP | 0.368 | 1.442 | 1.121 | 20.68% |
| LOEC | B descriptors + 512 FP | 0.381 | 1.426 | 1.104 | 20.51% |
| LOEC | C descriptors + 2048 FP | 0.325 | 1.489 | 1.155 | 21.22% |
| LOEC | D descriptors + filtered FP | 0.340 | 1.473 | 1.141 | 21.11% |
| NOEC | A descriptors no FP | 0.445 | 1.351 | 1.076 | 22.17% |
| NOEC | B descriptors + 512 FP | 0.447 | 1.348 | 1.080 | 22.80% |
| NOEC | C descriptors + 2048 FP | 0.465 | 1.327 | 1.062 | 22.25% |
| NOEC | D descriptors + filtered FP | 0.449 | 1.347 | 1.071 | 22.72% |

## 6. 结果解释

endpoint 重编码后，ECx 任务的验证 R2 达到 0.535，已经接近此前合并 endpoint baseline 的最佳水平。这说明粗粒度 `LC/EC/LOEC` 混合确实是压低模型表现的重要原因之一。

LOEC 是本轮最值得警惕的任务。512 位 fingerprint 有小幅增益，但 2048 位 fingerprint 使 R2 从 0.381 降到 0.325，说明 LOEC 对高维稀疏结构位更敏感，可能存在过拟合、终点异质性或慢性试验设计差异。

NOEC 中 2048 位 fingerprint 最好，但相对无指纹方案只提升约 0.020 R2，说明指纹有用但不是主要瓶颈。NOEC/LOEC 的核心问题仍是阈值统计终点、实验时长和响应域异质性。

## 7. 当前建议

1. ECx 主任务可保留 512 或 2048 位 fingerprint；如果后续转深度模型，优先用 512 位或频率过滤指纹降低参数量。
2. LOEC 不建议使用 2048 位 fingerprint 作为默认输入，优先采用 512 位或无指纹描述符方案。
3. NOEC 可以暂保留 2048 位 fingerprint，但需要进一步做响应域分层和 AD 诊断。
4. 深度模型下一版应使用 `response_domain + effect_percent` 替代旧的 `endpoint_family` one-hot。
5. 论文展示上，ECx、LOEC、NOEC 应分开报告，不再给一个合并 endpoint 的主 R2。

## 8. 输出图表

每个任务和汇总目录均输出 PNG/PDF：

- `*_ABCD_指纹消融_lightgbm`
- 单模型真实-预测/残差综合诊断图
- 化学类别、终点类型、物种类群和 AD 分层残差图

图表目录：

- `outputs/models/endpoint_fingerprint_ablation_v001/图表/`
- `outputs/models/endpoint_fingerprint_ablation_v001/ecx/图表/`
- `outputs/models/endpoint_fingerprint_ablation_v001/loec/图表/`
- `outputs/models/endpoint_fingerprint_ablation_v001/noec/图表/`
