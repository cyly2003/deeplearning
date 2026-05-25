# 传统机器学习 Baseline 分层诊断报告

日期：2026-05-25

## 1. 本轮诊断对象

本报告基于以下输出目录：

- `outputs/models/baseline_ml_full_curated_rf_xgb_lgbm/`

该版本使用 curated 化学类别和物种标准类群字段，包含：

- `chemical_class_l1/l2/l3`
- `taxon_group_l1/l2/l3`
- `is_standard_test_species`
- `is_us_invasive_species`
- `is_us_threatened_endangered`
- endpoint、lifestage、duration、`primary_medium`
- RDKit 描述符和固定描述符分组特征

本轮未重新训练模型，只基于已有预测表和指标表，用更新后的 `src/qsar_dl/visualization/style.py` 重新导出图表。

## 2. 数据范围

| 指标 | 数量 |
|---|---:|
| 主水相任务记录数 | 226,123 |
| 化合物数 | 4,987 |
| 训练集记录数 | 185,158 |
| 验证集记录数 | 40,965 |
| 目标变量 | `target_ptox` |
| 单位族 | `water_mg_l` |

验证集 endpoint 分布：

| endpoint | 记录数 |
|---|---:|
| LC | 20,259 |
| LOEC | 13,269 |
| EC | 7,437 |

## 3. 总体 Baseline 表现

| 特征集 | 模型 | R2 | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|---:|
| standard | RandomForest | 0.392 | 1.375 | 1.053 | 27.14 |
| standard | XGBoost | 0.447 | 1.312 | 1.019 | 32.13 |
| standard | LightGBM | 0.502 | 1.246 | 0.952 | 31.02 |
| fixed_descriptor_groups | RandomForest | 0.427 | 1.335 | 1.017 | 26.65 |
| fixed_descriptor_groups | XGBoost | 0.446 | 1.314 | 1.025 | 34.38 |
| fixed_descriptor_groups | LightGBM | 0.496 | 1.253 | 0.960 | 32.42 |

当前可作为主 baseline 的模型是 `standard + LightGBM`。固定描述符分组的 LightGBM 与标准特征差距很小，说明可解释压缩特征保留了大部分信号，后续可作为深度 residual QSAR 的结构先验输入和消融对照。

## 4. Endpoint 分层诊断

以 `standard + LightGBM` 为主：

| endpoint | 样本数 | 化合物数 | R2 | RMSE | MAE | 平均残差 |
|---|---:|---:|---:|---:|---:|---:|
| EC | 7,437 | 416 | 0.541 | 1.175 | 0.922 | -0.091 |
| LC | 20,259 | 567 | 0.454 | 1.222 | 0.923 | -0.145 |
| LOEC | 13,269 | 269 | 0.391 | 1.318 | 1.013 | -0.113 |

诊断结论：

- EC 的预测效果最好，说明急性效应浓度类 endpoint 的结构-毒性信号更稳定。
- LOEC 表现最弱，符合慢性/低效应阈值数据异质性更高的预期。
- 后续论文主结果中不宜只报告合并 endpoint 指标，应至少区分 LC、EC、LOEC。

## 5. 化学类别分层诊断

以 `standard + LightGBM` 为主，主要高误差类别：

| 化学类别 | 样本数 | 化合物数 | R2 | RMSE | MAE | 平均残差 |
|---|---:|---:|---:|---:|---:|---:|
| fluorinated_organic | 1,081 | 29 | 0.247 | 1.729 | 1.351 | -0.655 |
| unclassified | 13,234 | 377 | 0.528 | 1.419 | 1.078 | -0.270 |
| pharmaceutical_pcp | 682 | 15 | 0.302 | 1.413 | 1.147 | 0.040 |

相对低误差类别：

| 化学类别 | 样本数 | 化合物数 | R2 | RMSE | MAE | 平均残差 |
|---|---:|---:|---:|---:|---:|---:|
| phenolic | 2,281 | 22 | 0.634 | 0.968 | 0.727 | 0.106 |
| metal_metalloid | 13,210 | 41 | 0.268 | 1.013 | 0.785 | -0.196 |
| hydrocarbon | 742 | 20 | 0.205 | 1.166 | 0.940 | -0.444 |

诊断结论：

- 含氟有机物误差最高，可能与 PFAS/含氟结构的特殊疏水性、离子性、蛋白结合、膜转运和传统 RDKit 描述符表达不足有关。
- `unclassified` 样本量大且误差偏高，说明化学类别整理仍会显著影响模型解释。
- 金属/类金属 RMSE 较低但 R2 不高，提示该类内部毒性范围或机制异质性仍需单独解释，不应简单并入有机物 QSAR 机制框架。

## 6. 物种类群分层诊断

以 `standard + LightGBM` 为主，主要高误差类群：

| 类群 | 样本数 | 化合物数 | R2 | RMSE | MAE | 平均残差 |
|---|---:|---:|---:|---:|---:|---:|
| amphibian | 1,321 | 96 | 0.313 | 1.494 | 1.207 | -0.333 |
| cyanobacteria | 504 | 59 | 0.265 | 1.435 | 1.128 | 0.063 |
| algae | 2,614 | 258 | 0.377 | 1.305 | 1.026 | -0.012 |

相对低误差类群：

| 类群 | 样本数 | 化合物数 | R2 | RMSE | MAE | 平均残差 |
|---|---:|---:|---:|---:|---:|---:|
| worm | 946 | 66 | 0.535 | 0.994 | 0.795 | -0.128 |
| aquatic_plant | 481 | 48 | 0.605 | 1.115 | 0.918 | 0.317 |
| insect | 2,789 | 122 | 0.502 | 1.191 | 0.954 | -0.279 |

诊断结论：

- 两栖类和蓝藻误差偏高，说明当前通用物种上下文不足以捕捉特殊生理敏感性。
- 鱼类、甲壳类样本量大，整体表现稳定，但仍需在论文图表中展示残差分布，避免总体指标掩盖类群偏差。
- 后续 residual-QSAR 深度模型应优先检验加入 taxon embedding 后是否改善两栖类、蓝藻和藻类误差。

## 7. AD 诊断

当前分层表中验证集全部被标记为 `AD内`：

| AD 分层 | 样本数 | 化合物数 | R2 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| AD内 | 40,965 | 748 | 0.502 | 1.246 | 0.952 |

诊断结论：

- 当前 AD 规则过于宽松，无法形成有效的 AD 内外对比。
- 后续需要补充更严格的化学空间 AD，例如 Morgan fingerprint Tanimoto 距离、descriptor Mahalanobis/LOF、chemical category holdout 标记和物种层级覆盖度。

## 8. 本轮重新输出图表

已使用更新后的 `style.py` 重新导出 66 个图表文件，格式为 PNG 和 PDF：

- `全量基线_数据筛选与切分流程`
- `全量基线_模型性能对比`
- 各模型综合诊断图：真实值-预测值散点、边缘分布、残差分布
- 全部模型综合诊断拼图
- endpoint、化学类别、物种类群、AD 分层真实预测残差图

输出目录：

- `outputs/models/baseline_ml_full_curated_rf_xgb_lgbm/图表/`

## 9. 下一步建议

1. 固化当前代码和文档版本，作为传统 ML baseline 里程碑。
2. 优先补强 AD 规则，使验证集能区分 AD 内外样本。
3. 对 `fluorinated_organic`、`unclassified`、`amphibian`、`cyanobacteria` 单独做误差样本核查。
4. 在深度模型前先做消融：chemical-only、chemical + endpoint/duration、chemical + species/taxon。
