# 评估协议规范

版本：v0.1
日期：2026-05-24

## 1. 论文主线

主线：多物种、多终点联合 QSAR 框架提升新化学类别外推能力。

主评估对象：

- 水相 `mg/L/pTox` 主任务。
- LC、同族 EC、LOEC 都训练，论文主展示由结果决定。
- SSD 作为应用示范，不作为核心贡献。
- 土壤/沉积物作为迁移学习扩展。

## 2. 化学类别留出

主划分按“学术常见污染物类别”留出，而不是按单个化合物或 scaffold 留出。

类别体系第一版包括：

- PAHs
- TPHs/石油烃相关
- chlorinated organics
- PFAS
- pesticides
- pharmaceuticals/personal care products
- phenols
- organophosphates
- metals/metalloids
- surfactants
- dyes
- solvents/VOCs
- other/unknown

要求：

- 每个类别必须有分类解释。
- 类别来源可来自人工规则、化学名称关键词、结构 alerts、CompTox/ECOTOX group、外部字典。
- 无法可靠归类的化合物进入 `other/unknown`，不得强行归类。

## 3. 划分接口

输出：

```text
outputs/tables/chemical_category_assignments.csv
outputs/tables/data_splits.parquet
outputs/reports/split_report.json
```

必需字段：

| 字段 | 说明 |
|---|---|
| `chemical_id` | 化合物 ID |
| `chemical_category` | 主类别 |
| `category_confidence` | 分类置信度 |
| `category_evidence` | 分类依据 |
| `split` | train/validation/test |
| `split_strategy` | chemical_category_holdout |

Python 接口：

```python
def assign_chemical_categories(chemical_table, config) -> "pd.DataFrame":
    """Assign literature-relevant pollutant classes."""


def build_category_holdout_splits(modeling_table, category_table, config) -> "pd.DataFrame":
    """Create train/validation/test splits by chemical category."""
```

## 4. 指标

回归指标：

- R2
- RMSE
- MAE
- MAPE
- Spearman rho
- calibration error for uncertainty intervals

分层指标：

- endpoint family
- effect level bins
- chemical category
- primary medium
- taxonomic class/family
- conc1_type
- duration derivation method
- in-domain vs out-of-domain

## 5. 消融实验

必须运行：

1. Traditional ML + standard features
2. Traditional ML + fixed descriptor group features
3. Deep MLP + standard features
4. Deep MLP + descriptor group weighting
5. Deep MLP + descriptor group weighting + species/duration context
6. Deep MLP + descriptor group weighting + species/duration context + rules

规则层消融：

```text
with rules vs without rules
```

描述符模块消融：

```text
raw descriptors
fixed group weights
learn intragroup
learn intra + intergroup
```

## 6. 应用域与不确定度

AD 维度：

- chemical descriptor range
- Morgan fingerprint Tanimoto
- descriptor embedding distance
- chemical category seen/unseen
- species taxonomy support
- endpoint support
- rule input completeness

输出等级：

- `in_domain`
- `near_domain`
- `out_of_domain`
- `insufficient_information`

不确定度：

- 第一版使用 ensemble 或 bootstrap ensemble。
- 输出 `pred_mean/pred_std/p05/p50/p95`。
- AD 外样本必须在报告中单独统计。

## 7. SSD 应用示范

输入：

```text
chemical_id or SMILES/CAS/DTXSID
top_n species
```

流程：

```text
predict toxicity for species panel
select top_n sensitive species
fit SSD using log-normal and log-logistic
bootstrap HC5/HC10 intervals
export figure and table
```

输出：

```text
outputs/ssd/<chemical_id>/species_predictions.parquet
outputs/ssd/<chemical_id>/ssd_fit_summary.csv
outputs/ssd/<chemical_id>/ssd_curve.png
outputs/ssd/<chemical_id>/ssd_curve.pdf
```

## 8. 图表要求

所有评估图使用 `src/qsar_dl/visualization` 样式库：

- 中文黑体。
- 英文 Arial。
- 标题字号 20，可关闭。
- 轴刻度 18。
- 文字默认加粗。
- 轴线宽度 1.5。
- 图例无框，位置可调。
- 导出 PNG、TIFF、PDF、SVG。
- DPI >= 300。

## 9. 验收标准

- 评估报告必须包含主 split 和所有消融结果。
- 必须能追溯每个测试样本的化学类别和留出原因。
- 论文主表至少包含传统模型、深度 baseline、主模型。
- 如果规则层未提升性能，则在论文中定位为解释/QC 模块。
- 如果 LC/EC 表现显著好于 LOEC，论文主展示 LC/EC，LOEC 放扩展或补充。
