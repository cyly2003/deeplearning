# 描述符先验分组加权模块规范

版本：v0.1
日期：2026-05-24

## 1. 定位

描述符先验分组加权模块是本项目的核心创新之一。它主要服务深度学习端到端训练，同时必须导出冻结权重后的组级描述符特征，供 PLS、ElasticNet、SVR、RandomForest、XGBoost/LightGBM 等传统模型做公平对照。

第一版以 RDKit 描述符为主，后续可扩展到 PaDEL 和其他描述符库。不同描述符来源使用不同分组辞典。

## 2. 输入输出

输入：

```text
outputs/features/chemical_descriptors_rdkit.parquet
configs/features/descriptor_groups_rdkit.yaml
```

输出：

```text
outputs/features/descriptor_group_features.parquet
outputs/features/descriptor_group_weights.csv
outputs/reports/descriptor_grouping_report.json
```

## 3. 分组辞典格式

配置文件：

```text
configs/features/descriptor_groups_rdkit.yaml
```

格式：

```yaml
descriptor_source: rdkit
standardization:
  method: robust_zscore
  missing_strategy: train_median_with_mask
groups:
  hydrophobicity_partition:
    description: Hydrophobicity and partition-related descriptors.
    initial_group_weight: 1.0
    bias_init: 0.0
    descriptors:
      MolLogP:
        role: core
        initial_weight: 1.0
      MolMR:
        role: auxiliary
        initial_weight: 0.5
  polarity_hbond:
    description: Polarity, H-bond donor/acceptor and polar surface features.
    initial_group_weight: 1.0
    bias_init: 0.0
    descriptors:
      TPSA:
        role: core
        initial_weight: 1.0
      NumHDonors:
        role: core
        initial_weight: 1.0
      NumHAcceptors:
        role: core
        initial_weight: 1.0
```

建议大类：

- `hydrophobicity_partition`
- `polarity_hbond`
- `electronic_reactivity`
- `topology_connectivity`
- `size_shape`
- `ring_aromaticity`
- `flexibility_saturation`
- `charge_ionization_proxy`
- `fragment_alerts`

## 4. 模块公式

单个标准化描述符：

```text
x_j = standardized descriptor j
```

组内加权：

```text
g_k = phi( sum_{j in G_k} softmax(a_kj) * x_j + b_k )
```

组间加权：

```text
z_desc = concat(g_1, g_2, ..., g_K)
z_desc_global = sum_k softmax(A_k) * g_k
```

其中：

- `G_k`：第 k 个描述符组。
- `a_kj`：组内可学习权重。
- `b_k`：组偏置。
- `A_k`：组间可学习权重。
- `phi`：默认 `GELU`，可配置为 `tanh` 或 identity。

## 5. 训练阶段

必须支持四种模式：

| 模式 | 含义 | 用途 |
|---|---|---|
| `raw_descriptors` | 不使用分组 | 标准 baseline |
| `fixed_group_weights` | 使用先验分组和固定权重 | 证明人工知识是否有效 |
| `learn_intragroup` | 学习组内权重，固定组间权重 | 检查组内描述符贡献 |
| `learn_intra_and_intergroup` | 学习组内和组间权重 | 主深度模型 |

推荐训练顺序：

```text
1. 标准特征 baseline
2. 冻结分组权重
3. 解冻组内权重
4. 解冻组内 + 组间权重
```

正则：

```text
L = L_task + lambda_l1 * ||weights||_1 + lambda_entropy * entropy_penalty + lambda_group * group_sparsity
```

目的：

- 防止权重无约束漂移。
- 保持分组解释性。
- 允许模型发现少数真正重要的描述符组。

## 6. Python 接口

包路径：

```text
src/qsar_dl/features/descriptor_groups.py
src/qsar_dl/models/descriptor_weighting.py
```

必须实现：

```python
from pathlib import Path

import pandas as pd
import torch
from torch import nn


def load_descriptor_group_dictionary(path: Path) -> dict:
    """Load and validate descriptor grouping YAML."""


def validate_descriptor_coverage(descriptor_columns: list[str], group_dict: dict) -> pd.DataFrame:
    """Return coverage table: grouped, ungrouped, missing descriptors."""


def build_fixed_group_features(descriptor_df: pd.DataFrame, group_dict: dict) -> pd.DataFrame:
    """Export frozen-weight group-level features for traditional ML baselines."""


class DescriptorGroupWeighting(nn.Module):
    def __init__(self, group_dict: dict, mode: str):
        ...

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return group embeddings, global descriptor embedding, weights."""
```

`forward` 输出：

```python
{
    "group_embeddings": Tensor,
    "global_embedding": Tensor,
    "intragroup_weights": Tensor,
    "intergroup_weights": Tensor,
}
```

## 7. 传统模型对照接口

固定权重组级特征表必须包含：

| 字段 | 说明 |
|---|---|
| `chemical_id` | 化合物 ID |
| `desc_group_<group_name>` | 组级加权描述符 |
| `desc_group_<group_name>_missing_rate` | 该组缺失比例 |
| `desc_group_<group_name>_coverage` | 该组有效描述符数量 |

传统模型比较至少包含：

- PLS
- ElasticNet
- SVR
- RandomForest
- XGBoost 或 LightGBM

每个传统模型需分别运行：

```text
standard descriptors/fingerprints
fixed descriptor group features
```

## 8. 验收标准

- 分组辞典中每个描述符必须能检查是否存在于当前描述符表。
- 未分组描述符必须进入 report，不得静默丢弃。
- 固定权重组级特征可被传统 ML 直接读取。
- 深度模块必须能导出组内和组间权重。
- 消融报告必须比较四种模式性能。
