# 模型架构接口规范

版本：v0.1
日期：2026-05-24

## 1. 定位

第一篇论文主线是多物种、多终点 QSAR 框架提升新化学类别外推能力。模型必须保持结构-活性关系为核心，物种、暴露时长和显式规则层作为残差与解释增强。

第一版不做 GNN 主模型。化合物主通道使用：

```text
描述符先验分组加权模块 + Morgan fingerprint + MLP
```

预训练分子模型只作为扩展对照。

## 2. 总体公式

```text
z_desc = DescriptorGroupWeighting(descriptors)
z_fp = FingerprintMLP(morgan_fingerprint)
z_chem = ChemicalFusion(z_desc, z_fp)

z_species = SpeciesContextEncoder(taxonomy, eco_group, primary_medium, lifestage)
z_time = TimeEncoder(duration_h)
z_rule = RuleEncoder(rule_features)

y_chemical = ChemicalHead(z_chem, endpoint_family, effect_level)
y_context_residual = ContextResidualHead(z_chem, z_species, z_time)
y_rule_residual = RuleResidualHead(z_rule)

y_pred = y_chemical + alpha * y_context_residual + beta * y_rule_residual
```

约束：

- `y_chemical` 必须可单独训练和导出。
- `alpha`、`beta` 默认小，残差项有 L2 正则。
- 报告必须输出残差占比。

## 3. 模型组件接口

包路径：

```text
src/qsar_dl/models/
```

必须实现：

```python
import torch
from torch import nn


class ChemicalEncoder(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return z_chem, z_desc, z_fp and descriptor weights."""


class SpeciesContextEncoder(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode taxonomy, ecological group, primary_medium and lifestage."""


class TimeEncoder(nn.Module):
    def forward(self, duration_h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode duration with missing mask."""


class RuleEncoder(nn.Module):
    def forward(self, rule_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode explicit rule features."""


class ResidualQSARModel(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return decomposed prediction."""
```

标准输出：

```python
{
    "y_pred": Tensor,
    "y_chemical": Tensor,
    "y_context_residual": Tensor,
    "y_rule_residual": Tensor,
    "uncertainty": Tensor | None,
    "descriptor_group_weights": Tensor | None,
    "aux": dict,
}
```

## 4. Baseline 矩阵

传统模型：

- PLS
- ElasticNet
- SVR
- RandomForest
- XGBoost 或 LightGBM

深度模型：

- MLP with descriptors + fingerprint
- Descriptor-group weighted MLP
- Residual QSAR without rules
- Residual QSAR with rules

比较矩阵：

| 模型 | 标准特征 | 分组加权特征 | 物种/时长上下文 | 规则层 |
|---|---|---|---|---|
| Traditional ML standard | yes | no | optional | no |
| Traditional ML grouped | no | fixed export | optional | no |
| Deep standard | yes | no | yes | no |
| Deep grouped | no | learnable | yes | no |
| Deep grouped rules | no | learnable | yes | yes |

## 5. 迁移学习

主任务：

```text
water mg/L -> mol/L -> pTox
```

迁移对象：

```text
soil/sediment mg/kg
```

第一版迁移策略：

```text
freeze ChemicalEncoder
fine-tune task head + small context layers
```

目标：

- 水相模型学习共享结构-活性表示。
- 土壤/沉积物任务使用共享化合物编码器耦合。
- 不强行把 `mg/L` 和 `mg/kg` 混成同一标签。

## 6. 损失函数

主损失：

```text
L_task = MSE(y_pred, y_true) or Huber(y_pred, y_true)
```

残差正则：

```text
L_residual = lambda_context * ||y_context_residual||_2
           + lambda_rule * ||y_rule_residual||_2
```

描述符权重正则见 `DESCRIPTOR_GROUPING.md`。

总损失：

```text
L = L_task + L_residual + L_descriptor_regularization
```

## 7. 配置键

```yaml
model:
  architecture: residual_qsar_mlp
  chemical_encoder:
    use_descriptor_grouping: true
    use_morgan_fingerprint: true
    fingerprint_bits: 2048
    hidden_dims: [512, 256]
  context_encoder:
    use_species: true
    use_duration: true
  rule_encoder:
    enabled: true
  residual:
    alpha_init: 0.2
    beta_init: 0.1
    context_l2: 0.01
    rule_l2: 0.01
```

## 8. 验收标准

- 所有模型 forward 输出字段一致。
- `ChemicalOnly` 可独立训练。
- 关闭 species、duration、rules 后仍能运行。
- 训练日志记录 `chemical/context/rule` 三部分预测贡献。
- 土壤/沉积物微调时可冻结 `ChemicalEncoder`。
