# 显式毒理规则层接口规范

版本：v0.1
日期：2026-05-24

## 1. 定位

规则层用于把明确毒理学规律转成可计算中间量、质量标记、不确定度修正和解释文本。第一版不直接改写原始标签；只有在消融实验显示有效且方向合理后，才允许作为可学习残差项接入模型预测。

## 2. 统一接口

包路径：

```text
src/qsar_dl/rules/
```

必须实现：

```python
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass
class RuleOutput:
    features: dict[str, float | int | None]
    corrections: dict[str, float | None]
    flags: dict[str, bool | str | None]
    explanation: str


class MechanisticRule(Protocol):
    name: str
    required_inputs: list[str]

    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput:
        ...


def compute_rule_layer(batch_df, config) -> tuple["pd.DataFrame", dict]:
    """Return rule feature table and coverage report."""
```

输出：

```text
outputs/features/rule_features.parquet
outputs/reports/rule_coverage_report.json
```

## 3. 必需规则

### 3.1 水相疏水性 baseline toxicity

公式：

```text
y_base = a_group + b_group * logKow_eff
b_group > 0
logKow_eff = clip(logKow, logKow_min, logKow_max)
```

输出字段：

- `rule_aq_logkow_baseline_ptox`
- `rule_aq_logkow_applicable`
- `rule_aq_logkow_missing_inputs`

含义：对中性有机物和 baseline/narcosis 类化合物，`pTox` 通常随疏水性增加而升高。

### 3.2 水溶解度/饱和限制

公式：

```text
C_effect_mg_l = mol_l_to_mg_l(10^(-y_pred), MW)
solubility_ratio = C_effect_mg_l / water_solubility_mg_l
near_saturation_flag = I(1 <= solubility_ratio < 10)
no_effects_at_saturation_flag = I(solubility_ratio >= 10)
```

输出字段：

- `rule_solubility_ratio`
- `rule_near_saturation_flag`
- `rule_no_effects_at_saturation_flag`

含义：若预测效应浓度高于水溶解度，应提高不确定度，不能机械解释为可达到效应。

### 3.3 化学活度

公式：

```text
chemical_activity = C_free_mol_l / S_w_mol_l
activity_low_penalty = max(0, log10(0.01 / chemical_activity))
activity_high_flag = I(chemical_activity > 0.1)
```

输出字段：

- `rule_chemical_activity`
- `rule_activity_low_penalty`
- `rule_activity_high_flag`

### 3.4 MoA excess toxicity

公式：

```text
toxic_ratio = 10^(y_obs_or_pred - y_base)
excess_toxicity_flag = I(toxic_ratio > 10)
delta_moa = softplus(g_moa(MoA, ToxCast, alerts))
```

输出字段：

- `rule_toxic_ratio`
- `rule_excess_toxicity_flag`
- `rule_moa_positive_residual`

### 3.5 暴露时间不足

标准时长：

| 类群/终点 | `D_std` |
|---|---:|
| 鱼类急性 LC50 | 96 h |
| 溞类急性 EC50/LC50 | 48 h |
| 藻类/蓝藻 EC50 | 72 h |

公式：

```text
duration_ratio = duration_h / D_std
short_duration_flag = I(duration_ratio < 1)
y_std_candidate = y_obs + gamma * log10(D_std / duration_h)
gamma >= 0
```

输出字段：

- `rule_duration_ratio`
- `rule_short_duration_flag`
- `rule_duration_ptox_adjustment_candidate`

LOEC 默认不做急性时长硬校正。

### 3.6 TKTD 内暴露积累

公式：

```text
f_ss(D) = 1 - exp(-k_e * D)
k_e = softplus(k_raw)
y(D_std) = y(D_obs) + log10(f_ss(D_std) / f_ss(D_obs))
```

输出字段：

- `rule_tktd_fss_obs`
- `rule_tktd_fss_std`
- `rule_tktd_duration_adjustment_candidate`

### 3.7 挥发损失

公式：

```text
k_vol = softplus(h0 + h1 * log10(H + eps) + h2 * log10(VP + eps))
f_TWA = (1 - exp(-k_vol * duration_h)) / (k_vol * duration_h)
y_TWA = y_nominal - log10(f_TWA)
```

输出字段：

- `rule_volatility_loss_factor`
- `rule_nominal_concentration_risk_flag`
- `rule_volatile_uncertainty_penalty`

### 3.8 土壤/沉积物有机碳吸附

公式：

```text
KOC = 10^logKOC
C_free = C_total / (1 + KOC * OC)
C_free_solid_approx = (C_solid_dw / f_OC) / KOC
```

输出字段：

- `rule_koc_binding_strength`
- `rule_estimated_free_fraction`
- `rule_soil_sediment_bioavailability_penalty`

无 `f_OC` 时只输出缺失标记，不伪造自由浓度。

### 3.9 离子化/pH

弱酸：

```text
f_neutral = 1 / (1 + 10^(pH - pKa_acid))
```

弱碱：

```text
f_neutral = 1 / (1 + 10^(pKa_base - pH))
```

输出字段：

- `rule_neutral_fraction`
- `rule_ionization_flag`
- `rule_logkow_replaced_by_logd_flag`

### 3.10 分子量被动摄取限制

公式：

```text
mw_passive_penalty = clip((MW - 600) / 400, 0, 1)
passive_uptake_factor = 1 - mw_passive_penalty
```

输出字段：

- `rule_mw_passive_penalty`
- `rule_large_molecule_flag`

仅作为 AD/不确定度信号。

### 3.11 物种潜在暴露途径匹配

公式：

```text
species_route = [respiration, diet, dermal, sediment_contact, soil_ingestion]
chemical_route = softmax(W * [logKow, logKoc, logH, logS_w, MW])
route_access = dot(species_route, chemical_route)
```

输出字段：

- `rule_route_access`
- `rule_route_mismatch_flag`

物种 route 未制作时输出 missing mask，不启用。

## 4. 配置格式

默认位置：

```text
configs/rules/mechanistic_rules.yaml
```

示例：

```yaml
rules:
  aquatic_hydrophobicity:
    enabled: true
    logkow_min: -1
    logkow_max: 6
  solubility:
    enabled: true
    no_effects_ratio: 10
  duration:
    enabled: true
    gamma_grid: [0.1, 0.25, 0.5, 1.0]
    standard_hours:
      fish_lc50: 96
      daphnia_ec50: 48
      algae_ec50: 72
```

## 5. 验收标准

- 每条规则必须有正常、缺输入、不适用 3 类单元测试。
- 缺失输入不得抛异常，必须输出 `missing_inputs`。
- 所有规则输出必须包含解释文本。
- `duration` 与 `tktd` 不得同时作为最终校正项重复加成。
- 消融实验需报告 `with_rules` 和 `without_rules`。
