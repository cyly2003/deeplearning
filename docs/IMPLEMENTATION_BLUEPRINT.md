# 多物种多终点深度学习 QSAR 项目施工图

版本：v0.1
日期：2026-05-24
仓库远端：`git@github.com:cyly2003/deeplearning.git`
默认解释器：`E:\TOOLS\anaconda\python.exe`
常用环境：`E:\TOOLS\anaconda\envs\qsar-ph3`

## 1. 项目目标与硬性边界

本项目建立一套面向 ECOTOX 全量数据的多物种、多毒性终点深度学习 QSAR 框架。第一阶段的核心目标是建立可并行开发的数据契约、特征模块、显式毒理规则层、模型训练模块、预测服务模块、SSD 风险模块和科研图表样式库。

正式接口规范拆分在以下文档中，后续开发以这些文件为准：

- `docs/DATA_CONTRACT.md`
- `docs/CONFIG_SPEC.md`
- `docs/DESCRIPTOR_GROUPING.md`
- `docs/RULE_LAYER.md`
- `docs/MODEL_ARCHITECTURE.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/MODULE_TASKS.md`

两个最终应用入口：

1. 毒性预测：用户输入物种学名或物种 ID，以及 SMILES、CASRN、DTXSID 等化合物一对一标识，模型输出毒性预测值、不确定度、应用域、规则解释和相似训练样本。
2. 风险预测：用户输入化合物，模型输出该化合物在不同物种上的预测毒性，自动选择前 `n` 个敏感物种拟合 SSD 曲线，并输出 HC5、置信区间和拟合诊断。

模型设计边界：

- 核心必须保持 QSAR 属性，即主预测信号来自化合物结构、描述符、指纹和 MoA 信息。
- 物种、lifestage、生态类群、生存介质、潜在暴露途径和暴露时长只能作为上下文残差、显式规则中间量和外推能力增强项。
- 不允许把所有物种与暴露 one-hot 直接拼接到主干网络并作为唯一预测逻辑。
- 不引入 `exposure_medium` 字段，因为当前数据没有真实实验暴露介质。
- 生存介质仅作为物种生态属性，字段名固定为 `habitat_medium`。
- LC、同族 EC、LOEC 是第一阶段优先终点；终点数字如 50、10、20 拆成 `effect_level` 参与建模。
- R2 >= 0.6 视为阶段性合格，但必须同时报告分层指标、应用域和过拟合风险。

## 2. 推荐仓库结构

```text
deeplearning/
├── README.md
├── .gitignore
├── docs/
│   ├── IMPLEMENTATION_BLUEPRINT.md
│   ├── DATA_CONTRACT.md
│   ├── RULE_LAYER.md
│   └── SERVER_RUNBOOK.md
├── configs/
│   ├── data/
│   │   └── ecotox.yaml
│   ├── features/
│   │   ├── chemical.yaml
│   │   ├── species.yaml
│   │   └── endpoint.yaml
│   ├── rules/
│   │   └── mechanistic_rules.yaml
│   ├── experiments/
│   │   ├── baseline_chemical_only.yaml
│   │   ├── multitask_residual.yaml
│   │   └── transfer_soil_finetune.yaml
│   └── runtime/
│       ├── local.yaml
│       └── gpu_server.yaml
├── src/
│   └── qsar_dl/
│       ├── data/
│       ├── features/
│       ├── rules/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── applicability_domain/
│       ├── uncertainty/
│       ├── ssd/
│       ├── visualization/
│       └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/
├── scripts/
├── data/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

`data/raw/`、`data/interim/`、`outputs/`、`models/`、`checkpoints/` 不进入 Git。只提交代码、配置、小型测试样例、数据字典和可复现实验说明。

## 3. 标准数据契约

所有模块以标准长表为中心，不直接依赖 ECOTOX 原始字段名。ECOTOX 清洗模块负责把原始表转换成以下契约。

### 3.1 核心毒性长表

文件建议：`outputs/tables/clean_toxicity_long.parquet`

必需字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `record_id` | string | 清洗后唯一记录 ID |
| `source_database` | string | 数据来源，例如 `ECOTOX` |
| `source_row_id` | string | 原始数据行或记录 ID |
| `chemical_id` | string | 项目内部化合物 ID |
| `casrn` | string/null | CAS 号 |
| `dtxsid` | string/null | CompTox DTXSID |
| `smiles` | string/null | 规范化 SMILES |
| `inchikey` | string/null | InChIKey |
| `species_id` | string | 项目内部物种 ID |
| `scientific_name` | string | 物种学名 |
| `endpoint_family` | category | `LC`、`EC`、`LOEC` |
| `effect_level` | float/null | LC50 中的 50、EC10 中的 10 等 |
| `effect_name_raw` | string | 原始效应字段 |
| `duration_h` | float/null | 暴露时长，统一小时 |
| `duration_source` | string | 时长来源或解析方式 |
| `habitat_medium` | category/null | 生存介质，来自物种生态属性，不是实验暴露介质 |
| `lifestage` | category/null | life stage 标准化类别 |
| `value_original` | float | 原始毒性值 |
| `unit_original` | string | 原始单位 |
| `value_mol_l` | float/null | 统一到 mol/L 后的值；非水相或不可换算时允许空 |
| `target_ptox` | float/null | `-log10(value_mol_l)` |
| `censor_flag` | category | `exact`、`greater_than`、`less_than`、`range`、`unknown` |
| `qa_flags` | list/string | 清洗、单位、重复、异常值、规则标记 |

禁止事项：

- 不得在未记录原因的情况下删除记录。
- 不得把缺失 lifestage 当成 adult。
- 不得把 `habitat_medium` 当成实验暴露介质。
- 不得在原始单位未确认时计算 `target_ptox`。

### 3.2 化合物特征表

文件建议：`outputs/features/chemical_features.parquet`

必需字段：

- `chemical_id`
- `casrn`
- `dtxsid`
- `smiles`
- `inchikey`
- `molecular_weight_g_mol`
- `logkow`
- `logd_ph7`
- `water_solubility_mg_l`
- `henry_law_atm_m3_mol`
- `vapor_pressure_pa`
- `logkoc`
- `pka_acid`
- `pka_base`
- `rdkit_descriptor_*`
- `fingerprint_*`
- `toxcast_moa_*`
- `feature_source_flags`

若物化性质缺失，保留缺失标记，不用均值静默填补。填补只能在建模 pipeline 内完成，并写入 preprocessing report。

### 3.3 物种特征表

文件建议：`outputs/features/species_features.parquet`

必需字段：

- `species_id`
- `scientific_name`
- `taxonomy_kingdom`
- `taxonomy_phylum`
- `taxonomy_class`
- `taxonomy_order`
- `taxonomy_family`
- `taxonomy_genus`
- `eco_group`
- `habitat_medium`
- `potential_exposure_route_*`
- `lifestage_*`
- `taxonomy_source`
- `species_feature_status`

第一阶段物种编码只包含分类学路径、生态类群、可能暴露途径和 lifestage。潜在暴露途径与 lifestage 尚未制作时，模块必须支持 `unknown` 与 mask。

## 4. 总体架构公式

主模型采用结构优先残差式条件 QSAR：

```text
z_c = f_chem(G_c, descriptors_c, fingerprints_c, MoA_c)
z_s = f_species(taxonomy_s, eco_group_s, habitat_s, route_s, lifestage_s)
z_t = f_time(log_duration_h)

m = f_bridge(z_c, z_s, z_t, physicochem_c)

y_hat = f_QSAR(z_c, endpoint_family, effect_level)
        + alpha * f_context_residual(z_c, z_s, z_t, m)
        + beta  * f_rule(m)
```

符号约定：

- `y_hat`：预测 `pTox = -log10(C_mol/L)`。
- `f_QSAR`：化合物结构主模型，必须可单独训练和评估。
- `f_context_residual`：物种、lifestage、生态属性和时长带来的残差修正。
- `f_rule`：显式毒理规则层输出的修正项或可靠性权重。
- `alpha`：上下文残差强度，默认小于 1，并加入 L2 正则。
- `beta`：规则层强度，默认从 0 或很小值开始，经过消融实验确认后放开。

验收标准：

- 必须能运行 chemical-only baseline。
- 完整模型相对 chemical-only 的提升必须拆解到物种、时长、规则层。
- 若完整模型主要靠物种编码提升，而 chemical-only 完全失效，则报告中必须标记为“生态上下文预测模型”，不能声称强 QSAR。

## 5. 显式毒理规则层

规则层输出三个对象：

```text
rule_features: 可进入模型的连续中间量
rule_corrections: 可选的 pTox 修正项
rule_flags: 解释、质量控制和不确定度标记
```

默认实现策略：

- 第一版只把规则作为中间特征、可靠性权重和解释项。
- 不直接改写原始标签。
- 只有在消融实验证明有效且方向合理后，才允许作为 `f_rule` 的可学习修正项。
- 所有规则都必须输出 `applicable`、`missing_inputs`、`direction`、`explanation`。

### 5.1 水相疏水性 baseline toxicity 规则

适用条件：

- endpoint 为 LC 或同族 EC；
- `habitat_medium` 或测试对象主要为水生/水相暴露相关；
- 化合物为中性有机物或非特异性 narcosis 类；
- `logkow` 可用；
- 不适合直接用于强反应性、金属、无机盐、表面活性剂和特异性作用模式化合物。

公式：

```text
y_base = a_group + b_group * logKow_eff
b_group > 0
logKow_eff = clip(logKow, logKow_min, logKow_max)
```

含义：

- `y_base` 是 baseline/narcosis 机制下的结构-活性基线毒性。
- 当 `pTox = -log10(C_mol/L)` 时，`b_group > 0` 表示 logKow 增大，效应浓度降低，毒性增强。
- `a_group` 和 `b_group` 可按 endpoint family、物种大类或 MoA 类别学习。

默认输出：

- `rule_aq_logkow_baseline_ptox`
- `rule_aq_logkow_slope_positive = true`
- `rule_aq_logkow_applicable`
- `rule_aq_logkow_missing_inputs`

工程要求：

- 不把该规则当作全局硬线性关系。
- 对高 logKow 区间必须与溶解度/化学活度规则联动。
- 模型报告中必须绘制 `logKow` vs residual 或 PDP，检查方向是否合理。

### 5.2 水溶解度与 No Effects at Saturation 规则

适用条件：

- 水相 LC/EC；
- `water_solubility_mg_l` 可用；
- 预测或观测效应浓度可转为 mg/L 或 mol/L。

公式：

```text
C_effect_mg_l = mol_l_to_mg_l(10^(-y_pred), MW)
S_w = water_solubility_mg_l

solubility_ratio = C_effect_mg_l / S_w
nes_flag = I(solubility_ratio >= 10)
near_saturation_flag = I(1 <= solubility_ratio < 10)
r_solubility = max(0, log10(solubility_ratio))
```

含义：

- 若效应浓度高于水溶解度，尤其高出 10 倍以上，水相中可能无法达到该效应水平。
- 这类记录或预测应标记为低可靠性或 No Effects at Saturation 风险，而不是机械输出一个看似精确的 LC/EC。

默认输出：

- `rule_solubility_ratio`
- `rule_near_saturation_flag`
- `rule_no_effects_at_saturation_flag`
- `rule_solubility_reliability_penalty`

工程要求：

- 该规则优先作为预测可靠性和 AD 解释，不默认修改标签。
- 若原始记录本身 `value_original > S_w`，加入 `qa_flags`。

### 5.3 化学活度规则

适用条件：

- 疏水有机物；
- `water_solubility_mg_l` 和自由溶解浓度估计可用；
- 水相或可估计自由浓度的介质。

公式：

```text
C_free_mol_l = estimated_free_concentration
S_w_mol_l = water_solubility_mol_l
chemical_activity = C_free_mol_l / S_w_mol_l

r_activity_low  = max(0, log10(a_low / chemical_activity))
r_activity_high = max(0, log10(chemical_activity / a_high))
```

默认参数：

```text
a_low = 0.01
a_high = 0.1
```

含义：

- baseline toxicity 常可用化学活度解释。
- 若饱和状态下仍达不到关键化学活度阈值，水相效应可能受限。
- 若化学活度过高，说明预测可能接近溶解度/饱和边界，需要提高不确定度。

默认输出：

- `rule_chemical_activity`
- `rule_activity_low_penalty`
- `rule_activity_high_flag`

工程要求：

- 没有自由浓度时，不得用总浓度冒充自由浓度，必须输出缺失标记。

### 5.4 MoA excess toxicity 规则

适用条件：

- baseline toxicity 可估计；
- MoA、ToxCast assay、Verhaar/EnviroTox 类别或类似机制标签可用。

公式：

```text
toxic_ratio = 10^(y_obs_or_pred - y_base)
delta_moa = softplus(g_moa(MoA, ToxCast, structure_alerts))
y_pred = y_base + delta_moa
```

含义：

- 对特异性作用、反应性化合物、极性麻醉型化合物，毒性可能高于 baseline narcosis。
- `toxic_ratio > 10` 可标记为明显 excess toxicity。

默认输出：

- `rule_toxic_ratio`
- `rule_excess_toxicity_flag`
- `rule_moa_positive_residual`

工程要求：

- `delta_moa` 默认非负，防止 MoA 层把 baseline 毒性任意拉低。
- MoA 信息缺失时，不能填成“无 MoA”，应填 `unknown` + mask。

### 5.5 暴露时间不足规则

适用条件：

- LC 或同族 EC；
- `duration_h` 可用；
- endpoint 与类群有标准急性测试时长。

标准时长：

| 类群/终点 | `D_std` |
|---|---:|
| 鱼类急性 LC50 | 96 h |
| 溞类急性 EC50/LC50 | 48 h |
| 藻类/蓝藻生长抑制 EC50 | 72 h |
| 类群不明确急性水生终点 | 使用训练集中同 endpoint/taxon 的中位标准时长，并加 ambiguous flag |

LOEC 不设统一急性 `D_std`，因为 LOEC 常对应慢性或亚慢性设计。

公式：

```text
duration_ratio = duration_h / D_std
short_duration_flag = I(duration_ratio < 1)

y_std = y_obs + gamma * log10(D_std / duration_h), if duration_h < D_std
gamma >= 0
```

含义：

- 当暴露时间短于标准时，内暴露或毒效应可能尚未充分发展。
- 同一化合物和物种下，短时 LC/EC 往往需要更高外部浓度才能达到效应，因此 `pTox` 偏低。
- 公式把短时观测值转换到标准时长方向，但第一版只作为规则特征和偏差解释。

默认输出：

- `rule_duration_ratio`
- `rule_short_duration_flag`
- `rule_duration_ptox_adjustment_candidate`

工程要求：

- `gamma` 不手工固定，先由验证集学习或在 `{0.1, 0.25, 0.5, 1.0}` 网格中选择。
- 对 LOEC 默认只输出 `duration_h`、`log_duration_h` 和 `duration_bin`，不做急性校正。

### 5.6 TKTD 内暴露积累规则

适用条件：

- LC/EC；
- `duration_h` 可用；
- 需要用连续函数表达暴露时间影响。

公式：

```text
f_ss(D) = 1 - exp(-k_e * D)
k_e = softplus(k_raw)

y(D) = y_ss + log10(f_ss(D))
y(D_std) = y(D_obs) + log10(f_ss(D_std) / f_ss(D_obs))
```

含义：

- `f_ss(D)` 表示暴露时长 D 下接近稳态内暴露的比例。
- 短暴露时 `f_ss(D) < 1`，所以 `y(D)` 小于稳态毒性。
- `k_e` 可按化合物疏水性、分子量、物种和 lifestage 学习。

默认输出：

- `rule_tktd_fss_obs`
- `rule_tktd_fss_std`
- `rule_tktd_duration_adjustment_candidate`

工程要求：

- 该规则是软约束，不作为所有记录的硬校正。
- 与 5.5 只能选择一个作为最终修正项，不能重复加成；但两者都可作为候选特征。

### 5.7 挥发损失规则

适用条件：

- `henry_law_atm_m3_mol` 或 `vapor_pressure_pa` 可用；
- 化合物具有挥发性；
- 原始浓度为 nominal 或未说明 measured。

公式：

```text
k_vol = softplus(h0 + h1 * log10(H + eps) + h2 * log10(VP + eps))
f_TWA = (1 - exp(-k_vol * duration_h)) / (k_vol * duration_h)
C_TWA = C_nominal * f_TWA
y_TWA = y_nominal - log10(f_TWA)
```

含义：

- 挥发性化合物在测试期间真实时间加权浓度可能低于名义浓度。
- 使用 nominal concentration 计算的毒性可能被低估。
- 若有 measured concentration，优先使用 measured concentration，规则仅用于解释。

默认输出：

- `rule_volatility_loss_factor`
- `rule_nominal_concentration_risk_flag`
- `rule_volatile_uncertainty_penalty`

工程要求：

- 没有浓度测定状态字段时，该规则只增加不确定度，不直接修正标签。

### 5.8 沉积物/土壤有机碳吸附与自由浓度规则

适用条件：

- 沉积物、土壤或与颗粒/有机碳显著相关的生境；
- `logkoc`、`logkow` 或可估计 KOC；
- 有 `f_OC` 时优先使用实测有机碳比例。

公式：

```text
KOC = 10^logKOC

water_or_porewater:
C_free = C_total / (1 + KOC * OC)

sediment_or_soil_approx:
C_OC = C_solid_dw / f_OC
C_free ≈ C_OC / KOC
```

含义：

- 疏水有机物随 KOC 和有机碳含量增加，自由溶解浓度下降。
- 总浓度相同并不代表生物有效浓度相同。
- 该规则对后续土壤数据微调很重要。

默认输出：

- `rule_koc_binding_strength`
- `rule_estimated_free_fraction`
- `rule_soil_sediment_bioavailability_penalty`

工程要求：

- 没有 `f_OC` 时只输出缺失标记和高不确定度，不得伪造自由浓度。
- 该规则不用于第一阶段非土壤主训练的硬校正，只作为可解释变量。

### 5.9 离子化与 pH 规则

适用条件：

- 可获得 pKa 或 logD；
- 化合物为可离子化有机物；
- pH 可用或可使用默认 pH 场景。

公式：

弱酸：

```text
f_neutral = 1 / (1 + 10^(pH - pKa_acid))
```

弱碱：

```text
f_neutral = 1 / (1 + 10^(pKa_base - pH))
```

若 logD 可用：

```text
logKow_eff = logD_pH
```

含义：

- 中性分子通常更易被动跨膜。
- 离子态也可能通过特定转运或局部环境贡献毒性，因此该规则只能软修正。

默认输出：

- `rule_neutral_fraction`
- `rule_ionization_flag`
- `rule_logkow_replaced_by_logd_flag`

工程要求：

- 没有 pH 时默认用 pH 7 生成情景特征，并标记 `ph_assumed_flag`。
- 不允许把离子态化合物简单判定为无毒。

### 5.10 分子量与被动摄取限制规则

适用条件：

- 水生急性 LC/EC；
- 被动跨膜摄取是主要假设；
- `molecular_weight_g_mol` 可用。

公式：

```text
mw_passive_penalty = clip((MW - 600) / 400, 0, 1)
passive_uptake_factor = 1 - mw_passive_penalty
```

含义：

- 高分子量化合物被动跨膜吸收可能降低。
- 该规则对聚合物、表面活性剂、离子型化合物和特殊 MoA 不一定适用。

默认输出：

- `rule_mw_passive_penalty`
- `rule_large_molecule_flag`

工程要求：

- 只作为 AD 和不确定度信号，不能单独把毒性调低。

### 5.11 物种潜在暴露途径与化合物分配匹配规则

适用条件：

- 物种潜在暴露途径制作完成；
- 化合物有 logKow、Henry、logKoc、溶解度等分配参数。

公式：

```text
species_route = [respiration, diet, dermal, sediment_contact, soil_ingestion]
chemical_route = softmax(W * [logKow, logKoc, logH, logS_w, MW])

route_access = dot(species_route, chemical_route)
```

含义：

- 该规则是物种-化合物中间层，不是原始 one-hot 堆叠。
- 例如高挥发/高 Henry 化合物在纯水相呼吸暴露中可能难以维持；高 KOC 化合物在沉积物或土壤接触/摄食路径中可能更相关。

默认输出：

- `rule_route_access`
- `rule_route_mismatch_flag`

工程要求：

- 物种 route 未制作时输出缺失 mask，不启用该规则。

## 6. 模块施工图

以下每个模块都必须能独立交付给子代理开发。任何模块不得绕过标准数据契约直接读取其他模块内部临时文件。

### 6.1 Git 与项目骨架模块

交付目标：

- 初始化 Git 仓库并绑定 SSH 远端。
- 建立基础目录、`.gitignore`、`README.md`、文档入口和空目录占位。

输入：

- 本地项目根目录：`C:\Users\Lenovo\Documents\深度学习QSAR`
- 远端：`git@github.com:cyly2003/deeplearning.git`

输出：

- 可 push 的 `main` 分支。
- 首次提交信息：`Initialize QSAR deep learning project blueprint`

验收：

```powershell
git remote -v
git status --short --branch
git log --oneline -1
```

远端必须显示 SSH 地址。

### 6.2 数据摄取与清洗模块

包路径：`src/qsar_dl/data/`

交付目标：

- 将 ECOTOX 原始表转换为标准长表。
- 所有删除、单位换算、重复合并、异常值处理均有审计记录。

核心文件：

- `loaders.py`
- `schema.py`
- `clean_ecotox.py`
- `unit_conversion.py`
- `endpoint_parser.py`
- `audit.py`

必须实现的接口：

```python
def load_raw_ecotox(config_path: Path) -> dict[str, pd.DataFrame]: ...
def parse_endpoint(raw_endpoint: str, raw_effect: str) -> EndpointRecord: ...
def standardize_duration(value, unit) -> float | None: ...
def convert_to_molar(value, unit, molecular_weight_g_mol) -> float | None: ...
def build_clean_toxicity_long(raw_tables, chemical_map, species_map, config) -> pd.DataFrame: ...
def write_audit_report(audit_events, output_path: Path) -> None: ...
```

关键规则：

- endpoint family 只接受 `LC`、`EC`、`LOEC`，其他终点进入候选表，不进入第一版训练。
- LC50/EC50/EC10/LOEC 等必须拆分为 `endpoint_family` 和 `effect_level`。
- `duration_h` 统一为小时。
- 单位无法可靠换算到 mol/L 时，`target_ptox` 为空，记录进入待处理清单。
- censored 数据第一版默认不训练，单独输出；后续可做 Tobit/区间损失。

输出：

- `outputs/tables/clean_toxicity_long.parquet`
- `outputs/tables/excluded_records.csv`
- `outputs/reports/data_cleaning_audit.json`

测试：

- endpoint parser 单元测试。
- 单位换算测试。
- 重复记录合并测试。
- 缺失 MW 导致无法换算的测试。

### 6.3 化合物特征模块

包路径：`src/qsar_dl/features/chemical.py`

交付目标：

- 从 SMILES/CASRN/DTXSID 生成化合物主表征。
- 集成 RDKit 描述符、分子指纹、ToxCast/MoA、多种物化性质。

必须实现的接口：

```python
def normalize_smiles(smiles: str) -> str | None: ...
def compute_rdkit_descriptors(smiles: str) -> dict[str, float]: ...
def compute_fingerprints(smiles: str, radius: int, n_bits: int) -> np.ndarray: ...
def load_toxcast_moa(path: Path) -> pd.DataFrame: ...
def build_chemical_features(chemical_table: pd.DataFrame, config) -> pd.DataFrame: ...
```

关键规则：

- SMILES 标准化失败不得删除原记录，只记录 `structure_lookup_status`。
- RDKit 失败的描述符保留缺失和错误标记。
- ToxCast/MoA 缺失编码为 unknown，不等于无活性。
- 描述符聚类作为后续解释和消融模块，不作为第一版强制降维。

输出：

- `outputs/features/chemical_features.parquet`
- `outputs/reports/chemical_feature_report.json`

测试：

- 已知 SMILES 的 MW、logP、指纹长度测试。
- 无效 SMILES 的错误处理测试。
- ToxCast 缺失值 mask 测试。

### 6.4 物种特征模块

包路径：`src/qsar_dl/features/species.py`

交付目标：

- 生成分类学路径、生态类群、habitat medium、潜在暴露途径和 lifestage 编码。

必须实现的接口：

```python
def normalize_species_name(name: str) -> str: ...
def build_taxonomy_features(species_table: pd.DataFrame, taxonomy_ref: pd.DataFrame) -> pd.DataFrame: ...
def encode_habitat_medium(value: str | None) -> dict[str, int]: ...
def encode_lifestage(value: str | None) -> dict[str, int]: ...
def build_species_features(species_table: pd.DataFrame, config) -> pd.DataFrame: ...
```

关键规则：

- `habitat_medium` 是物种生态属性，不是暴露条件。
- lifestage 缺失必须是 `unknown` + mask。
- 潜在暴露途径未制作完成前，所有 route 特征输出 unknown/missing，不启用 route_access 规则。

输出：

- `outputs/features/species_features.parquet`
- `outputs/reports/species_feature_report.json`

测试：

- 分类学层级缺失测试。
- habitat multi-hot 测试。
- lifestage unknown 测试。

### 6.5 规则层模块

包路径：`src/qsar_dl/rules/`

交付目标：

- 实现第 5 节所有显式规则的可计算版本。
- 规则层输出可拼接到模型、可写入预测解释、可用于 AD 与不确定度。

核心文件：

- `base.py`
- `aquatic_hydrophobicity.py`
- `solubility.py`
- `chemical_activity.py`
- `duration.py`
- `tktd.py`
- `volatility.py`
- `bioavailability.py`
- `ionization.py`
- `route_access.py`
- `registry.py`

统一接口：

```python
@dataclass
class RuleOutput:
    features: dict[str, float | int | None]
    corrections: dict[str, float | None]
    flags: dict[str, bool | str | None]
    explanation: str

class MechanisticRule(Protocol):
    name: str
    required_inputs: list[str]
    def compute(self, row: Mapping[str, Any], config: Mapping[str, Any]) -> RuleOutput: ...
```

规则注册：

```python
def get_rule_registry(config) -> list[MechanisticRule]: ...
def compute_rule_layer(batch_df: pd.DataFrame, config) -> pd.DataFrame: ...
```

输出：

- `outputs/features/rule_features.parquet`
- `outputs/reports/rule_coverage_report.json`

测试：

- 每条规则至少 3 个 fixture：正常、缺关键输入、不适用。
- 检查方向约束：duration 短时修正不得降低 `pTox`；baseline logKow 斜率不得为负。
- 检查缺失输入不会抛异常。

### 6.6 模型模块

包路径：`src/qsar_dl/models/`

交付目标：

- 实现 chemical-only baseline、结构优先残差模型、多任务 endpoint 模型和迁移学习入口。

核心模型：

1. `ChemicalOnlyModel`
2. `ContextResidualModel`
3. `RuleAwareResidualModel`
4. `SoilFineTuneModel`

必须实现的接口：

```python
class QSARDataset(torch.utils.data.Dataset): ...
class ChemicalEncoder(nn.Module): ...
class SpeciesContextEncoder(nn.Module): ...
class TimeEncoder(nn.Module): ...
class RuleEncoder(nn.Module): ...
class RuleAwareResidualModel(nn.Module):
    def forward(self, batch) -> dict[str, torch.Tensor]: ...
```

模型输出：

```python
{
    "y_pred": Tensor,
    "y_chemical": Tensor,
    "y_context_residual": Tensor,
    "y_rule_residual": Tensor,
    "uncertainty": Tensor | None,
    "aux": dict
}
```

关键规则：

- `y_chemical` 必须可单独导出。
- `y_context_residual` 和 `y_rule_residual` 必须有正则项。
- 模型训练日志必须报告残差占比。
- 不允许上下文残差长期主导预测。

测试：

- forward shape 测试。
- chemical-only 与 full model 输入兼容测试。
- 缺失 species/rule mask 测试。

### 6.7 训练与实验模块

包路径：`src/qsar_dl/training/`

交付目标：

- 支持本地和远程服务器训练。
- 支持随机划分、化合物 scaffold split、物种 split、时间/来源 split。

必须实现的接口：

```python
def build_dataloaders(config) -> dict[str, DataLoader]: ...
def train_one_epoch(model, loader, optimizer, loss_fn, device) -> dict: ...
def evaluate(model, loader, metrics, device) -> dict: ...
def run_experiment(config_path: Path) -> Path: ...
```

默认训练方案：

1. chemical-only baseline。
2. chemical + endpoint/effect。
3. chemical + species context。
4. chemical + species + duration。
5. full rule-aware residual。
6. 非土壤预训练。
7. 土壤数据微调。

指标：

- R2
- RMSE
- MAE
- MAPE
- endpoint 分层指标
- habitat 分层指标
- taxon 分层指标
- chemical scaffold 分层指标

输出：

- `outputs/models/<experiment_id>/model.pt`
- `outputs/models/<experiment_id>/config_resolved.yaml`
- `outputs/models/<experiment_id>/metrics.json`
- `outputs/models/<experiment_id>/predictions.parquet`
- `outputs/models/<experiment_id>/training_log.csv`

测试：

- 小样本 smoke test 必须 2 分钟内跑通。
- 相同 seed 输出一致。
- 缺失 GPU 时自动回退 CPU。

### 6.8 应用域模块

包路径：`src/qsar_dl/applicability_domain/`

交付目标：

- 输出化学、物种、终点、规则四类应用域信息。

AD 维度：

| 维度 | 指标 |
|---|---|
| 化学 AD | descriptor range、fingerprint Tanimoto、embedding distance、scaffold 是否见过 |
| 物种 AD | taxonomic distance、同属/同科支持度、lifestage 支持度 |
| 终点 AD | endpoint family/effect level 是否在训练分布内 |
| 规则 AD | 规则输入完整度、溶解度/挥发/高 KOC 风险标记 |

必须实现的接口：

```python
def compute_chemical_ad(query, train_features, config) -> dict: ...
def compute_species_ad(query, train_species, config) -> dict: ...
def compute_rule_ad(rule_outputs) -> dict: ...
def summarize_ad(ad_parts: dict) -> dict: ...
```

输出等级：

- `in_domain`
- `near_domain`
- `out_of_domain`
- `insufficient_information`

测试：

- 已见化合物应为 in-domain。
- 随机无关化合物应触发 chemical out-of-domain。
- 未见物种但同属物种存在时应为 near-domain。

### 6.9 不确定度模块

包路径：`src/qsar_dl/uncertainty/`

交付目标：

- 为单点预测和 SSD 输入提供不确定度。

第一版方法：

- deep ensemble 或 bootstrap ensemble。
- 可选 MC dropout。
- conformal prediction 作为后续扩展。

必须实现的接口：

```python
def ensemble_predict(models, batch) -> pd.DataFrame: ...
def summarize_uncertainty(predictions: pd.DataFrame) -> pd.DataFrame: ...
def calibrate_intervals(validation_predictions: pd.DataFrame, config) -> dict: ...
```

输出：

- `pred_mean`
- `pred_std`
- `pred_p05`
- `pred_p50`
- `pred_p95`
- `uncertainty_source`

测试：

- ensemble size 为 1 时仍可输出。
- out-of-domain 样本不确定度应增加或至少被 AD 标记。

### 6.10 预测 CLI 与服务模块

包路径：`src/qsar_dl/cli/`

交付目标：

- 提供可被脚本、服务器和 Notebook 调用的预测入口。

命令：

```powershell
python -m qsar_dl.cli prepare-data --config configs/data/ecotox.yaml
python -m qsar_dl.cli train --config configs/experiments/multitask_residual.yaml
python -m qsar_dl.cli predict-toxicity --species "Danio rerio" --chemical "50-00-0"
python -m qsar_dl.cli predict-risk --chemical "50-00-0" --top-n 20
python -m qsar_dl.cli make-figures --style thesis_cn
```

`predict-toxicity` 输出：

- 输入标准化结果。
- `pred_ptox`
- 原单位反算结果。
- 95% 预测区间。
- AD 等级。
- 规则解释。
- 最近邻训练样本。

`predict-risk` 输出：

- 多物种预测表。
- top-n 敏感物种。
- SSD 拟合图。
- HC5、HC10、置信区间。
- 不确定度传播说明。

测试：

- CLI help 正常。
- fixture 化合物和物种可完成预测。
- 缺失化合物结构时输出明确错误，不崩溃。

### 6.11 SSD 风险模块

包路径：`src/qsar_dl/ssd/`

交付目标：

- 基于多物种预测毒性自动拟合 SSD 曲线。

必须实现的接口：

```python
def select_sensitive_species(predictions: pd.DataFrame, top_n: int) -> pd.DataFrame: ...
def fit_ssd(values: np.ndarray, distribution: str = "lognormal") -> dict: ...
def bootstrap_hc(values: np.ndarray, n_boot: int, seed: int) -> pd.DataFrame: ...
def plot_ssd(fit_result: dict, output_path: Path, style: str) -> None: ...
```

默认分布：

- log-normal
- log-logistic
- Burr III 或其他扩展分布作为后续可选项。

输出：

- `outputs/ssd/<chemical_id>/species_predictions.parquet`
- `outputs/ssd/<chemical_id>/ssd_fit_summary.csv`
- `outputs/ssd/<chemical_id>/ssd_curve.png`
- `outputs/ssd/<chemical_id>/ssd_curve.pdf`

测试：

- 少于最小物种数时拒绝拟合并输出原因。
- bootstrap seed 固定可复现。
- HC5 位于合理浓度范围内。

### 6.12 可解释性模块

包路径：`src/qsar_dl/evaluation/interpretability.py`

交付目标：

- 输出化学结构、描述符、物种上下文、时长、规则层的贡献解释。

方法：

- SHAP：用于表格描述符、规则特征和 LightGBM baseline。
- PDP/ICE：用于 logKow、duration_h、solubility_ratio、KOC 等关键连续变量。
- attention/embedding 可视化：用于深度模型内部表征。
- 消融解释：chemical-only 与 full model 差异。

输出：

- `outputs/figures/shap_summary.*`
- `outputs/figures/pdp_logkow.*`
- `outputs/figures/pdp_duration.*`
- `outputs/tables/feature_contribution_summary.csv`

测试：

- 小样本可运行。
- 图中变量名和单位完整。
- SHAP 缺失深度模型解释器时可回退 permutation importance。

### 6.13 科研可视化样式库

包路径：`src/qsar_dl/visualization/`

交付目标：

- 固定论文和学位论文图表风格。

默认样式：

- 中文字体：黑体。
- 英文字体：Arial。
- 图标题字号：20，可通过配置关闭标题。
- 坐标轴标题字号：18。
- 坐标轴刻度字号：18。
- 所有文字默认加粗。
- 轴线宽度：1.5。
- 图例无边框，位置可通过参数设置。
- 配色：Nature/Science 风格，色盲友好。
- 默认导出：PNG、TIFF、PDF、SVG。
- 默认 DPI：300 或更高。

必须实现的接口：

```python
def set_publication_style(style: str = "thesis_cn") -> None: ...
def save_figure(fig, path_stem: Path, formats=("png", "pdf", "svg", "tiff"), dpi=300) -> None: ...
def get_palette(name: str, n: int) -> list[str]: ...
```

测试：

- 中文标题不乱码。
- 导出四种格式。
- 图例无边框。
- 轴线宽度为 1.5。

### 6.14 远程服务器运行模块

包路径：`scripts/` 与 `configs/runtime/`

交付目标：

- 支持本地开发、远程服务器训练和结果回收。

必须提供：

- `scripts/train_remote.ps1`
- `scripts/sync_to_server.ps1`
- `scripts/sync_from_server.ps1`
- `configs/runtime/gpu_server.yaml`

配置字段：

```yaml
server:
  host: ""
  user: ""
  project_dir: ""
  conda_env: ""
  cuda_visible_devices: "0"
training:
  experiment_config: configs/experiments/multitask_residual.yaml
  output_dir: outputs/models
```

工程要求：

- 不把服务器密码、token、私钥写入仓库。
- 训练脚本必须记录 git commit hash。
- 每次训练输出 resolved config。

测试：

- 本地 dry-run 输出将要执行的命令。
- 缺少服务器配置时给出明确错误。

## 7. 并行开发任务拆分

推荐第一批并行任务：

| 任务 | 依赖 | 交付 |
|---|---|---|
| A. 仓库骨架与配置 | 无 | 目录、README、gitignore、基础配置 |
| B. 数据契约与 ECOTOX 清洗 | A | clean toxicity long table |
| C. 化合物特征 | A | chemical features |
| D. 物种特征 | A | species features |
| E. 显式规则层 | B/C/D | rule features |
| F. baseline 模型 | B/C | chemical-only model |
| G. 残差深度模型 | B/C/D/E/F | rule-aware residual model |
| H. 评估、AD、不确定度 | F/G | metrics、AD、uncertainty |
| I. SSD 风险预测 | G/H | SSD outputs |
| J. 可视化样式库 | 无 | plotting style package |
| K. 远程训练脚本 | A/G | server runbook |

合并原则：

- 所有模块以标准 parquet/csv/json 输出交互。
- 不允许模块读取其他模块未声明的私有中间文件。
- 每个模块必须有 fixture 和单元测试。
- 配置项必须写入 `configs/`，不得散落在脚本内部。

## 8. 最小可运行里程碑

Milestone 1：仓库与文档

- 远端 SSH 配置完成。
- 本施工图提交到 GitHub。
- `.gitignore` 避免大数据误提交。

Milestone 2：数据管线

- ECOTOX 原始表转换为标准长表。
- 输出清洗审计报告。
- 能解析 LC、EC、LOEC 与 effect level。

Milestone 3：特征与规则

- 化合物、物种、规则特征表生成。
- 每条规则有覆盖率报告。

Milestone 4：模型 baseline

- chemical-only 模型跑通。
- LightGBM 或 RandomForest baseline 跑通。
- 输出基础指标和图。

Milestone 5：结构优先残差深度模型

- `y_chemical`、`y_context_residual`、`y_rule_residual` 可拆分。
- 完成消融实验。
- R2 达到或接近 0.6 后进入 AD 和解释性分析。

Milestone 6：预测与 SSD

- `predict-toxicity` 和 `predict-risk` CLI 可用。
- SSD 输出 HC5 与图。

Milestone 7：迁移学习

- 非土壤数据预训练。
- 土壤数据微调。
- 输出迁移前后性能对比和机制解释。

## 9. 质量控制与科研解释要求

每次实验报告必须包含：

- 数据量、物种数、化合物数、endpoint 分布。
- 训练/验证/测试划分逻辑。
- 单位换算说明。
- 删除和排除记录清单。
- R2、RMSE、MAE、MAPE。
- 分 endpoint、taxon、habitat、chemical scaffold 的指标。
- chemical-only 与 full model 消融。
- 规则层覆盖率和贡献。
- AD 覆盖率。
- 不确定度校准。
- 图表和表格输出路径。
- 是否过拟合、是否欠拟合、是否可能数据泄漏、是否需要外部验证。

## 10. 文献与方法依据

本施工图的规则层和模型边界主要参考以下资料。实现时应在代码注释和报告中引用对应规则来源。

- OECD QSAR Assessment Framework, 2024：强调定义终点、应用域、预测可靠性、机制解释和用途适配。
- EPA ECOTOX Knowledgebase：作为跨物种、多终点生态毒性数据来源。
- EPA ECOSAR Technical Reference Document：baseline toxicity、logKow、化学类别和水溶解度边界。
- OECD Test No. 203 Fish Acute Toxicity Test：鱼类急性测试常用 96 h。
- OECD Test No. 202 Daphnia sp. Acute Immobilisation Test：溞类急性测试常用 48 h。
- OECD Test No. 201 Alga Growth Inhibition Test：藻类生长抑制测试常用 72 h。
- OECD Guidance Document on Aquatic Toxicity Testing of Difficult Substances and Mixtures：挥发、低溶解度、吸附等 difficult substances 测试问题。
- Verhaar 等 MoA 分类与 EnviroTox MoA 分类：baseline narcosis、polar narcosis、reactive、specific acting 的机制分层。
- TKTD/GUTS 文献：暴露时间、内暴露积累和毒效应随时间变化。
- equilibrium partitioning 与有机碳归一化理论：沉积物/土壤中 KOC、fOC 和自由浓度对生物有效性的影响。

## 11. 首批开发默认配置

随机种子：

```yaml
seed: 20260524
```

目标：

```yaml
target:
  name: target_ptox
  transform: negative_log10_molar
  endpoints: [LC, EC, LOEC]
```

物种上下文：

```yaml
species:
  include_taxonomy: true
  include_eco_group: true
  include_habitat_medium: true
  include_potential_routes: true
  include_lifestage: true
  missing_category: unknown
```

暴露：

```yaml
exposure:
  include_exposure_medium: false
  include_duration_h: true
  include_duration_bin: true
```

模型：

```yaml
model:
  architecture: rule_aware_residual
  enforce_chemical_baseline: true
  context_residual_l2: 0.01
  rule_residual_l2: 0.01
  report_residual_ratio: true
```

图表：

```yaml
plot:
  cn_font: SimHei
  en_font: Arial
  title_size: 20
  axis_label_size: 18
  tick_size: 18
  bold_text: true
  axis_linewidth: 1.5
  legend_frame: false
  dpi: 300
  formats: [png, tiff, pdf, svg]
```
