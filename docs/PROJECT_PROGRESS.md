# 项目进度总览

日期：2026-05-25

## 1. 当前阶段定位

本项目已经完成第一阶段接口规范和第一轮模块骨架集成，当前进入真实数据 baseline 诊断阶段。代码层面已经具备从 clean ECOTOX SQLite 构建建模长表、生成化合物/物种/规则特征、运行传统机器学习 baseline、运行残差式深度 QSAR smoke、评估划分、应用域、不确定度和 SSD 示例的基础能力。

当前主线仍保持以下边界：

- 主预测信号以化合物结构、描述符、指纹和机制信息为核心。
- 物种、lifestage、`primary_medium`、暴露时长和规则层作为上下文残差、可靠性解释和外推能力增强项。
- 不把 `habitat_medium` 或 `primary_medium` 当成实验暴露介质。
- 第一阶段主任务聚焦水相 `LC/EC/LOEC -> mg/L -> mol/L -> pTox`。
- 土壤、沉积物和经口剂量记录暂作为迁移学习或后续扩展候选。

## 2. 数据准备进展

已完成 clean SQLite 准备层：

- 数据库：`outputs/databases/ecotox_clean.sqlite`
- 构建脚本：`src/build_clean_ecotox_sqlite.py`
- 构建说明：`docs/ECOTOX_CLEAN_DATABASE.md`
- 构建报告：`outputs/reports/ecotox_clean_build_report.json`

最近一次构建结果：

| 表 | 行数 |
|---|---:|
| `results` | 1,234,077 |
| `tests` | 724,182 |
| `chemicals` | 18,520 |
| `species` | 29,598 |
| `references` | 131,197 |
| `ecotox_toxicity_joined` | 1,234,077 |

SMILES 字典合并状态：

- 有 SMILES 的化合物：16,582
- 无 SMILES 的化合物：1,938
- `results -> tests`、`tests -> chemicals/species/references` 无缺失关联

物种主生存介质标注已完成：

| `primary_medium` | 物种数 |
|---|---:|
| `aquatic` | 9,775 |
| `sediment` | 513 |
| `soil` | 5,019 |
| `terrestrial` | 13,277 |
| `unknown` | 1,014 |

说明：

- `primary_medium` 存在于 `species` 表中。
- `src/qsar_dl/data/contract.py` 保留了从 `species.primary_medium` 合并补齐的兼容逻辑。
- 2026-05-25 已运行 `src/standardize_clean_ecotox_sqlite.py`，当前 `ecotox_toxicity_joined` 已直接包含 `primary_medium`、MW、标准化浓度和标准化时间字段。

标准化写回结果：

| 项目 | 数量 |
|---|---:|
| RDKit MW 可用化合物 | 16,580 |
| 缺 SMILES 化合物 | 1,938 |
| 无效 SMILES 化合物 | 2 |
| `results.conc1_*` 可标准化记录 | 551,804 |
| `results.obs_duration_mean` 可标准化记录 | 1,052,479 |
| `tests.exposure_duration_*` 可标准化记录 | 366,100 |

标准化规则：

- MW 写入 `chemicals.molecular_weight_g_mol`，来源为 RDKit SMILES。
- 时间字段统一为小时，后缀 `_h`。
- 可识别的水相质量浓度统一为 `mg/L`，摩尔浓度统一为 `mol/L`。
- 可识别的土壤/沉积物浓度统一为 `mg/kg`。
- 经口剂量统一为 `mg/kg/d`。
- 无法识别的浓度单位保留为 `other`，不进入主水相建模。

## 3. 建模长表生成状态

已完成真实数据契约管线运行：

- 输出表：`outputs/tables/modeling_toxicity_long.parquet`
- 兼容 CSV：`outputs/tables/modeling_toxicity_long.csv`
- 审计报告：`outputs/reports/modeling_table_build_report.json`

生成结果：

| 指标 | 数量 |
|---|---:|
| 总记录数 | 1,234,077 |
| 主水相任务记录数 | 226,123 |
| 迁移候选记录数 | 526,114 |
| 严格迁移可建模记录数 | 8,067 |
| 缺 SMILES 记录数 | 15,589 |
| 缺 MW 记录数 | 15,590 |

Endpoint 分布：

| endpoint family | 记录数 |
|---|---:|
| `LC` | 162,629 |
| `EC` | 77,897 |
| `LOEC` | 127,686 |
| unsupported/missing | 865,865 |

单位族分布：

| target unit family | 记录数 |
|---|---:|
| `water_mg_l` | 513,489 |
| `soil_mg_kg` | 29,784 |
| `oral_mg_kg_d` | 8,476 |
| `other` | 682,328 |

派生方法分布：

| 字段 | 方法 | 记录数 |
|---|---|---:|
| 浓度 | `mean` | 1,083,511 |
| 浓度 | `direct_range_midpoint` | 149,749 |
| 浓度 | `missing` | 817 |
| 时长 | `exposure_mean` | 797,370 |
| 时长 | `observation_mean` | 290,568 |
| 时长 | `exposure_range_grid_mid` | 26,469 |
| 时长 | `missing_manual_review` | 119,670 |

主要无法进入主任务原因：

| reason | 记录数 |
|---|---:|
| `unsupported_endpoint` | 865,865 |
| `missing_or_invalid_target` | 726,637 |
| `non_main_unit_family` | 720,588 |
| `non_main_medium` | 530,006 |
| `missing_mw` | 15,590 |
| `missing_smiles` | 15,589 |

工程说明：

- 初次全量构建因逐行 records 复制和重复 RDKit MW 计算超时。
- 已将数据契约管线改为迭代行处理，并缓存 SMILES 到 MW 的 RDKit 计算。
- 优化后全量构建完成，parquet 写入成功。
- 2026-05-25 后建模长表优先使用 clean SQLite 中的标准化 MW、浓度和时间字段。
- `is_transfer_candidate` 保留宽松候选池，`is_transfer_model_ready` 表示其中已可用于当前 pTox 框架的严格子集。

## 4. 模块实现进展

| 任务 | 状态 | 主要交付 |
|---|---|---|
| A 配置系统 | 已完成 | `src/qsar_dl/config/`、分层 include、路径解析、`${...}` 插值、resolved config 导出 |
| B 数据契约管线 | 已完成单元实现 | `load_clean_sqlite`、浓度/时长派生、endpoint parser、单位族和 `target_ptox` 派生 |
| C 化合物特征 | 已完成单元实现 | RDKit 描述符、Morgan fingerprint、物化性质接口和缺失 mask |
| D 描述符分组 | 已完成 | RDKit 描述符先验分组、固定权重组级特征、PyTorch 分组加权层 |
| E 物种上下文 | 已完成 | taxonomy、eco group、`primary_medium`、lifestage 编码和 missing/unknown mask |
| F 规则层 | 已完成骨架 | `RuleOutput`、规则注册、可测试规则、TODO/stub 规则 |
| G 传统 ML baseline | 已完成并完成真实数据首轮运行 | PLS、ElasticNet、RandomForest smoke、可复用 runner、标准特征和固定描述符分组特征 |
| H 深度模型骨架 | 已完成 | `ResidualQSARModel`、prediction decomposition、深度 smoke 训练 |
| I 评估与划分 | 已完成 | 化学类别规则、category holdout split、回归指标 |
| J AD 与不确定度 | 已完成 | chemical/species/rule AD、ensemble uncertainty、conformal offset |
| K SSD | 已完成 | 敏感物种选择、lognormal/loglogistic SSD、bootstrap HC |
| L 可视化样式 | 已完成 | 论文/学位论文风格、调色板、多格式高分辨率导出 |
| M 远程运行骨架 | 已完成 | sync/train PowerShell dry-run 脚本和服务器运行说明 |

## 5. 已知占位和暂缓项

规则层中以下模块仍为明确 TODO/stub，不能在论文结果中解释为已校准机制模型：

- chemical activity：等待自由浓度和水溶解度 mol/L 字段标准化。
- MoA excess toxicity：等待 MoA、ToxCast 和 structure alert 输入稳定。
- TKTD：暂为候选，不得与 duration correction 双重计入。
- volatility：等待 Henry 常数和蒸气压字段稳定。
- soil/sediment bioavailability：等待 `f_OC` 或可靠自由浓度字段，不伪造自由浓度。
- route access：等待物种潜在暴露途径字典和编码规则。

其他限制：

- 当前已完成传统 ML 真实数据 baseline 首轮运行；深度模型尚未在真实 ECOTOX 建模长表上运行。
- `baseline_ml` 在 base Anaconda 环境中会因 `sklearn/scipy/numpy` DLL 问题跳过训练测试；真实训练应使用 `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe`。

## 6. 集成验证状态

当前推荐解释器：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe
```

已完成验证：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m pytest
```

结果：

```text
102 passed
```

base 解释器兼容性检查：

```powershell
E:\TOOLS\anaconda\python.exe -m pytest
```

结果：

```text
90 passed, 4 skipped, 1 warning
```

说明：

- 4 个 skipped 来自 base 环境中 `sklearn/scipy/numpy` DLL 链接问题以及缺少 parquet 引擎。
- `baseline_ml.py` 已改为延迟导入 scikit-learn，因此依赖损坏不会再导致测试收集阶段崩溃。
- `git diff --check` 无 whitespace error，仅有 Git 在 Windows 下的 CRLF 提示。

## 7. 下一步任务

### 7.1 已完成：真实建模长表生成

已完成：

- `outputs/tables/modeling_toxicity_long.parquet`
- `outputs/tables/modeling_toxicity_long.csv`
- `outputs/reports/modeling_table_build_report.json`

后续仍需人工解释：

- 为什么 unsupported endpoint 占比较高，是否应补充 NOEC 或其他慢性终点到扩展任务。
- 为什么 `other` 单位族较多，是否需要新增单位解析规则。
- 主水相任务中 LC/EC/LOEC 是否应先分层建模。

### 7.2 已完成首轮：真实数据 baseline

已新增可复用入口：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe src\run_baseline_ml_experiment.py --models pls elasticnet
```

全量主水相任务运行结果：

- 输入范围：`is_main_water_task == true` 且 `target_ptox` 可用。
- 记录数：226,123。
- 化合物数：4,987。
- 切分：同一 `chemical_id` 不跨训练/验证，`train=192,379`，`validation=33,744`。
- 输出报告：`outputs/models/baseline_ml_v001/baseline_metrics.json`。

全量 baseline 指标：

| 特征集 | 模型 | R2 | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|---:|
| `standard` | PLS | 0.307 | 1.661 | 1.363 | 31.98% |
| `standard` | ElasticNet | 0.280 | 1.693 | 1.394 | 32.38% |
| `standard` | RandomForest | 0.519 | 1.383 | 1.072 | 22.76% |
| `standard` | XGBoost | 0.502 | 1.408 | 1.126 | 24.53% |
| `standard` | LightGBM | 0.556 | 1.329 | 1.047 | 23.07% |
| `fixed_descriptor_groups` | PLS | 0.303 | 1.665 | 1.357 | 31.97% |
| `fixed_descriptor_groups` | ElasticNet | 0.269 | 1.706 | 1.407 | 33.11% |
| `fixed_descriptor_groups` | RandomForest | 0.504 | 1.405 | 1.084 | 23.55% |
| `fixed_descriptor_groups` | XGBoost | 0.477 | 1.443 | 1.160 | 25.22% |
| `fixed_descriptor_groups` | LightGBM | 0.544 | 1.347 | 1.065 | 23.13% |

全量输出产物：

- 指标 JSON：`outputs/models/baseline_ml_v001/baseline_metrics.json`
- 指标表：`outputs/models/baseline_ml_v001/表格/全量基线_模型指标汇总.csv`
- 验证集预测表：`outputs/models/baseline_ml_v001/表格/全量基线_验证集预测结果.parquet`
- 中文命名图表目录：`outputs/models/baseline_ml_v001/图表/`
- 关键过程图：`全量基线_数据筛选与切分流程.png`、`全量基线_模型性能对比.png`
- 每个模型/特征集均输出真实值-预测值图和残差分布图，PNG/PDF 各一份。

解释：

- 当前 baseline 使用统一后的水相 `mg/L -> mol/L -> pTox` 目标。
- 传统线性模型已有中等解释能力，说明化学结构和基础上下文含有稳定信号。
- 全量非线性模型明显优于线性 baseline；当前最佳为标准特征 LightGBM，R2=0.556、RMSE=1.329。
- 固定描述符分组特征略低于标准特征，但差距不大，说明分组先验可以作为可解释压缩特征继续用于深度模型和消融。
- 当前化学类别主要来自 SMILES/name heuristic，`other_unknown` 仍较多，类别外推结论应谨慎解释。

后续仍需：

- 分 endpoint、chemical category、taxon 和 AD 的分层评估。
- 进一步输出论文级分层图、误差来源诊断图和外推类别表现图。

### 7.3 科研解释重点

后续结果报告不能只看总体 R2，需要同时判断：

- 是否存在数据泄漏，尤其是同一化合物跨 train/test。
- chemical-only 是否有合理性能。
- full model 的提升是否主要来自物种上下文，而不是结构信息。
- LOEC 与 LC/EC 是否应分开主展示。
- 外推类别、AD 外样本和低溶解度/高 KOC/高 MW 化合物是否有系统偏差。

### 7.4 已完成：curated baseline 分层诊断与图表重导出

已生成诊断报告：

- `docs/BASELINE_DIAGNOSTIC_REPORT.md`

已使用更新后的 `src/qsar_dl/visualization/style.py` 重新导出图表：

- 输出目录：`outputs/models/baseline_ml_full_curated_rf_xgb_lgbm/图表/`
- 图表数量：66 个
- 图表格式：PNG 和 PDF
- 内容：数据筛选流程、模型性能对比、单模型综合诊断、全部模型拼图、endpoint/化学类别/物种类群/AD 分层残差图

本轮重新导出后已修正中文图表的字体 fallback 问题。`language="zh"` 当前优先使用 `Microsoft YaHei`、`SimHei`、`SimSun`，避免中文标题和图例被 Arial 绑定后出现 glyph warning；`language="en"` 仍优先使用 Arial。

curated baseline 当前主结果如下：

| 特征集 | 模型 | R2 | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|---:|
| `standard` | RandomForest | 0.392 | 1.375 | 1.053 | 27.14% |
| `standard` | XGBoost | 0.447 | 1.312 | 1.019 | 32.13% |
| `standard` | LightGBM | 0.502 | 1.246 | 0.952 | 31.02% |
| `fixed_descriptor_groups` | RandomForest | 0.427 | 1.335 | 1.017 | 26.65% |
| `fixed_descriptor_groups` | XGBoost | 0.446 | 1.314 | 1.025 | 34.38% |
| `fixed_descriptor_groups` | LightGBM | 0.496 | 1.253 | 0.960 | 32.42% |

主要诊断结论：

- `standard + LightGBM` 是当前 curated baseline 主结果。
- endpoint 中 EC 表现最好，LOEC 最弱，后续论文展示应分 endpoint。
- 化学类别中 `fluorinated_organic`、`unclassified` 和 `pharmaceutical_pcp` 是主要高误差类别。
- 物种类群中 `amphibian`、`cyanobacteria` 和 `algae` 是主要高误差类群。
- 当前 AD 分层验证集全部为 `AD内`，说明 AD 规则仍偏宽松，下一轮应补充 Morgan fingerprint Tanimoto、descriptor 距离和类别外推标记。

### 7.5 已完成：真实数据 deep baseline 训练入口与全量消融

已新增深度学习真实数据训练链路：

- 训练模块：`src/qsar_dl/training/train_deep.py`
- 命令行入口：`src/run_deep_qsar_experiment.py`
- 配置文件：`configs/experiments/baseline_deep.yaml`
- 开发记录：`docs/DEEP_QSAR_DEVELOPMENT.md`

当前 deep baseline 已从 chemical-only smoke 推进到全量三组消融：

| 消融实验 | 化合物结构 | Endpoint | Duration | Species/Taxon | 规则层 |
|---|---|---|---|---|---|
| `chemical_only` | 是 | 否 | 否 | 否 | 否 |
| `chemical_endpoint_duration` | 是 | 是 | 是 | 否 | 否 |
| `chemical_species_context` | 是 | 是 | 是 | 是 | 否 |

默认配置：

| 参数 | 当前值 |
|---|---:|
| `max_rows` | null |
| `batch_size` | 1,024 |
| `max_epochs` | 8 |
| `learning_rate` | 0.001 |
| `weight_decay` | 0.0001 |
| `patience` | 3 |

已完成全量运行：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe src\run_deep_qsar_experiment.py `
  --device cpu `
  --output-dir outputs\models\deep_ablation_full_v001
```

输出目录：

- `outputs/models/deep_ablation_full_v001/`

全量数据范围：

| 指标 | 数量 |
|---|---:|
| 主水相任务记录数 | 226,123 |
| 训练集记录数 | 185,158 |
| 验证集记录数 | 40,965 |
| 化合物数 | 4,987 |

全量消融结果：

| 消融实验 | 训练 R2 | 验证 R2 | 验证 RMSE | 验证 MAE | 验证 MAPE |
|---|---:|---:|---:|---:|---:|
| `chemical_only` | 0.590 | 0.288 | 1.489 | 1.150 | 32.60% |
| `chemical_endpoint_duration` | 0.637 | 0.350 | 1.423 | 1.096 | 33.24% |
| `chemical_species_context` | 0.711 | 0.413 | 1.351 | 1.026 | 32.32% |

解释：

- 消融结果符合原始三层 residual-QSAR 设计：结构主效应是基础，endpoint/duration 带来增益，species/taxon context 进一步改善跨物种预测。
- 当前最佳 deep 模型仍低于 curated 传统 ML 的 `standard + LightGBM`，但已经证明上下文残差层有效。
- 规则层仍保持禁用，因为 mechanistic-rule residual 尚未校准，不能用于正式机制解释。
- 已输出对应可视化图表：验证集指标对比、训练/验证损失曲线、验证集真实-预测散点、endpoint/化学类别/物种类群残差箱线图。
