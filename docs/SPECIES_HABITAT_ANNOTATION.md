# Species 生存介质标注说明

## 目标

为 `outputs/databases/ecotox_clean.sqlite` 中 `species` 全表 29,598 个物种增加生命周期主要栖居介质标注。

主标签字段为 `primary_medium`，取值：

- `aquatic`：水生；按本项目约定，两栖类也计入水生。
- `soil`：土壤生境为主。
- `sediment`：沉积物或底栖沉积物界面为主。
- `terrestrial`：陆生或非土壤陆生暴露为主。
- `unknown`：证据不足，不强行归类。

## 数据库新增字段

脚本会在 `species` 表中新增或更新：

- `primary_medium`
- `habitat_labels`
- `habitat_confidence`
- `habitat_evidence_tier`
- `habitat_evidence_source`
- `habitat_evidence_detail`
- `habitat_decision_rule`
- `habitat_review_status`
- `habitat_annotation_date`

同时创建完整审计表：

- `species_habitat_annotations`

## 输出文件

- 完整标注表：`outputs/tables/species_habitat_annotations.csv`
- 待复核清单：`outputs/tables/species_habitat_review_candidates.csv`
- 标注报告：`outputs/reports/species_habitat_annotation_report.json`
- WoRMS 检索缓存：`outputs/cache/worms_species_environment_cache.jsonl`

## 标注优先级

1. `ECOTOX tests.organism_habitat`
   - 若某物种测试记录中 `Water`、`Soil`、`Non-Soil` 有明确优势，则作为最高优先级证据。
   - `Water -> aquatic`
   - `Soil -> soil`
   - `Non-Soil -> terrestrial`

2. 分类学规则
   - 基于 `species` 表已有 `ecotox_group`、`phylum_division`、`class`、`tax_order`、`family` 等字段。
   - 鱼类、藻类、两栖类、典型水生无脊椎、典型土壤环节动物、鸟类、哺乳类、植物等按保守规则标注。
   - 对异质性强的宽泛类群使用较低置信度并标记 `review_recommended`。

3. WoRMS 外部权威数据库
   - 对 unknown 或低置信物种可分批调用 WoRMS REST API。
   - 使用 WoRMS 返回的 `isMarine`、`isBrackish`、`isFreshwater`、`isTerrestrial` 环境标记。
   - 检索结果写入本地缓存，后续可继续补跑。

## 当前结果

最近一次运行后：

- `aquatic`：9,775
- `sediment`：513
- `soil`：5,019
- `terrestrial`：13,277
- `unknown`：1,014

证据层级：

- `ecotox_test_habitat`：11,223
- `taxonomy_rule`：17,297
- `external_authority`：64
- `unknown`：1,014

## 运行命令

不联网初筛：

```powershell
E:\TOOLS\anaconda\python.exe src\annotate_species_habitat.py
```

仅使用已有 WoRMS 缓存合并：

```powershell
E:\TOOLS\anaconda\python.exe src\annotate_species_habitat.py `
  --use-worms `
  --worms-max-records 0 `
  --worms-confidence-threshold 0.01
```

继续对 unknown 分批检索 WoRMS：

```powershell
E:\TOOLS\anaconda\python.exe src\annotate_species_habitat.py `
  --use-worms `
  --worms-max-records 200 `
  --worms-confidence-threshold 0.01 `
  --worms-sleep-seconds 0 `
  --worms-timeout-seconds 5
```

`--worms-confidence-threshold 0.01` 表示只补 `unknown`；如果需要补低置信分类学规则，可提高到 `0.65`，但检索量会显著增加。

## 科研使用注意

- `primary_medium` 是为建模和分层分析准备的主介质标签，不等同于物种所有可能生境。
- `habitat_labels` 可保留辅助标签，例如 `aquatic;sediment` 或 `aquatic;amphibious`。
- 对 `review_recommended` 和 `needs_external_review` 的物种，在正式论文分析、SSD 分组或外部验证前应抽样或定向复核。
- 腹足纲、原生生物、寄生扁形动物、部分真菌和部分昆虫类群生态介质异质性较强，不能简单一刀切。
