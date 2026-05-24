# 深度学习 QSAR

本仓库用于构建多物种、多毒性终点的深度学习 QSAR 框架，目标包括：

- 毒性预测：输入物种与化合物标识，输出毒性预测、不确定度和应用域信息。
- 风险预测：输入化合物，批量预测多物种毒性并自动拟合 SSD 曲线。
- 迁移学习：先在非土壤全量 ECOTOX 数据上预训练，再使用土壤数据微调。

第一阶段施工图见 [docs/IMPLEMENTATION_BLUEPRINT.md](docs/IMPLEMENTATION_BLUEPRINT.md)。

接口规范文档：

- [DATA_CONTRACT.md](docs/DATA_CONTRACT.md)：clean SQLite 到建模长表的数据契约。
- [CONFIG_SPEC.md](docs/CONFIG_SPEC.md)：拆分配置与 resolved config 规范。
- [DESCRIPTOR_GROUPING.md](docs/DESCRIPTOR_GROUPING.md)：描述符先验分组加权模块。
- [RULE_LAYER.md](docs/RULE_LAYER.md)：显式毒理规则层接口。
- [MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)：主模型、baseline 与迁移学习架构。
- [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)：化学类别外推、消融、AD 与 SSD 评估协议。
- [MODULE_TASKS.md](docs/MODULE_TASKS.md)：可拆给子代理并行实现的任务说明。
