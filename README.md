# 深度学习 QSAR

本仓库用于构建多物种、多毒性终点的深度学习 QSAR 框架，目标包括：

- 毒性预测：输入物种与化合物标识，输出毒性预测、不确定度和应用域信息。
- 风险预测：输入化合物，批量预测多物种毒性并自动拟合 SSD 曲线。
- 迁移学习：先在非土壤全量 ECOTOX 数据上预训练，再使用土壤数据微调。

第一阶段施工图见 [docs/IMPLEMENTATION_BLUEPRINT.md](docs/IMPLEMENTATION_BLUEPRINT.md)。
