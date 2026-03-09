# 每日主要进度
*2026-03-09*
- 完成第四阶段的脚本配置
    | 文件路径 | 功能描述 |
    |----------|----------|
    | `scripts/plot_results.py` | 绘制对比柱状图，直观展示监督基线、线性评估、全微调在不同shot下的准确率，保存为 `docs/figures/comparison.png`。 |
    | `scripts/tsne_visualize.py` | 加载预训练编码器，提取CIFAR-10测试集特征，使用t-SNE降维至2D并可视化，保存为 `docs/figures/tsne_pretrained.png`。 |
    | `docs/figures/comparison.png` | 对比柱状图。 |
    | `docs/figures/tsne_pretrained.png` | t-SNE特征分布图。 |
- 新增code_analysis.ipynb
*2026-03-03*
- 在deepseek帮助下完成了第三阶段的代码文件
    | 文件路径 | 功能描述 |
    |----------|----------|
    | `datasets/fewshot_dataset.py` | 少样本数据集类，根据划分文件（JSON）加载指定索引的图像和标签，用于微调和监督基线训练。 |
    | `scripts/finetune.py` | 统一的微调/监督训练脚本，支持三种模式：监督基线（无预训练）、线性评估（冻结编码器）、全微调。加载预训练编码器，添加分类头，在少样本集上训练并测试。 |
    | `configs/supervised_5shot.yaml` | 5-shot 监督基线配置文件（随机初始化，全微调）。 |
    | `configs/finetune_5shot_linear.yaml` | 5-shot 线性评估配置文件（加载预训练，冻结编码器）。 |
    | `configs/finetune_5shot_full.yaml` | 5-shot 全微调配置文件（加载预训练，更新所有参数）。 |
    | `configs/supervised_10shot.yaml` | 10-shot 监督基线配置文件。 |
    | `configs/finetune_10shot_linear.yaml` | 10-shot 线性评估配置文件。 |
    | `configs/finetune_10shot_full.yaml` | 10-shot 全微调配置文件。 |
- 完成第三阶段的训练，新建result.md文件记录结果

*2026-02-27*
- 学习了对比学习的基本理论和SimCLR的基本架构
- 在deepseek帮助下完成了SimCLR模型的代码编写

    | 文件路径 | 功能描述 |
    |----------|----------|
    | `models/encoder.py` | 定义编码器（ResNet-18），输出512维特征向量。 |
    | `models/projection_head.py` | 定义投影头（两层MLP），将512维映射到128维对比空间。 |
    | `losses/nt_xent.py` | 实现 NT-Xent 对比损失函数，用于 SimCLR 训练。 |
    | `datasets/simclr_augment.py` | SimCLR 数据增强：随机裁剪、颜色抖动、灰度化等，返回两个视图。 |
    | `datasets/pretrain_dataset.py` | 预训练数据集类，加载 CIFAR-10 训练集并应用增强，返回两个视图。 |
    | `scripts/train_simclr.py` | 训练脚本，整合模型、损失、优化器、数据加载，支持配置文件和 TensorBoard 日志。 |
    | `configs/simclr_cifar.yaml` | 配置文件，包含所有超参数（batch size、学习率、epochs、温度等）。 |
- 开始进行模型训练，预计几小时后完成
*2026-02-26*
- 将项目移动到了vscode中，方便运行与调试
- 完善了项目记录，新增changelog和troubleshooting

*2026-02-25*
- 建立了GitHub仓库
- 配置了虚拟环境
- 完成了数据集的下载