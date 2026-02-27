# 每日主要进度
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