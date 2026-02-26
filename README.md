# 基于对比学习的少样本图像分类

本项目实现 SimCLR 对比学习框架，在 CIFAR-10 上进行无监督预训练，然后在少量标注数据（如每类5张）上微调，并与直接监督学习对比，验证对比学习在数据稀缺场景下的优势。

## 环境要求
- Python 3.8
- PyTorch 1.10+
- torchvision
- numpy, matplotlib, tensorboard, tqdm, scikit-learn

## 项目结构
- `data/`: 数据集存放及划分文件
- `configs/`: 配置文件
- `models/`: 模型定义
- `losses/`: 损失函数
- `datasets/`: 数据集加载
- `scripts/`: 训练脚本
- `utils/`: 工具函数
- `experiments/`: 实验记录（日志、模型权重）
- `notebooks/`: 分析可视化
- `docs/`: 文档（理论学习笔记、实验结果等）

## 进度
- [√] 阶段一：基础准备与环境搭建
- [ ] 阶段二：实现 SimCLR 预训练
- [ ] 阶段三：少样本微调与监督基线
- [ ] 阶段四：实验对比与分析
- [ ] 阶段五：拓展实验（可选）
- [ ] 阶段六：项目总结