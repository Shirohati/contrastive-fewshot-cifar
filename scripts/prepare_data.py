import torchvision
import torch
import numpy as np
import json
import os
from collections import defaultdict

# 设置随机种子保证可重复
np.random.seed(42)
torch.manual_seed(42)

# 参数设置
data_dir = './data'          # 数据集存放位置
K_shot = 5                   # 每个类别选多少张用于少样本
output_dir = './data/splits' # 划分文件输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 下载并加载CIFAR-10训练集
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
images = trainset.data       # numpy数组，形状 (50000, 32, 32, 3)
labels = trainset.targets    # 列表，长度50000

# 1. 生成无标注预训练数据列表：所有图像路径
# 注意：torchvision下载的图片实际已保存在data_dir，但为了方便，我们记录索引
# 实际使用时，我们会通过Dataset根据索引读取图像。
# 这里生成一个txt文件，每行一个图像索引（0-49999）
with open(os.path.join(output_dir, 'pretrain_indices.txt'), 'w') as f:
    for idx in range(len(images)):
        f.write(f"{idx}\n")

# 2. 生成少样本标注数据：按类别随机选取K张
label_to_indices = defaultdict(list)
for idx, label in enumerate(labels):
    label_to_indices[label].append(idx)

fewshot_indices = []
fewshot_labels = []
for label in range(10):  # CIFAR-10有10类
    selected = np.random.choice(label_to_indices[label], size=K_shot, replace=False)
    fewshot_indices.extend(selected.tolist())
    fewshot_labels.extend([label] * K_shot)

# 保存为json文件，包含索引和对应标签
fewshot_data = {
    'indices': fewshot_indices,
    'labels': fewshot_labels
}
with open(os.path.join(output_dir, f'fewshot_{K_shot}shot.json'), 'w') as f:
    json.dump(fewshot_data, f)

print("数据划分完成！")
print(f"预训练样本数: {len(images)}")
print(f"少样本标注样本数: {len(fewshot_indices)}")