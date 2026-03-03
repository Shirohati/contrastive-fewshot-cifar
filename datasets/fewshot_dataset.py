import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import json
import os

class FewShotDataset(Dataset):
    """少样本数据集：根据给定的索引列表加载图像和标签"""
    def __init__(self, root, split_file, transform=None, train=True):
        """
        root: 数据根目录（包含 cifar-10-batches-py）
        split_file: json文件路径，包含 'indices' 和 'labels'
        transform: 数据增强
        train: True 表示使用训练集（CIFAR-10 训练集），False 表示测试集（但此处仅用于训练集）
        """
        self.data = CIFAR10(root=root, train=train, download=False)
        self.transform = transform

        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        self.indices = split_data['indices']
        self.labels = split_data['labels']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.data[self.indices[idx]]  # 忽略原始标签
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    