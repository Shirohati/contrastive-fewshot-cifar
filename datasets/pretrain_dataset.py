import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import os

class PretrainDataset(Dataset):
    """用于SimCLR预训练的数据集，返回两个增强视图"""
    def __init__(self, root='./data', transform=None):
        self.data = CIFAR10(root=root, train=True, download=False)  # 假设数据已存在，不自动下载
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]  # 忽略标签
        if self.transform:
            view1, view2 = self.transform(img)  # 返回两个视图
        else:
            view1, view2 = img, img
        return view1, view2
    