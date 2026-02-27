import torchvision.transforms as T
from PIL import Image
import random

class SimCLRTransform:
    """为每张图像生成两个增强视图"""
    def __init__(self, size=32):
        # CIFAR-10 原始大小 32x32
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),  # 随机裁剪缩放
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),  # 颜色抖动
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 统计数据
        ])

    def __call__(self, x):
        # 返回两个增强视图
        return self.transform(x), self.transform(x)