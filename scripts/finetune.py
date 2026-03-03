import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import Encoder
from datasets.fewshot_dataset import FewShotDataset

def evaluate(encoder, classifier, loader, device):
    """测试集评估函数"""
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def train_finetune(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据增强（训练集使用随机增强，测试集仅归一化）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # 少样本训练集
    train_dataset = FewShotDataset(
        root=config['data_root'],
        split_file=config['fewshot_split'],
        transform=train_transform,
        train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 测试集（全部测试图像）
    test_dataset = CIFAR10(
        root=config['data_root'],
        train=False,
        transform=test_transform,
        download=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 模型
    encoder = Encoder(pretrained=False).to(device)
    # 加载预训练权重（如果指定）
    if config['pretrained']:
        if not os.path.exists(config['pretrained_path']):
            raise FileNotFoundError(f"Pretrained model not found: {config['pretrained_path']}")
        checkpoint = torch.load(config['pretrained_path'], map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"Loaded pretrained encoder from {config['pretrained_path']}")
    else:
        print("Training from scratch (random initialization)")

    # 分类头
    classifier = nn.Linear(config['encoder_dim'], 10).to(device)

    # 根据模式设置是否冻结编码器
    if config['mode'] == 'linear':
        for param in encoder.parameters():
            param.requires_grad = False
        optimizer = optim.SGD(
            classifier.parameters(),
            lr=float(config['lr']),
            momentum=float(config['momentum']),
            weight_decay=float(config['weight_decay'])
        )
    else:  # 'full' 全微调
        optimizer = optim.SGD(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=float(config['lr']),
            momentum=float(config['momentum']),
            weight_decay=float(config['weight_decay'])
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['epochs']))

    best_acc = 0.0
    for epoch in range(1, int(config['epochs']) + 1):
        if config['mode'] == 'full':
            encoder.train()
        else:
            encoder.eval()  # 线性模式编码器不训练，但需设为eval模式以影响BN等
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            features = encoder(images)
            outputs = classifier(features)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)

        # 测试
        test_acc = evaluate(encoder, classifier, test_loader, device)
        print(f"Epoch {epoch}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            # 保存最佳模型
            checkpoint = {
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'test_acc': test_acc,
                'epoch': epoch
            }
            save_path = os.path.join(config['save_dir'], f"best_{config['mode']}.pth")
            torch.save(checkpoint, save_path)
            print(f"Best model saved to {save_path}")

        scheduler.step()

    print(f"Best test accuracy: {best_acc:.2f}%")
    return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    train_finetune(config)