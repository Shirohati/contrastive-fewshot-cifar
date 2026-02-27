import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import Encoder
from models.projection_head import ProjectionHead
from losses.nt_xent import NTXentLoss
from datasets.simclr_augment import SimCLRTransform
from datasets.pretrain_dataset import PretrainDataset

def train(config):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据增强和数据集
    transform = SimCLRTransform(size=config['image_size'])
    dataset = PretrainDataset(root=config['data_root'], transform=transform)

    # 数据加载器
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # 模型
    encoder = Encoder(pretrained=False).to(device)
    projector = ProjectionHead(
        input_dim=config['encoder_dim'],
        hidden_dim=config['proj_hidden_dim'],
        output_dim=config['proj_output_dim']
    ).to(device)

    # 优化器
    optimizer = optim.SGD(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # 损失函数
    criterion = NTXentLoss(temperature=config['temperature'])

    # TensorBoard 日志
    writer = SummaryWriter(log_dir=config['log_dir'])

    # 训练循环
    global_step = 0
    for epoch in range(1, config['epochs'] + 1):
        encoder.train()
        projector.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for view1, view2 in pbar:
            view1 = view1.to(device)
            view2 = view2.to(device)

            # 前向
            h1 = encoder(view1)
            h2 = encoder(view2)
            z1 = projector(h1)
            z2 = projector(h2)

            # 计算损失
            loss = criterion(z1, z2)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # 记录到 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)
            pbar.set_postfix({'loss': loss.item()})

        # 每个epoch结束后记录平均损失
        avg_loss = total_loss / len(loader)
        writer.add_scalar('Loss/epoch', avg_loss, epoch)

        # 更新学习率
        scheduler.step()

        # 保存模型检查点
        if epoch % config['save_every'] == 0 or epoch == config['epochs']:
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            checkpoint_path = os.path.join(config['ckpt_dir'], f'simclr_epoch{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Training finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/simclr_cifar.yaml', help='配置文件路径')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建必要的目录
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['ckpt_dir'], exist_ok=True)

    train(config)