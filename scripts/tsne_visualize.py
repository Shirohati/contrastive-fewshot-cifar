import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.datasets import CIFAR10
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoder import Encoder

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载预训练编码器
    encoder = Encoder(pretrained=False).to(device)
    checkpoint_path = './experiments/simclr/checkpoints/simclr_epoch400.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    print("预训练编码器加载成功")

    # 准备测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    testset = CIFAR10(root='./data', train=False, transform=transform, download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # 提取特征
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            feats = encoder(images).cpu().numpy()
            features_list.append(feats)
            labels_list.append(labels.numpy())
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    print(f"特征矩阵形状: {features.shape}")

    # 随机采样 2000 个点加速 t-SNE
    np.random.seed(42)
    indices = np.random.choice(len(features), 2000, replace=False)
    features_sub = features[indices]
    labels_sub = labels[indices]

    # t-SNE 降维
    print("正在运行 t-SNE（可能需要几分钟）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features_sub)
    print("t-SNE 完成")

    # 绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_sub, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Class')
    plt.title('t-SNE visualization of pretrained encoder features on CIFAR-10 test set')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    os.makedirs('docs/figures', exist_ok=True)
    plt.savefig('docs/figures/tsne_pretrained.png', dpi=150)
    plt.show()
    print("t-SNE 图已保存至 docs/figures/tsne_pretrained.png")

if __name__ == '__main__':
    main()