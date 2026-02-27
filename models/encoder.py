import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class Encoder(nn.Module):
    """编码器：返回图像的特征向量（去掉最后的分类层）"""
    def __init__(self, pretrained=False):
        super().__init__()
        # 加载标准的resnet18，不使用预训练权重（因为我们要自监督训练）
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # 去掉最后的全连接层和池化层？实际上resnet18最后是 avgpool + fc，我们需要保留avgpool的输出
        # 通常取 avgpool 后的 512 维特征
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的fc层和avgpool？注意：list(resnet.children()) 最后一个是fc，之前是avgpool
        # 更准确的做法：取到 avgpool 输出，然后自己展平
        # 但为了简便，我们可以用 resnet 的 features 部分，最后输出 shape (batch, 512, 1, 1)
        # 我们需要展平为 (batch, 512)
    def forward(self, x):
        # x: (batch, 3, 32, 32)
        features = self.encoder(x)  # (batch, 512, 1, 1)
        return features.flatten(start_dim=1)  # (batch, 512)