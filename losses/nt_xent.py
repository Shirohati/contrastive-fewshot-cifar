import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent损失，用于SimCLR"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, z_i, z_j):
        """
        输入: z_i, z_j 为同一个batch内两个不同视图的投影向量，形状均为 (batch_size, dim)
        返回: 标量损失
        """
        batch_size = z_i.shape[0]
        # 拼接得到 (2*batch_size, dim)
        z = torch.cat([z_i, z_j], dim=0)

        # 计算所有样本对之间的余弦相似度矩阵 (2*batch_size, 2*batch_size)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        # 将温度参数应用到相似度上
        sim = sim / self.temperature

        # 创建掩码，排除自身对自身的相似度（对角线）
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, -float('inf'))

        # 正样本对的索引：对于每个i，其正样本是 i+batch_size（如果i<batch_size）或 i-batch_size（如果i>=batch_size）
        # 更简单：用矩阵方式
        # 我们构造标签：前batch_size个样本的正样本是后batch_size个，后batch_size个的正样本是前batch_size个
        labels = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)], dim=0).to(z.device)

        loss = self.criterion(sim, labels)
        return loss / (2 * batch_size)  # 平均损失