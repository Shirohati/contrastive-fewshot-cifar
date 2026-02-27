import torch.nn as nn

class ProjectionHead(nn.Module):
    """SimCLR的投影头：两层MLP，输出128维向量用于对比损失"""
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)