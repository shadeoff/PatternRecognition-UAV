# 03 创建网络模型
from torch import nn


class Train(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),  # 添加ReLU激活函数
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),  # 添加ReLU激活函数
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),  # 添加ReLU激活函数
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),  # 修改输入大小
            nn.ReLU(),  # 添加ReLU激活函数
        )

    def forward(self,x):
        x = self.model(x)
        return x