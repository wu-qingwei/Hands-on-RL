import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gym

# %%
'''
普适逼近定理：多层感知器近似于紧子集上的任何连续函数。
一个具有至少一层的隐含层的前馈神经网络，并且隐含层包含有限数量的神经元（即多层感知机），
它可以以任意精度逼近任意一个定义在ℝ𝑛𝑛的闭集里的连续函数。
•前提是这个前馈神经网络的激活函数满足某些性质，例如Sigmoid函数，Tanh函数，ReLU函数

2024.1.8  by wqw
'''
x = np.arange(-5, 5, 0.1, dtype=np.float32)
x = torch.from_numpy(x)
print('x = ', x)
n1 = torch.relu(-5 * x - 7.7)
n2 = torch.relu(-1.2 * x - 1.3)
n3 = torch.relu(1.2 * x + 1)
n4 = torch.relu(1.2 * x - .2)
n5 = torch.relu(2 * x - 1.1)
n6 = torch.relu(5 * x - 5)
y = -n1 - n2 - n3 + n4 + n5 + n6

plt.figure(1)
plt.plot(x, y)
plt.show()

print(n1)

# %%
# ResNet
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        res = self.block(x)
        return res + x
