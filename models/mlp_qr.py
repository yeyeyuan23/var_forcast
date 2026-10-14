"""MLP 分位数网络：一个隐藏层，同时预测三个左尾分位数。"""

import numpy as np
import torch
from torch import nn
from .common import standardize_window


def fit_mlp_quantiles(X_train, y_train, x_next, config):
    """一个隐藏层的 MLP，同时最小化三个分位数的 pinball loss。

    与线性回归的比较点：输入/目标相同；模型变为非线性，训练变为梯度下降。
    每天固定同一初始化种子、从头训练，避免隐含的扩展窗口记忆。
    """
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    X = torch.as_tensor(X_train, dtype=torch.float32)
    y = torch.as_tensor(y_train[:, None], dtype=torch.float32)
    a = torch.tensor(config.alphas, dtype=torch.float32)
    network = nn.Sequential(
        nn.Linear(X.shape[1], config.hidden),
        nn.Tanh(),
        nn.Linear(config.hidden, len(config.alphas)),
    )
    # 以训练窗口的经验分位数初始化截距，使初始预测有合理尺度。
    with torch.no_grad():
        network[-1].bias.copy_(
            torch.tensor(np.quantile(y_train, config.alphas), dtype=torch.float32)
        )
        network[-1].weight.mul_(0.01)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in range(config.epochs):
        prediction = network(X)  # shape: (W, 3)
        error = y - prediction  # 真实值减预测分位数
        loss = torch.maximum(a * error, (a - 1) * error).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        return network(torch.as_tensor(x_next, dtype=torch.float32)).numpy()[0]


def forecast(X_train, y_train, x_next, config):
    """标准化特征，以百分数单位训练，再将预测还原为小数单位。"""
    X, x = standardize_window(X_train, x_next)
    y = 100.0 * y_train
    return fit_mlp_quantiles(X, y, x, config).astype(float) / 100.0
