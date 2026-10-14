"""单层 GRU 分位数网络：读取多日指标，以最后的隐藏状态预测下一日 VaR。"""

import numpy as np
import torch
from torch import nn


def standardize_sequences(X_train, x_next):
    """每个指标共用一个尺度，只由训练序列内的值估计。

    重叠日期按其在训练样本中的出现次数参与计算；不包含待预测序列的新数据。
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    scale = X_train.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X_train - mean) / scale, (x_next - mean) / scale


class QuantileGRU(nn.Module):
    """每个序列从零状态开始；不在样本或预测日之间保留隐藏状态。"""

    def __init__(self, input_size, hidden, n_quantiles):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_quantiles)

    def forward(self, inputs):
        _, hidden = self.encoder(inputs)
        return self.head(hidden[-1])


def forecast(X_train, y_train, x_next, config):
    """每天从头训练；输入 (W, L, 3)，标签是实际收益，损失是 pinball loss。"""
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    X, x = standardize_sequences(X_train, x_next)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(100.0 * y_train[:, None], dtype=torch.float32)
    alphas = torch.tensor(config.alphas, dtype=torch.float32)
    network = QuantileGRU(X.shape[-1], config.hidden, len(config.alphas))
    with torch.no_grad():
        network.head.bias.copy_(
            torch.tensor(
                np.quantile(100.0 * y_train, config.alphas), dtype=torch.float32
            )
        )
        network.head.weight.mul_(0.01)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in range(config.epochs):
        error = y - network(X)
        loss = torch.maximum(alphas * error, (alphas - 1) * error).mean()
        if not torch.isfinite(loss):
            raise ValueError("GRU 训练损失出现非有限值")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        prediction = network(torch.as_tensor(x, dtype=torch.float32))
    return prediction.numpy()[0].astype(float) / 100.0
