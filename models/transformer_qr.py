"""Small one-layer encoder with sinusoidal positions and a last-token quantile head."""

import math
import numpy as np
import torch
from torch import nn
from .common import standardize_sequences


class QuantileTransformer(nn.Module):
    def __init__(self, input_size, hidden, heads, n_quantiles):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden)
        self.encoder = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=2 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.head = nn.Linear(hidden, n_quantiles)

    def forward(self, inputs):
        z = self.projection(inputs)
        position = torch.arange(z.shape[1], device=z.device, dtype=z.dtype)[:, None]
        frequencies = torch.exp(
            torch.arange(0, z.shape[2], 2, device=z.device, dtype=z.dtype)
            * (-math.log(10000.0) / z.shape[2])
        )
        encoding = torch.zeros(z.shape[1:], device=z.device, dtype=z.dtype)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        # All tokens precede the target date, so within-window bidirectional attention is causal for this forecast.
        return self.head(self.encoder(z + encoding)[..., -1, :])


def forecast(X_train, y_train, x_next, config):
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    X, x = standardize_sequences(X_train, x_next)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(100 * y_train[:, None], dtype=torch.float32)
    a = torch.tensor(config.alphas, dtype=torch.float32)
    network = QuantileTransformer(3, config.hidden, config.heads, len(config.alphas))
    with torch.no_grad():
        network.head.bias.copy_(
            torch.tensor(np.quantile(100 * y_train, config.alphas), dtype=torch.float32)
        )
        network.head.weight.mul_(0.01)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in range(config.epochs):
        error = y - network(X)
        loss = torch.maximum(a * error, (a - 1) * error).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite Transformer loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        return (
            network(torch.as_tensor(x, dtype=torch.float32)).numpy()[0].astype(float)
            / 100
        )
