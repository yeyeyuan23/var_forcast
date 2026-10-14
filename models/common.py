"""模型共用的训练窗口标准化；实验设置读取config/ 中选中模型的 YAML。"""

import numpy as np


def standardize_window(X_train, x_next):
    """每次只在当前训练窗口计算均值和标准差，禁止全样本标准化。"""
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X_train - mean) / scale, (x_next - mean) / scale


def standardize_sequences(X_train, x_next):
    """每个指标共用一个尺度，只由训练序列内的值估计。

    重叠日期按其在训练样本中的出现次数参与计算；不包含待预测序列的新数据。
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    scale = X_train.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X_train - mean) / scale, (x_next - mean) / scale
