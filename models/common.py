"""模型共用的训练窗口标准化；实验设置读取config/ 中选中模型的 YAML。"""

import numpy as np


def standardize_window(X_train, x_next):
    """每次只在当前训练窗口计算均值和标准差，禁止全样本标准化。"""
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X_train - mean) / scale, (x_next - mean) / scale
