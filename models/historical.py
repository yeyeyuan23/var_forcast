"""历史模拟法：用训练窗口内收益率的经验分位数预测 VaR。"""

import numpy as np


def forecast(X_train, y_train, x_next, config):
    """只使用历史收益率 y_train；特征参数保留以统一四个模型的接口。"""
    # 经验分布路线：排序后取指定位置，使用线性插值。
    return np.quantile(y_train, config.alphas, method="linear")
