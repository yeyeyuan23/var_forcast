"""线性分位数回归：每个尾概率分别拟合一个线性预测函数。"""

import numpy as np
from sklearn.linear_model import QuantileRegressor
from .common import standardize_window


def forecast(X_train, y_train, x_next, config):
    """特征只按训练窗口标准化；以百分数训练后，还原为小数收益率。"""
    X, x = standardize_window(X_train, x_next)
    y = 100.0 * y_train
    predictions = []
    for tail_probability in config.alphas:
        # sklearn 的 alpha 是 L1 正则强度，不是本项目的尾概率！
        model = QuantileRegressor(
            quantile=tail_probability, alpha=config.l1_penalty, solver="highs"
        )
        model.fit(X, y)
        predictions.append(model.predict(x)[0])
    return np.array(predictions) / 100.0
