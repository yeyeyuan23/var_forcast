"""只导入本次需要的模型；不触发其他模型的依赖或训练。"""

from importlib import import_module
from ..config import Config as Config, METHODS, LABELS as LABELS


def forecast(method, X_train, y_train, x_next, config):
    if method not in METHODS:
        raise ValueError(f"未知模型：{method}")
    module = import_module(f".{method}", __name__)
    return module.forecast(X_train, y_train, x_next, config)
