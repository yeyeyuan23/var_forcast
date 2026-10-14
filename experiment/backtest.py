"""单模型滚动回测：每个预测日仅使用此前已经兑现的训练标签。"""

from time import perf_counter
import numpy as np
import pandas as pd
from .data import FEATURES, make_feature_sequences
from ..models import forecast


def rolling_backtest(table, start, stop, config, progress=None):
    """第 i 行为预测目标；训练固定取 [i-window, i)，绝不包含 y_i。"""
    window, method = config.window, config.model
    lookback = config.lookback if method == "gru_qr" else 1
    offset = lookback - 1
    if start < window + offset or stop > len(table) or start >= stop:
        raise ValueError("评估范围无效或没有足够的训练数据")
    if not np.isfinite(table[FEATURES + ["y"]].to_numpy()).all():
        raise ValueError("监督表中存在非有限数值")
    if table.index.has_duplicates or not table.index.is_monotonic_increasing:
        raise ValueError("监督表日期必须唯一且有序")
    inputs = (
        make_feature_sequences(table, lookback)
        if method == "gru_qr"
        else table[FEATURES].to_numpy()
    )
    records = []
    for i in range(start, stop):
        train = table.iloc[i - window : i]
        before = perf_counter()
        raw = forecast(
            method,
            inputs[i - window - offset : i - offset],
            train["y"].to_numpy(),
            inputs[i - offset : i - offset + 1],
            config,
        )
        elapsed = perf_counter() - before
        if raw.shape != (len(config.alphas),) or not np.isfinite(raw).all():
            raise ValueError(f"{method} 在 {table.index[i]} 输出无效")
        crossed = bool(np.any(np.diff(raw) < 0))
        ordered = np.sort(raw)
        actual = float(table["y"].iloc[i])  # 只在预测完成后保存，用于评估。
        for j, alpha in enumerate(config.alphas):
            records.append(
                dict(
                    date=table.index[i],
                    model=method,
                    alpha=alpha,
                    var=ordered[j],
                    raw_var=raw[j],
                    actual=actual,
                    breach=actual < ordered[j],
                    raw_crossing=crossed,
                    train_start=train.index[0],
                    train_end=train.index[-1],
                    n_train=len(train),
                    fit_seconds=elapsed,
                )
            )
        if progress and ((i - start + 1) % 100 == 0 or i == stop - 1):
            progress(i - start + 1, stop - start)
    return pd.DataFrame(records)
