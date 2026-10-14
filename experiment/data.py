"""第一步：读数据，并把每一天的特征与预测目标对齐。"""

import numpy as np
import pandas as pd
from ..config import DEFAULT_DATA

FEATURES = ["ret_lag1", "log_rv5_lag1", "log_bv_lag1"]


def load_data(path=DEFAULT_DATA):
    """读取原始小数单位的收益率；数据不合格时停止，不悄悄删行。"""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="raise")
    required = ["log_ret", "rv5", "bv"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"需要以下列：{required}")
    df = df[required].astype(float)
    if len(df) < 3 or df.index.isna().any():
        raise ValueError("数据太少或存在缺失日期")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("日期必须唯一且按从早到晚排列")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("数据存在缺失值或无穷值，请先检查来源")
    if (df[["rv5", "bv"]] <= 0).any().any():
        raise ValueError("对数特征要求 rv5 和 bv 为正，请先检查零值或负值")
    return df


def make_supervised(df):
    """行日期为待预测日 s：X_s 使用 s-1 日数据，y_s 为 s 日收益率。

    注意 shift(1) 是上一条交易日记录，不是前一个自然日。
    只使用当天已知的指标；不对未确认单位的 rv5/bv 开平方或年化。
    """
    table = pd.DataFrame(index=df.index)
    table["ret_lag1"] = df["log_ret"].shift(1)
    table["log_rv5_lag1"] = np.log(df["rv5"]).shift(1)
    table["log_bv_lag1"] = np.log(df["bv"]).shift(1)
    table["y"] = df["log_ret"]
    # 只有第一行因没有上一交易日而无法构造特征。
    return table.iloc[1:].copy()


def make_feature_sequences(table, lookback):
    """按时间从早到晚拼成 (样本, 天数, 指标)；第 k 个序列对应表第 k+L-1 行的标签。

    监督表的每行特征已经滞后一天，因此序列最后一天严格早于对应标签日。
    此函数只排列数据，不拟合标准化参数。
    """
    if type(lookback) is not int or not 1 <= lookback <= len(table):
        raise ValueError("lookback 必须为正整数且不能超过已有历史")
    features = table[FEATURES].to_numpy()
    windows = np.lib.stride_tricks.sliding_window_view(features, lookback, axis=0)
    return windows.transpose(0, 2, 1).copy()
