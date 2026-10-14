"""生成报告中的数据探索图表：在项目根目录运行 python -m var_forcast.exploration.run。

使用 config/exploration.yaml 中的 data 路径，描述全部给定观测，不拟合预测模型。
图表和统计表写到 results/exploration，原始数据保持原样。
"""

from pathlib import Path
import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from var_forcast.experiment.data import load_data

BLUE, GOLD, INK = "#3366A1", "#BC8B26", "#26313D"
EXPLORATION_CONFIG = Path(__file__).resolve().parents[1] / "config" / "exploration.yaml"


def load_settings(config_path):
    config_path = Path(config_path).resolve()
    settings = yaml.safe_load(config_path.read_text())
    if not isinstance(settings, dict) or set(settings) != {
        "data",
        "output",
        "rolling_std_windows",
    }:
        raise ValueError("探索配置必须包含 data、output、rolling_std_windows")
    for key in ("data", "output"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"{key} 必须是非空路径")
    source = (config_path.parent / settings["data"]).resolve()
    output = (config_path.parent / settings["output"]).resolve()
    return source, output, settings["rolling_std_windows"]


def validate_windows(windows, n):
    if (
        not isinstance(windows, list)
        or len(windows) != 2
        or any(type(w) is not int or not 2 <= w <= n for w in windows)
        or windows[0] >= windows[1]
    ):
        raise ValueError(
            "rolling_std_windows 必须是两个递增的整数，范围为 2 到样本数。"
        )


def save_statistics(df, extended, windows, output):
    """保存统计表，并返回后续图表使用的数据。"""
    n = len(df)
    r = df.log_ret
    summary = extended.describe(percentiles=[0.01, 0.05, 0.10, 0.5, 0.95, 0.99]).T
    summary["skewness"] = extended.skew()  # pandas 的偏差校正样本偏度
    summary["excess_kurtosis"] = extended.kurt()  # 正态分布基准为 0
    summary.to_csv(output / "descriptive_statistics.csv", index_label="variable")
    df.head(10).to_csv(output / "first_ten_rows.csv", index_label="date")

    rolling_std = pd.DataFrame(
        {f"std_{w}d": r.rolling(w, min_periods=w).std(ddof=1) for w in windows}
    )
    rolling_std.to_csv(output / "rolling_return_std.csv", index_label="date")
    tails = pd.DataFrame({"alpha": [0.01, 0.05, 0.10]})
    tails["empirical"] = r.quantile(tails.alpha).to_numpy()
    tails["normal_reference"] = r.mean() + r.std(ddof=1) * norm.ppf(tails.alpha)
    tails.to_csv(output / "return_quantiles.csv", index=False)

    # 日期严格对齐：t 日的指标配 t+1 日的绝对收益率，只用于描述关系。
    aligned = extended[["log_rv5", "log_bv"]].copy()
    aligned["next_abs_return"] = r.abs().shift(-1)
    aligned = aligned.dropna()
    aligned.to_csv(output / "next_day_pairs.csv", index_label="feature_date")
    correlations = {
        c: float(aligned[c].corr(aligned.next_abs_return))
        for c in ["log_rv5", "log_bv"]
    }

    # 这里定义为两段错开 k 个观测的序列的 Pearson 相关系数。
    # 不画显著性带：这些是描述统计，不在此处进行独立性假设检验。
    series = {
        "return": r,
        "absolute_return": r.abs(),
        "log_rv5": extended.log_rv5,
        "log_bv": extended.log_bv,
    }
    lag_corr = pd.DataFrame(
        {name: [s.autocorr(k) for k in range(1, 21)] for name, s in series.items()},
        index=range(1, 21),
    )
    lag_corr.index.name = "lag_observations"
    lag_corr["n_pairs"] = n - lag_corr.index.to_numpy()
    lag_corr.to_csv(output / "lag_correlations.csv")
    largest = pd.concat(
        [
            r.nsmallest(5).to_frame().assign(tail="lower"),
            r.nlargest(5).to_frame().assign(tail="upper"),
        ]
    )
    largest.to_csv(output / "extreme_returns.csv", index_label="date")

    return summary, rolling_std, aligned, correlations, lag_corr


def save_evidence(
    source, df, extended, windows, rolling_std, aligned, correlations, output
):
    """记录数据检查、计算口径和图表说明。"""
    n = len(df)
    r = df.log_ret
    evidence = {
        "data_path": str(source),
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n": n,
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "checks": {
            "missing_cells": int(df.isna().sum().sum()),
            "nonfinite_cells": int((~np.isfinite(df.to_numpy())).sum()),
            "invalid_dates": int(df.index.isna().sum()),
            "duplicate_dates": int(df.index.duplicated().sum()),
            "ascending_dates": bool(df.index.is_monotonic_increasing),
            "nonpositive_rv5_bv": int((df[["rv5", "bv"]] <= 0).sum().sum()),
            "exchange_calendar_completeness": "not independently verified",
        },
        "bv_above_rv5_count": int((df.bv > df.rv5).sum()),
        "rv5_bv_pearson": float(df.rv5.corr(df.bv)),
        "log_rv5_bv_pearson": float(extended.log_rv5.corr(extended.log_bv)),
        "next_abs_return_correlations": correlations,
        "next_day_pair_count": len(aligned),
        "beyond_three_sigma_count": int(((r - r.mean()).abs() > 3 * r.std()).sum()),
        "scope": "Full supplied sample; retrospective description, no holdout-performance claims.",
        "rolling_std": {
            "windows": windows,
            "ddof": 1,
            "alignment": "trailing, including current observation",
            "min_periods": "full window",
            "annualized": False,
            "choice": "Author-selected display scales; defaults approximate one month and one quarter; not optimized or assignment-mandated.",
            "valid_counts": {
                str(w): int(rolling_std[f"std_{w}d"].count()) for w in windows
            },
        },
        "statistics": "Sample std ddof=1; pandas adjusted skew/excess kurtosis; linear-interpolated quantiles; pairwise Pearson lag correlations.",
        "chart_contracts": [
            {
                "figure": "06_next_day_relationship",
                "question": "How do today's logged RV5/BV relate to next-day absolute returns?",
                "type": "two scatter panels",
                "rows": len(aligned),
                "alignment": "indicator t against absolute return t+1; all pairs, no trimming",
            },
            {
                "figure": "05_rolling_return_std",
                "question": "How does historical return dispersion vary over time at two display scales?",
                "type": "two trailing sample-standard-deviation lines",
                "rows": n,
                "windows": windows,
            },
            {
                "figure": "01_time_series",
                "question": "How do returns and activity vary over time?",
                "type": "three aligned line panels",
                "rows": n,
            },
            {
                "figure": "02_return_distribution",
                "question": "How does the return distribution compare with a fitted normal reference?",
                "type": "histogram and normal quantile plot",
                "rows": n,
            },
            {
                "figure": "03_rv_bv_distributions",
                "question": "How does taking logs change positive-variable skew?",
                "type": "four histograms",
                "rows": n,
            },
            {
                "figure": "04_lag_correlations",
                "question": "Is magnitude more persistent than signed return?",
                "type": "lag-correlation dot and line panels",
                "rows": "N-k pairs at each lag",
            },
        ],
        "chart_style": "Static research figures for PDF; blue/gold plus neutral references; solid/dashed and dot/square distinguish series; inspect final PDF.",
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def save_figure(fig, figures, name):
    fig.savefig(figures / f"{name}.png", dpi=190, facecolor="white")
    fig.savefig(figures / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def plot_rolling_std(df, rolling_std, windows, figures):
    fig, ax = plt.subplots(figsize=(8, 2.3), layout="constrained")
    ax.set_title("Trailing sample standard deviation of daily log returns", fontsize=11)
    for w, color, style in zip(windows, [BLUE, GOLD], ["-", "--"]):
        ax.plot(
            df.index,
            100 * rolling_std[f"std_{w}d"],
            color=color,
            ls=style,
            linewidth=1.1,
            label=f"{w} observed trading days",
        )
    ax.set(xlabel="Window end date", ylabel="Daily std (%)", ylim=(0, None))
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, figures, "05_rolling_return_std")


def plot_time_series(df, scope, figures):
    r = df.log_ret
    fig, ax = plt.subplots(3, 1, figsize=(8, 5.2), sharex=True, layout="constrained")
    fig.suptitle("Daily SPY observations\n" + scope, fontsize=12)
    ax[0].plot(df.index, r * 100, color=BLUE, linewidth=0.6)
    ax[0].axhline(0, color=INK, linewidth=0.5)
    ax[0].set_ylabel("Log return (%)")
    for a, name in zip(ax[1:], ["rv5", "bv"]):
        a.plot(df.index, df[name], color=BLUE, linewidth=0.6)
        a.set_yscale("log")
        a.set_ylabel(f"{name.upper()}\n(log axis)")
    for a in ax:
        a.grid(axis="y", alpha=0.18)
    ax[-1].set_xlabel("Date; RV5/BV retain their supplied units")
    save_figure(fig, figures, "01_time_series")


def plot_return_distribution(r, scope, figures):
    n = len(r)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
    fig.suptitle("Distribution of daily log returns\n" + scope, fontsize=12)
    rp = r * 100
    grid = np.linspace(rp.min(), rp.max(), 600)
    ax[0].hist(rp, bins=75, density=True, color=BLUE, edgecolor="white", linewidth=0.3)
    ax[0].plot(
        grid,
        norm.pdf(grid, rp.mean(), rp.std()),
        color=GOLD,
        ls="--",
        lw=1.5,
        label="Fitted normal",
    )
    ax[0].set(xlabel="Log return (%)", ylabel="Density per percentage point")
    ax[0].legend(fontsize=8)
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = rp.mean() + rp.std() * norm.ppf(p)
    ax[1].scatter(theoretical, np.sort(rp), s=5, color=BLUE, alpha=0.65)
    limits = [min(theoretical.min(), rp.min()), max(theoretical.max(), rp.max())]
    ax[1].plot(limits, limits, "--", color=INK, lw=1)
    ax[1].set(
        xlabel="Fitted-normal quantiles (%)",
        ylabel="Observed log-return quantiles (%)",
        xlim=limits,
        ylim=limits,
    )
    for a in ax:
        a.grid(alpha=0.18)
    save_figure(fig, figures, "02_return_distribution")


def plot_indicator_distributions(df, scope, figures):
    fig, ax = plt.subplots(2, 2, figsize=(8, 5.2), layout="constrained")
    fig.suptitle(
        "RV5 and BV: original and log-transformed values\n" + scope, fontsize=12
    )
    for row, name in enumerate(["rv5", "bv"]):
        for col in range(2):
            values = df[name] * 1e4 if col == 0 else np.log(df[name])
            a = ax[row, col]
            counts, edges, _ = a.hist(
                values, bins=60, color=BLUE, edgecolor="white", linewidth=0.4
            )
            a.set_title(
                f"{name.upper() if col == 0 else 'ln(' + name.upper() + ')'} | skew = {values.skew():.2f}"
            )
            if col == 0:
                a.set_yscale("log")
                a.set(
                    xlabel=f"{name.upper()} x 10,000 (full range)",
                    ylabel="Count (log axis)",
                )
            else:
                # 直方图纵轴为计数：正态密度乘以观测数和箱宽，才能在同一尺度比较。
                grid = np.linspace(values.min(), values.max(), 600)
                reference = norm.pdf(grid, values.mean(), values.std(ddof=1))
                a.plot(
                    grid,
                    reference * len(values) * (edges[1] - edges[0]),
                    color=GOLD,
                    ls="--",
                    lw=1.5,
                    label="Fitted normal",
                )
                a.set(xlabel=f"Natural log of supplied {name.upper()}", ylabel="Count")
                a.legend(fontsize=8)
            a.grid(axis="y", alpha=0.18)
    save_figure(fig, figures, "03_rv_bv_distributions")


def plot_next_day_relationship(aligned, correlations, figures):
    # 每个点是一对相邻交易日：横轴 t 日指标，纵轴 t+1 日绝对收益。
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, layout="constrained")
    for a, name, label in zip(ax, ["log_rv5", "log_bv"], ["RV5", "BV"]):
        a.scatter(
            aligned[name],
            100 * aligned.next_abs_return,
            s=8,
            color=BLUE,
            alpha=0.22,
            edgecolors="none",
            rasterized=True,
        )
        a.set_title(f"{label} | Pearson r = {correlations[name]:.3f}", fontsize=10)
        a.set_xlabel(f"ln({label}) on day t")
        a.set_ylim(bottom=0)
        a.grid(alpha=0.18)
    ax[0].set_ylabel("Absolute log return on t+1 (%)")
    save_figure(fig, figures, "06_next_day_relationship")


def plot_lag_correlations(lag_corr, scope, figures):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True, layout="constrained")
    fig.suptitle(
        "Sample lag correlations\n" + scope + " | N-k pairs at lag k", fontsize=12
    )
    for a, names, labels in zip(
        ax,
        [("return", "absolute_return"), ("log_rv5", "log_bv")],
        [("Return", "Absolute return"), ("ln(RV5)", "ln(BV)")],
    ):
        for name, label, color, style, marker in zip(
            names, labels, [BLUE, GOLD], ["-", "--"], ["o", "s"]
        ):
            a.plot(
                lag_corr.index,
                lag_corr[name],
                label=label,
                color=color,
                ls=style,
                marker=marker,
                markersize=3,
                lw=1.2,
            )
        a.axhline(0, color=INK, lw=0.7)
        a.set(
            xlabel="Lag in observed trading days",
            xticks=[1, 5, 10, 15, 20],
            ylim=(-0.2, 1),
        )
        a.grid(alpha=0.18)
        a.legend(fontsize=9)
    ax[0].set_ylabel("Pearson correlation")
    save_figure(fig, figures, "04_lag_correlations")


def main(config_path=EXPLORATION_CONFIG):
    source, output, windows = load_settings(config_path)
    df = load_data(source)
    validate_windows(windows, len(df))
    extended = df.assign(log_rv5=np.log(df.rv5), log_bv=np.log(df.bv))
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary, rolling_std, aligned, correlations, lag_corr = save_statistics(
        df, extended, windows, output
    )
    save_evidence(
        source, df, extended, windows, rolling_std, aligned, correlations, output
    )
    configure_plot_style()
    scope = f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} | N = {len(df):,}"
    plot_rolling_std(df, rolling_std, windows, figures)
    plot_time_series(df, scope, figures)
    plot_return_distribution(df.log_ret, scope, figures)
    plot_indicator_distributions(df, scope, figures)
    plot_next_day_relationship(aligned, correlations, figures)
    plot_lag_correlations(lag_corr, scope, figures)
    print(summary.round(6).to_string())
    print(f"\nEvidence and figures saved to: {output}")


if __name__ == "__main__":
    main()
