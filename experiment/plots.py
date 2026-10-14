"""第五步：把原始数据、预测及评估变成可以核对的静态图。"""

from pathlib import Path
import os
import tempfile

# 避免依赖用户主目录的可写字体缓存；在无图形界面的环境也能运行。
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "var-learning-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from ..models import LABELS

COLORS = {
    "historical": "#3366A1",
    "linear_qr": "#708044",
    "mlp_qr": "#C16884",
    "gru_qr": "#7A64A3",
}
HATCHES = {"historical": "", "linear_qr": "..", "mlp_qr": "xx", "gru_qr": "//"}


def prepare_axes(axes):
    for ax in np.asarray(axes).ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#333B45")


def save_plots(predictions, scores, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#26313D",
            "text.color": "#26313D",
        }
    )
    first, last = predictions["date"].min(), predictions["date"].max()
    period = f"{first:%Y-%m-%d} to {last:%Y-%m-%d} | {predictions.date.nunique():,} trading days"

    # 同一纵轴范围，避免不同模型因缩放不同而显得更好。
    models = list(predictions.model.unique())
    selected_alpha = (
        0.05 if 0.05 in scores.alpha.values else float(scores.alpha.iloc[0])
    )
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(12, 2.5 * len(models) + 1),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    fig.suptitle(
        f"One-day-ahead {selected_alpha:.0%} VaR forecasts\n{period}",
        fontsize=16,
        fontweight="bold",
    )
    for ax, model in zip(np.atleast_1d(axes), models):
        p = predictions[
            (predictions.model == model) & (predictions.alpha == selected_alpha)
        ]
        ax.plot(p.date, p.actual, color="#7C8793", lw=0.65, label="Observed log return")
        ax.plot(
            p.date,
            p["var"],
            color=COLORS[model],
            lw=1.25,
            label=f"Forecast {selected_alpha:.0%} VaR",
        )
        breaches = p[p.breach]
        ax.scatter(
            breaches.date,
            breaches.actual,
            s=20,
            marker="x",
            color="#A65732",
            label="Return below VaR",
            zorder=3,
        )
        window = int(p.n_train.iloc[0])
        ax.set_title(
            f"{LABELS[model]} | W={window} | {len(breaches)} breaches / {len(p)} days",
            loc="left",
            fontsize=11,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Log return")
    np.atleast_1d(axes)[0].legend(loc="upper left", ncol=3, fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel(
        "Target trading day | predictions use only earlier observations"
    )
    prepare_axes(axes)
    fig.savefig(output / "var_forecasts.png", dpi=150)
    plt.close(fig)

    # 比较图：覆盖是否接近目标 + 分位数损失，而不是以突破越少越好。
    alphas = sorted(scores.alpha.unique())
    x = np.arange(len(alphas))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    fig.suptitle(f"Model diagnostics\n{period}", fontsize=16, fontweight="bold")
    for j, model in enumerate(models):
        s = scores[scores.model == model].set_index("alpha").loc[alphas]
        positions = x + (j - (len(models) - 1) / 2) * width
        style = dict(
            width=width * 0.93,
            color=COLORS[model],
            edgecolor="#26313D",
            linewidth=0.4,
            hatch=HATCHES[model],
            label=LABELS[model],
        )
        bars = axes[0].bar(positions, s.failure_rate, **style)
        axes[0].bar_label(
            bars, labels=[f"{v:.1%}" for v in s.failure_rate], padding=3, fontsize=9
        )
        bars = axes[1].bar(positions, s.pinball_loss * 10000, **style)
        axes[1].bar_label(
            bars,
            labels=[f"{v:.2f}" for v in s.pinball_loss * 10000],
            padding=3,
            fontsize=9,
        )
    for i, a in enumerate(alphas):
        axes[0].hlines(a, i - 0.44, i + 0.44, color="#26313D", ls="--", lw=1.4)
    axes[0].set_ylabel("Observed breach rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title(
        "Dashed lines: target tail probability. Fewer breaches is not always better.",
        fontsize=10,
    )
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    axes[0].set_ylim(0, max(scores.failure_rate.max(), max(alphas)) * 1.3)
    axes[1].set_ylabel("Mean pinball loss × 10,000")
    axes[1].set_title(
        "Lower loss is better within the same tail level; scores alone do not establish significance.",
        fontsize=10,
    )
    axes[1].set_ylim(0, scores.pinball_loss.max() * 10000 * 1.3)
    axes[1].set_xticks(x, [f"{a:.0%} tail" for a in alphas])
    axes[1].set_xlabel(
        "Source: supplied spy_data.csv | Initial fixed settings, no test-set tuning"
    )
    prepare_axes(axes)
    fig.savefig(output / "model_comparison.png", dpi=150)
    plt.close(fig)
