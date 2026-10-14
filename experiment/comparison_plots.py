"""Figures from completed comparison tables."""

from . import plots as plots  # configure non-interactive backend before pyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_scores(scores, output):
    models = list(scores.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    for ax, (alpha, group) in zip(axes, scores.groupby("alpha", sort=True)):
        values = group.set_index("model").loc[models]
        ax.barh(models, values.pinball_loss * 10000, color="#3366A1")
        ax.set_title(f"{alpha:.0%} lower tail")
        ax.set_xlabel("Mean pinball loss × 10,000")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "loss_comparison.png", dpi=180)
    plt.close(fig)


def plot_quantiles(predictions, output):
    import pandas as pd

    fig, ax = plt.subplots(figsize=(11, 3.2), layout="constrained")
    one = predictions.drop_duplicates("date")
    ax.plot(
        pd.to_datetime(one.date),
        one.actual,
        color="#9CA6AF",
        lw=0.6,
        label="Observed return",
    )
    for (alpha, p), color in zip(
        predictions.groupby("alpha"), ["#BC8B26", "#3366A1", "#7A64A3"]
    ):
        ax.plot(
            pd.to_datetime(p.date),
            p["var"],
            color=color,
            lw=0.85,
            label=f"{alpha:.0%} VaR",
        )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.set(
        xlabel="Target trading day",
        ylabel="Daily log return",
        title="GRU: three predicted lower-tail cutoffs",
    )
    ax.grid(alpha=0.2)
    fig.savefig(output / "gru_forecasts.png", dpi=180)
    plt.close(fig)
