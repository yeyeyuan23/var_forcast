"""Compare saved runs on identical targets; no model training or parameter selection."""

from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
from .metrics import evaluate, pinball_loss


def collect_predictions(runs):
    frames, reference = [], None
    for label, path in runs.items():
        path = Path(path)
        manifest = json.loads((path / "run_manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete run: {label}")
        frame = pd.read_csv(path / "predictions.csv").sort_values(["date", "alpha"])
        target = frame[["date", "alpha", "actual"]].reset_index(drop=True)
        if reference is not None and not target.equals(reference):
            raise ValueError(
                "All runs must have identical dates, tail levels and actual returns"
            )
        reference = target
        frame["model"] = label
        frames.append(frame)
    if not frames:
        raise ValueError("No runs supplied")
    return pd.concat(frames, ignore_index=True)


def paired_loss_interval(difference, block_length=20, draws=2000, seed=42):
    """Circular block bootstrap of paired daily loss differences; percentile interval."""
    values = np.asarray(difference, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or not 1 <= block_length <= len(values)
        or draws < 2
    ):
        raise ValueError("Invalid paired differences or bootstrap settings")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block_length))
    starts = rng.integers(0, len(values), size=(draws, blocks))
    indices = (starts[:, :, None] + np.arange(block_length)) % len(values)
    means = values[indices.reshape(draws, -1)[:, : len(values)]].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def compare_group(runs, output, block_length=20, draws=2000, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = collect_predictions(runs)
    scores = evaluate(predictions)
    scores.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    daily = predictions[["date", "alpha", "model"]].copy()
    daily["loss"] = pinball_loss(
        predictions.actual, predictions["var"], predictions.alpha
    )
    daily.to_csv(output / "daily_loss.csv", index=False)
    records = []
    scopes = [
        (str(a), d.pivot(index="date", columns="model", values="loss"))
        for a, d in daily.groupby("alpha")
    ]
    scopes.append(("mean", daily.groupby(["date", "model"]).loss.mean().unstack()))
    for scope, wide in scopes:
        for left, right in combinations(wide.columns, 2):
            estimate, low, high = paired_loss_interval(
                (wide[left] - wide[right]).to_numpy(), block_length, draws, seed
            )
            records.append(
                dict(
                    alpha=scope,
                    left=left,
                    right=right,
                    mean_difference=estimate,
                    lower_95=low,
                    upper_95=high,
                    block_length=block_length,
                    draws=draws,
                )
            )
    pd.DataFrame(records).to_csv(output / "paired_loss_intervals.csv", index=False)
    from .comparison_plots import plot_scores, plot_quantiles

    plot_scores(scores, output)
    if "GRU" in runs:
        plot_quantiles(predictions[predictions.model == "GRU"], output)
    return scores
