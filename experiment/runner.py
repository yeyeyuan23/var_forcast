"""Execute one explicitly selected configuration on a chronological target interval."""

from time import perf_counter
from .data import load_data, make_supervised
from .backtest import rolling_backtest
from .metrics import evaluate
from .artifacts import create_run_directory, make_manifest, save_manifest


def execute(
    experiment,
    start=None,
    stop=None,
    role="exploratory evaluation; previously viewed period",
    selection=None,
):
    config = experiment.config
    data = load_data(experiment.data)
    table = make_supervised(data)
    stop = len(table) if stop is None else stop
    start = stop - experiment.test_size if start is None else start
    extra_history = (
        config.lookback - 1
        if config.model in ("mlp_qr", "gru_qr", "transformer_qr")
        else 0
    )
    if start < config.window + extra_history or not start < stop <= len(table):
        raise ValueError(
            "test_size 太大：必须为训练窗口、序列输入和最初滞后留出足够数据"
        )
    output = create_run_directory(experiment)
    manifest = make_manifest(experiment, data, table, start, stop)
    manifest["evaluation_role"] = role
    if selection is not None:
        manifest["selection"] = selection
        manifest["hyperparameter_selection"] = (
            "earlier rolling validation; mean pinball loss across tail levels"
        )
    save_manifest(output, manifest)
    print(
        f"本次只运行：{config.model}；窗口 {config.window}；逐日预测 {stop - start} 日",
        flush=True,
    )
    print(
        f"预测日期：{table.index[start].date()} 至 {table.index[stop - 1].date()}",
        flush=True,
    )
    print(f"本次结果目录：{output}", flush=True)
    before = perf_counter()
    try:
        predictions = rolling_backtest(
            table,
            start,
            stop,
            config,
            progress=lambda n, total: print(
                f"已完成 {n}/{total} 日（{perf_counter() - before:.1f} 秒）", flush=True
            ),
        )
        manifest["forecast_seconds"] = perf_counter() - before
        scores = evaluate(predictions)
        predictions.to_csv(output / "predictions.csv", index=False)
        scores.to_csv(output / "metrics.csv", index=False)
        table.head(10).to_csv(output / "aligned_data_preview.csv")
        from var_forcast.experiment.plots import save_plots

        save_plots(predictions, scores, output)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = perf_counter() - before
        save_manifest(output, manifest)
    print(
        scores[["model", "alpha", "failure_rate", "pinball_loss"]].to_string(
            index=False
        ),
        flush=True,
    )
    print(f"完成：{output}", flush=True)
    return output
