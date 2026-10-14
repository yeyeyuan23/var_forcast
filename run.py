"""单模型入口：修改 YAML 后运行 python -m var_forcast.run；不接受命令行参数。"""

from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from var_forcast.config import DEFAULT_CONFIG, load_experiment
from var_forcast.experiment.data import load_data, make_supervised
from var_forcast.experiment.backtest import rolling_backtest
from var_forcast.experiment.metrics import evaluate
from var_forcast.experiment.artifacts import (
    create_run_directory,
    make_manifest,
    save_manifest,
)


def main(config_path=DEFAULT_CONFIG):
    experiment = load_experiment(config_path)
    config = experiment.config
    data = load_data(experiment.data)
    table = make_supervised(data)
    start = len(table) - experiment.test_size
    extra_history = config.lookback - 1 if config.model == "gru_qr" else 0
    if start < config.window + extra_history:
        raise ValueError(
            "test_size 太大：必须为训练窗口、序列输入和最初滞后留出足够数据"
        )
    output = create_run_directory(experiment)
    manifest = make_manifest(experiment, data, table, start)
    save_manifest(output, manifest)
    print(
        f"本次只运行：{config.model}；窗口 {config.window}；逐日预测 {experiment.test_size} 日",
        flush=True,
    )
    print(
        f"预测日期：{table.index[start].date()} 至 {table.index[-1].date()}", flush=True
    )
    print(f"本次结果目录：{output}", flush=True)
    before = perf_counter()
    try:
        predictions = rolling_backtest(
            table,
            start,
            len(table),
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "请在 config/experiment.yaml 和 config/<模型>.yaml 中配置，不接受命令行参数。"
        )
    main()
