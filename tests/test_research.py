"""重点防止时间泄漏、损失方向错误和错误的统计结论。"""

import numpy as np
import pandas as pd
import pytest

from var_forcast.experiment.data import load_data, make_supervised
from var_forcast.models import Config, forecast
from var_forcast.experiment.backtest import rolling_backtest
from var_forcast.experiment.metrics import pinball_loss, coverage_tests, evaluate


def sample_data(n=60):
    rng = np.random.default_rng(17)
    return pd.DataFrame(
        {
            "log_ret": rng.normal(0, 0.01, n),
            "rv5": rng.uniform(0.00001, 0.0003, n),
            "bv": rng.uniform(0.00001, 0.0002, n),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def test_lags_use_previous_day_but_target_is_current_day():
    raw = sample_data()
    table = make_supervised(raw)
    row = table.iloc[0]
    assert row.name == raw.index[1]
    assert row["y"] == raw.iloc[1].log_ret
    assert row["ret_lag1"] == raw.iloc[0].log_ret
    assert row["log_rv5_lag1"] == pytest.approx(np.log(raw.iloc[0].rv5))
    assert row["log_bv_lag1"] == pytest.approx(np.log(raw.iloc[0].bv))


@pytest.mark.parametrize("method", ["historical", "linear_qr", "mlp_qr", "gru_qr"])
def test_future_values_and_prefix_length_cannot_change_forecast(method):
    raw = sample_data()
    table = make_supervised(raw)
    config = Config(
        window=30, model=method, epochs=4, lookback=5 if method == "gru_qr" else 1
    )
    first = rolling_backtest(table, 40, 41, config)
    changed = raw.copy()
    # 改变待预测日及其之后的所有数据，预测应完全不变。
    changed.loc[table.index[40] :, "log_ret"] = -0.8
    changed.loc[table.index[40] :, ["rv5", "bv"]] *= 1000
    other = rolling_backtest(make_supervised(changed), 40, 41, config)
    prefix = make_supervised(raw.loc[: table.index[40]])
    truncated = rolling_backtest(prefix, 40, 41, config)
    np.testing.assert_allclose(first["var"], other["var"], rtol=0, atol=0)
    np.testing.assert_allclose(first["var"], truncated["var"], rtol=0, atol=0)
    assert (first["train_end"] < first["date"]).all()
    assert (first["n_train"] == 30).all()
    assert first.iloc[0]["train_start"] == table.index[10]
    assert first.iloc[0]["train_end"] == table.index[39]
    assert np.isfinite(first["var"]).all()
    assert np.all(np.diff(first["var"]) >= 0)


def test_baselines_have_hand_calculated_predictions():
    y = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    x = np.zeros((5, 3))
    config = Config(alphas=(0.25, 0.5, 0.75))
    h = forecast("historical", x, y, x[:1], config)
    np.testing.assert_allclose(h, [-0.01, 0, 0.01], atol=1e-14)


def test_pinball_penalizes_underestimated_left_tail_more():
    # alpha=0.05：真实收益低于预测 1 个单位，罚 0.95；反向罚 0.05。
    result = pinball_loss(np.array([-2.0, 0.0]), np.array([-1.0, -1.0]), 0.05)
    np.testing.assert_allclose(result, [0.95, 0.05])


def test_no_breaches_does_not_mean_perfect_calibration():
    result = coverage_tests(np.zeros(100, dtype=bool), 0.05)
    assert result["kupiec_lr"] == pytest.approx(-200 * np.log(0.95))
    assert result["kupiec_p"] < 0.01
    assert np.isnan(result["independence_p"])
    assert np.isnan(result["conditional_coverage_p"])
    assert "undefined" in result["test_note"]


def test_all_breaches_and_exact_binomial_boundary():
    result = coverage_tests(np.ones(10, dtype=bool), 0.05)
    assert np.isfinite(result["kupiec_lr"])
    assert result["binomial_exact_p"] == pytest.approx(0.05**10)
    assert np.isnan(result["independence_p"])


def test_independence_detects_clustering_not_only_total_count():
    dispersed = np.zeros(1000, dtype=bool)
    dispersed[::10] = True
    clustered = np.zeros(1000, dtype=bool)
    clustered[450:550] = True
    a = coverage_tests(dispersed, 0.1)
    b = coverage_tests(clustered, 0.1)
    assert a["kupiec_lr"] == pytest.approx(b["kupiec_lr"])
    assert b["independence_lr"] > a["independence_lr"] + 100
    assert b["conditional_coverage_p"] < 0.001


@pytest.mark.parametrize(
    "problem", ["duplicate", "unsorted", "missing", "infinite", "nonpositive"]
)
def test_invalid_data_is_rejected_not_silently_dropped(tmp_path, problem):
    raw = sample_data()
    if problem == "duplicate":
        raw.index = [raw.index[0]] * len(raw)
    if problem == "unsorted":
        raw = raw.iloc[::-1]
    if problem == "missing":
        raw.iloc[1, 0] = np.nan
    if problem == "infinite":
        raw.iloc[1, 0] = np.inf
    if problem == "nonpositive":
        raw.iloc[1, 1] = 0
    path = tmp_path / "bad.csv"
    raw.to_csv(path)
    with pytest.raises(ValueError):
        load_data(path)


def test_evaluation_breaches_and_common_dates():
    table = make_supervised(sample_data())
    pred = pd.concat(
        [
            rolling_backtest(table, 40, 44, Config(window=30, model=m))
            for m in ("historical", "linear_qr")
        ],
        ignore_index=True,
    )
    scores = evaluate(pred)
    assert len(scores) == 6
    assert (scores["n"] == 4).all()
    assert np.all(pred["breach"] == (pred["actual"] < pred["var"]))
    assert not pred.duplicated(["date", "model", "alpha"]).any()
    with pytest.raises(ValueError):
        evaluate(pred.iloc[1:])


def test_evaluation_rejects_a_whole_missing_model_quantile_group():
    table = make_supervised(sample_data())
    pred = pd.concat(
        [
            rolling_backtest(table, 40, 44, Config(window=30, model=m))
            for m in ("historical", "linear_qr")
        ],
        ignore_index=True,
    )
    incomplete = pred[~((pred.model == "linear_qr") & (pred.alpha == 0.01))]
    with pytest.raises(ValueError, match="分位数"):
        evaluate(incomplete)


@pytest.mark.parametrize("model,window", [("historical", 10), ("linear_qr", 20)])
def test_independent_models_use_their_own_windows_on_common_dates(model, window):
    table = make_supervised(sample_data())
    pred = rolling_backtest(table, 40, 42, Config(model=model, window=window))
    assert (pred.n_train == window).all()
    assert list(pred.date.unique()) == list(table.index[40:42])
    for i in [40, 41]:
        day = pred[pred.date == table.index[i]]
        assert (day.train_start == table.index[i - window]).all()
        assert (day.train_end == table.index[i - 1]).all()


def test_window_cannot_reach_before_available_history():
    table = make_supervised(sample_data())
    with pytest.raises(ValueError):
        rolling_backtest(table, 40, 42, Config(window=45))


@pytest.mark.parametrize(
    "update",
    [
        {"window": 0},
        {"window": 10.5},
        {"window": True},
        {"epochs": 0},
        {"hidden": 1.2},
        {"alphas": (0.05, 0.01)},
        {"alphas": (float("nan"),)},
        {"learning_rate": float("inf")},
        {"model": "normal"},
    ],
)
def test_invalid_parameters_fail_before_training(update):
    with pytest.raises(ValueError):
        Config(**update)


def test_old_cli_options_are_rejected_before_running():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "run.py"
    result = subprocess.run(
        [sys.executable, str(script), "--window", "250"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "config/experiment.yaml" in result.stderr
