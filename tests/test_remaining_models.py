"""新增模型的定义、预测边界与配置契约。"""

import numpy as np
import pytest
from var_forcast.config import Config
from var_forcast.experiment.data import make_supervised
from var_forcast.experiment.backtest import rolling_backtest
from var_forcast.tests.test_research import sample_data


@pytest.mark.parametrize("model", ["garch", "transformer_qr"])
def test_remaining_model_forecast_is_finite_and_ignores_future(model):
    raw = sample_data(100)
    config = Config(model=model, window=60, lookback=5, hidden=8, epochs=2)
    table = make_supervised(raw)
    first = rolling_backtest(table, 80, 81, config)
    changed = raw.copy()
    changed.loc[table.index[80] :, "log_ret"] = -0.9
    changed.loc[table.index[80] :, ["rv5", "bv"]] *= 100
    other = rolling_backtest(make_supervised(changed), 80, 81, config)
    prefix = rolling_backtest(
        make_supervised(raw.loc[: table.index[80]]), 80, 81, config
    )
    np.testing.assert_allclose(first["var"], other["var"], rtol=0, atol=0)
    np.testing.assert_allclose(first["var"], prefix["var"], rtol=0, atol=0)
    assert np.isfinite(first["var"]).all()


def test_student_t_cutoff_is_standardized_to_unit_variance():
    from scipy.stats import t
    from var_forcast.models.garch import innovation_quantiles

    a = (0.01, 0.05, 0.1)
    np.testing.assert_allclose(
        innovation_quantiles(a, "t", 6), t.ppf(a, 6) * np.sqrt(4 / 6)
    )
    with pytest.raises(ValueError):
        innovation_quantiles(a, "t", 2)


def test_garch_one_step_forecast_matches_variance_recursion():
    from var_forcast.models.garch import fit_garch

    y = sample_data(250).log_ret.to_numpy() * 100
    fitted = fit_garch(y, "normal")
    p = fitted.params
    expected = (
        p["omega"]
        + p["alpha[1]"] * fitted.resid[-1] ** 2
        + p["beta[1]"] * fitted.conditional_volatility[-1] ** 2
    )
    actual = fitted.forecast(horizon=1, reindex=False).residual_variance.values[-1, 0]
    assert actual == pytest.approx(expected)


def test_transformer_records_order_with_positional_encoding():
    import torch
    from var_forcast.models.transformer_qr import QuantileTransformer

    torch.manual_seed(42)
    net = QuantileTransformer(3, 8, 2, 3).eval()
    x = torch.randn(1, 5, 3)
    permuted = x[:, [3, 2, 1, 0, 4], :]
    assert not torch.allclose(net(x), net(permuted))


def test_invalid_transformer_width_fails_before_training():
    with pytest.raises(ValueError):
        Config(model="transformer_qr", hidden=7, heads=2)
