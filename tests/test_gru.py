"""GRU 的序列边界、训练窗口和实际运行契约。"""

import json
import yaml
import pandas as pd
import numpy as np
import pytest

from var_forcast.config import Config, load_experiment
from var_forcast.experiment import data, backtest
from var_forcast.tests.test_research import sample_data


def test_sequence_features_are_chronological_and_end_before_target():
    raw = sample_data(12)
    table = data.make_supervised(raw)
    assert hasattr(data, "make_feature_sequences")
    sequences = data.make_feature_sequences(table, 3)
    assert sequences.shape == (9, 3, 3)
    for k, sequence in enumerate(sequences):
        expected = raw.iloc[k : k + 3]
        np.testing.assert_allclose(sequence[:, 0], expected.log_ret)
        np.testing.assert_allclose(sequence[:, 1], np.log(expected.rv5))
        np.testing.assert_allclose(sequence[:, 2], np.log(expected.bv))
        assert raw.index[k + 2] < table.index[k + 2]


def test_gru_keeps_all_training_labels_and_extra_input_history(monkeypatch):
    table = data.make_supervised(sample_data())
    config = Config(model="gru_qr", window=30, lookback=5, epochs=2)
    observed = []

    def capture(method, X, y, x, settings):
        observed.append((X.copy(), y.copy(), x.copy()))
        return np.quantile(y, settings.alphas)

    monkeypatch.setattr(backtest, "forecast", capture)
    result = backtest.rolling_backtest(table, 40, 42, config)
    for i, (X, y, x) in zip([40, 41], observed):
        assert X.shape == (30, 5, 3) and x.shape == (1, 5, 3)
        np.testing.assert_array_equal(y, table.y.iloc[i - 30 : i])
        np.testing.assert_array_equal(X[0], table[data.FEATURES].iloc[i - 34 : i - 29])
        np.testing.assert_array_equal(X[-1], table[data.FEATURES].iloc[i - 5 : i])
        np.testing.assert_array_equal(x[0], table[data.FEATURES].iloc[i - 4 : i + 1])
    assert (result.n_train == 30).all()
    with pytest.raises(ValueError):
        backtest.rolling_backtest(table, 33, 34, config)


def test_gru_uses_only_training_sequence_statistics():
    from var_forcast.models.gru_qr import standardize_sequences

    X = np.arange(24, dtype=float).reshape(4, 2, 3)
    x = np.full((1, 2, 3), 999.0)
    transformed, next_x = standardize_sequences(X, x)
    np.testing.assert_allclose(transformed.mean(axis=(0, 1)), 0, atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=(0, 1)), 1)
    np.testing.assert_allclose(next_x, (x - X.mean(axis=(0, 1))) / X.std(axis=(0, 1)))
    constant, constant_next = standardize_sequences(
        np.ones((4, 2, 3)), np.ones((1, 2, 3))
    )
    assert np.isfinite(constant).all() and np.isfinite(constant_next).all()


def test_gru_end_to_end_loads_one_config_and_records_sequence_length(tmp_path):
    from var_forcast.run import main

    sample_data().to_csv(tmp_path / "data.csv")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            dict(
                model="gru_qr",
                test_size=2,
                alphas=[0.01, 0.05, 0.1],
                data="data.csv",
                output="results",
            )
        )
    )
    (tmp_path / "gru_qr.yaml").write_text(
        yaml.safe_dump(
            dict(
                window=30,
                lookback=5,
                epochs=2,
                hidden=4,
                learning_rate=0.01,
                weight_decay=0.001,
                seed=42,
            )
        )
    )
    (tmp_path / "mlp_qr.yaml").write_text("broken: [")
    experiment = load_experiment(config_path)
    assert experiment.config.lookback == 5
    output = main(config_path)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["input_lookback"] == manifest["config"]["lookback"] == 5
    assert "torch" in manifest["versions"]
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions.model.unique().tolist() == ["gru_qr"]
    assert len(predictions) == 6 and (predictions.n_train == 30).all()
    assert (output / "var_forecasts.png").exists()


@pytest.mark.parametrize("lookback", [0, 1.5, True])
def test_invalid_sequence_length_is_rejected(lookback):
    with pytest.raises(ValueError):
        Config(model="gru_qr", lookback=lookback)
