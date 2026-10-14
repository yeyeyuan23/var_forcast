"""Validation selections use only the dates before the evaluation interval."""

import json
import numpy as np
import pandas as pd
import pytest
import yaml
from var_forcast.tests.test_execution import write_experiment


def test_validation_excludes_final_period_and_selection_is_future_invariant(tmp_path):
    from var_forcast import validate

    config = write_experiment(tmp_path)
    design = tmp_path / "validation.yaml"
    design.write_text(
        yaml.safe_dump(
            dict(
                validation_size=8,
                windows=[10, 20],
                lookbacks=[1, 3],
                distributions=["normal", "t"],
            )
        )
    )
    first = validate.main(config, design)
    trials = pd.read_csv(first / "trials.csv")
    assert len(trials) == 2 and trials.n_validation.eq(8).all()
    source = pd.read_csv(tmp_path / "data.csv", index_col=0)
    expected_dates = source.index[-12:-4].tolist()
    for path in trials.run_path:
        predictions = pd.read_csv(path + "/predictions.csv")
        assert predictions.date.unique().tolist() == expected_dates
    before = json.loads((first / "selection.json").read_text())
    source.iloc[-4:, 0] = -0.99
    source.to_csv(tmp_path / "data.csv")
    second = validate.main(config, design)
    after = json.loads((second / "selection.json").read_text())
    assert before["selected_parameters"] == after["selected_parameters"]
    np.testing.assert_array_equal(
        trials.score, pd.read_csv(second / "trials.csv").score
    )


def test_validation_rejects_insufficient_history(tmp_path):
    from var_forcast import validate

    config = write_experiment(tmp_path)
    design = tmp_path / "validation.yaml"
    design.write_text(
        yaml.safe_dump(
            dict(validation_size=20, windows=[40], lookbacks=[1], distributions=["t"])
        )
    )
    with pytest.raises(ValueError):
        validate.main(config, design)


def test_manifest_does_not_claim_untuned_settings_without_selection_record(tmp_path):
    from var_forcast.config import load_experiment
    from var_forcast.experiment.data import load_data, make_supervised
    from var_forcast.experiment.artifacts import make_manifest

    experiment = load_experiment(write_experiment(tmp_path))
    data = load_data(experiment.data)
    table = make_supervised(data)
    manifest = make_manifest(experiment, data, table, len(table) - 4)
    assert (
        manifest["hyperparameter_selection"]
        == "not recorded; configuration snapshot retained"
    )
