import numpy as np
import pytest


def test_paired_block_interval_respects_sign_and_pairing():
    from var_forcast.experiment.comparison import paired_loss_interval

    values = np.full(100, -0.002)
    estimate, lower, upper = paired_loss_interval(
        values, block_length=10, draws=200, seed=1
    )
    np.testing.assert_allclose([estimate, lower, upper], [-0.002] * 3)
    assert paired_loss_interval(-values, 10, 200, 1)[0] > 0
    with pytest.raises(ValueError):
        paired_loss_interval(values, 0, 200, 1)


def test_comparison_rejects_different_dates_before_scoring(tmp_path):
    import pandas as pd
    import json
    from var_forcast.experiment.comparison import collect_predictions

    for name, date in [("a", "2020-01-01"), ("b", "2020-01-02")]:
        p = tmp_path / name
        p.mkdir()
        (p / "run_manifest.json").write_text(json.dumps({"status": "complete"}))
        pd.DataFrame(
            {
                "date": [date],
                "actual": [0.01],
                "var": [-0.01],
                "alpha": [0.05],
                "model": ["historical"],
                "raw_crossing": [False],
                "fit_seconds": [0.0],
            }
        ).to_csv(p / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="date|日期"):
        collect_predictions({"A": tmp_path / "a", "B": tmp_path / "b"})
