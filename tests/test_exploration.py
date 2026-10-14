"""探索入口独立于模型训练配置，搬目录后路径和统计结果仍正确。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from var_forcast.exploration.run import main, EXPLORATION_CONFIG


def test_default_exploration_paths_resolve_to_project_data_and_results():
    root = Path(__file__).resolve().parents[2]
    values = yaml.safe_load(EXPLORATION_CONFIG.read_text())
    assert (
        EXPLORATION_CONFIG.parent / values["data"]
    ).resolve() == root / "spy_data.csv"
    assert (
        EXPLORATION_CONFIG.parent / values["output"]
    ).resolve() == root / "results/exploration"


def test_exploration_runs_without_loading_experiment_configuration(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(41)
    data = pd.DataFrame(
        {
            "log_ret": rng.normal(0, 0.01, 70),
            "rv5": rng.uniform(0.00001, 0.0003, 70),
            "bv": rng.uniform(0.00001, 0.0002, 70),
        },
        index=pd.bdate_range("2020-01-01", periods=70),
    )
    data.to_csv(tmp_path / "data.csv")
    (tmp_path / "experiment.yaml").write_text("broken: [yaml")
    config = tmp_path / "exploration.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(data="data.csv", output="analysis", rolling_std_windows=[5, 10])
        )
    )
    monkeypatch.chdir(tmp_path.parent)
    main(config)
    result = tmp_path / "analysis"
    evidence = json.loads((result / "evidence.json").read_text())
    assert evidence["n"] == 70 and evidence["next_day_pair_count"] == 69
    rolling = pd.read_csv(result / "rolling_return_std.csv", index_col=0)
    np.testing.assert_allclose(
        rolling.std_5d.to_numpy(),
        data.log_ret.rolling(5).std().to_numpy(),
        equal_nan=True,
    )
    assert (result / "figures/06_next_day_relationship.png").exists()
