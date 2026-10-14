"""运行入口的约束：一次一个模型、独立配置、结果不覆盖。"""

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from var_forcast.config import load_experiment


def write_experiment(root, model="historical"):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{model}.yaml").write_text(yaml.safe_dump({"window": 30}))
    # 没选中的配置即使损坏，也不应阻止本次运行。
    (root / "mlp_qr.yaml").write_text("not: [valid")
    values = dict(
        model=model,
        test_size=4,
        alphas=[0.01, 0.05, 0.10],
        data="data.csv",
        output="results",
    )
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(values))
    rng = np.random.default_rng(17)
    pd.DataFrame(
        {
            "log_ret": rng.normal(0, 0.01, 50),
            "rv5": rng.uniform(0.00001, 0.0003, 50),
            "bv": rng.uniform(0.00001, 0.0002, 50),
        },
        index=pd.bdate_range("2020-01-01", periods=50),
    ).to_csv(root / "data.csv")
    return path


def test_reads_only_selected_model_configuration(tmp_path, monkeypatch):
    path = write_experiment(tmp_path / "experiment")
    monkeypatch.chdir(tmp_path)
    experiment = load_experiment(path)
    assert experiment.config.model == "historical"
    assert experiment.config.window == 30
    assert experiment.data == path.parent / "data.csv"
    assert experiment.model_source == path.parent / "historical.yaml"


def test_all_configuration_lives_in_one_directory():
    from var_forcast.config import DEFAULT_CONFIG, DEFAULT_DATA

    root = Path(__file__).resolve().parents[1]
    assert DEFAULT_CONFIG == root / "config/experiment.yaml"
    assert DEFAULT_DATA == root.parent / "spy_data.csv"
    assert not (root / "config.yaml").exists()
    assert not (root / "configs").exists()
    assert (root / "config/exploration.yaml").exists()


@pytest.mark.parametrize("model", ["all", ["historical", "mlp_qr"], "typo"])
def test_model_lists_and_all_are_rejected(tmp_path, model):
    path = write_experiment(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["model"] = model
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="model"):
        load_experiment(path)


def test_historical_import_does_not_import_neural_dependencies():
    command = "from var_forcast.models import forecast; import sys; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_each_run_is_single_model_and_preserves_previous_results(tmp_path):
    from var_forcast.run import main

    path = write_experiment(tmp_path)
    first = main(path)
    before = (first / "predictions.csv").read_bytes()
    second = main(path)
    assert first != second
    assert first.parent == second.parent == tmp_path / "results/historical"
    assert (first / "predictions.csv").read_bytes() == before
    frame = pd.read_csv(second / "predictions.csv")
    assert frame.model.unique().tolist() == ["historical"]
    assert len(frame) == 12
    assert (frame.train_end < frame.date).all()
    assert (second / "model_config_snapshot.yaml").exists()
    manifest = json.loads((second / "run_manifest.json").read_text())
    assert manifest["config"]["model"] == "historical"
    assert manifest["status"] == "complete"
    assert not (second / "summary.md").exists()


def test_default_outputs_are_outside_source_repository():
    root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(root / "config/experiment.yaml")
    assert experiment.output == root.parent / "results"
    assert not (root / "experiment/reporting.py").exists()
    assert not (root / "results").exists()
