"""一次运行的目录、配置快照、环境记录。"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import tempfile

from ..config import MODEL_FIELDS
from .data import FEATURES


def create_run_directory(experiment):
    parent = experiment.output / experiment.config.model
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    output = Path(tempfile.mkdtemp(prefix=stamp, dir=parent))
    (output / "config_snapshot.yaml").write_text(
        experiment.source_text, encoding="utf-8"
    )
    (output / "model_config_snapshot.yaml").write_text(
        experiment.model_source_text, encoding="utf-8"
    )
    return output


def make_manifest(experiment, data, table, start):
    model = experiment.config.model
    dependencies = ["numpy", "pandas", "scipy", "PyYAML", "matplotlib"]
    if model == "linear_qr":
        dependencies.append("scikit-learn")
    if model in ("mlp_qr", "gru_qr"):
        dependencies.append("torch")
    config = asdict(experiment.config)
    config = {key: config[key] for key in ("model", "alphas", *MODEL_FIELDS[model])}
    root = Path(__file__).resolve().parents[1]
    return dict(
        status="running",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
        config_path=str(experiment.source),
        model_config_path=str(experiment.model_source),
        data_path=str(experiment.data),
        data_sha256=hashlib.sha256(experiment.data.read_bytes()).hexdigest(),
        data_start=str(data.index[0].date()),
        data_end=str(data.index[-1].date()),
        n_raw=len(data),
        n_supervised=len(table),
        n_test=len(table) - start,
        test_start=str(table.index[start].date()),
        test_end=str(table.index[-1].date()),
        features=[] if model == "historical" else FEATURES,
        input_lookback=experiment.config.lookback
        if model in ("mlp_qr", "gru_qr")
        else 1,
        refit="daily, from scratch; fixed-size trailing window",
        target="next trading-day log-return quantiles, decimal units",
        quantile_postprocessing="increasing rearrangement; raw outputs retained",
        evaluation_role="exploratory; previously viewed evaluation period",
        hyperparameter_selection="fixed starting settings; no validation search in this run",
        python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in dependencies},
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        },
        config_sha256=hashlib.sha256(experiment.source_text.encode()).hexdigest(),
        model_config_sha256=hashlib.sha256(
            experiment.model_source_text.encode()
        ).hexdigest(),
    )


def save_manifest(output, manifest):
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
