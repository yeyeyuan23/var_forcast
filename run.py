"""Single-model evaluation entry. Settings live in config/; no CLI flags."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from var_forcast.config import DEFAULT_CONFIG, load_experiment
from var_forcast.experiment.runner import execute


def main(config_path=DEFAULT_CONFIG):
    return execute(load_experiment(config_path))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("请在 config/experiment.yaml 中配置，不接受命令行参数。")
    main()
