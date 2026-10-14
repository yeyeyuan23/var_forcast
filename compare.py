"""Read explicit completed-run paths from config/comparison.yaml; never trains models."""

from pathlib import Path
import sys
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from var_forcast.experiment.comparison import compare_group

DEFAULT_COMPARISON = Path(__file__).resolve().parent / "config/comparison.yaml"


def main(path=DEFAULT_COMPARISON):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text())
    output = (path.parent / settings["output"]).resolve()
    for name, group in settings["groups"].items():
        runs = {
            label: (path.parent / value).resolve() for label, value in group.items()
        }
        compare_group(
            runs,
            output / name,
            settings["block_length"],
            settings["bootstrap_draws"],
            settings["seed"],
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_snapshot.yaml").write_text(path.read_text())
    print(output)
    return output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Edit config/comparison.yaml; no CLI flags.")
    main()
