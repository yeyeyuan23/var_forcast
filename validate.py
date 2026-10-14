"""Rolling validation of ONE model; saves a selected YAML, never evaluates the final period."""

from pathlib import Path
import sys
from dataclasses import asdict, replace
from itertools import product
import json
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from var_forcast.config import (
    DEFAULT_CONFIG,
    MODEL_FIELDS,
    NEURAL_MODELS,
    load_experiment,
    check_keys,
    integer,
)
from var_forcast.experiment.data import load_data, make_supervised
from var_forcast.experiment.runner import execute
from var_forcast.experiment.artifacts import create_run_directory

DEFAULT_VALIDATION = DEFAULT_CONFIG.with_name("validation.yaml")


def candidates(experiment, design):
    check_keys(
        design,
        ("validation_size", "windows", "lookbacks", "distributions"),
        "validation.yaml",
    )
    integer(design["validation_size"], "validation_size", 2)
    for field, minimum in [("windows", 5), ("lookbacks", 1)]:
        if not isinstance(design[field], list) or not design[field]:
            raise ValueError(f"{field} must be a non-empty list")
        for value in design[field]:
            integer(value, field, minimum)
        if len(set(design[field])) != len(design[field]):
            raise ValueError(f"{field} must not repeat values")
    if (
        not isinstance(design["distributions"], list)
        or not design["distributions"]
        or any(x not in ("normal", "t") for x in design["distributions"])
    ):
        raise ValueError("distributions must contain normal or t")
    model = experiment.config.model
    lengths = design["lookbacks"] if model in NEURAL_MODELS else [1]
    distributions = (
        design["distributions"]
        if model == "garch"
        else [experiment.config.distribution]
    )
    for window, length, distribution in product(
        design["windows"], lengths, distributions
    ):
        yield replace(
            experiment.config, window=window, lookback=length, distribution=distribution
        )


def with_parameters(experiment, config, output=None):
    values = asdict(config)
    model_text = yaml.safe_dump(
        {k: values[k] for k in MODEL_FIELDS[config.model]}, sort_keys=False
    )
    return replace(
        experiment,
        config=config,
        model_source_text=model_text,
        output=experiment.output if output is None else output,
    )


def main(config_path=DEFAULT_CONFIG, validation_path=DEFAULT_VALIDATION):
    experiment = load_experiment(config_path)
    design = yaml.safe_load(Path(validation_path).read_text())
    choices = list(candidates(experiment, design))
    table = make_supervised(load_data(experiment.data))
    stop = len(table) - experiment.test_size
    start = stop - design["validation_size"]
    for config in choices:
        if start < config.window + config.lookback - 1:
            raise ValueError("Not enough pre-validation history for every candidate")
    output = create_run_directory(
        replace(experiment, output=experiment.output / "validation")
    )
    (output / "validation_snapshot.yaml").write_text(yaml.safe_dump(design))
    rows = []
    for index, config in enumerate(choices):
        candidate = with_parameters(experiment, config, output / "candidates")
        run = execute(
            candidate,
            start,
            stop,
            role="rolling validation; used for parameter selection",
        )
        scores = pd.read_csv(run / "metrics.csv")
        rows.append(
            dict(
                candidate=index,
                window=config.window,
                lookback=config.lookback,
                distribution=config.distribution,
                score=float(scores.pinball_loss.mean()),
                n_validation=stop - start,
                run_path=str(run),
            )
        )
        pd.DataFrame(rows).to_csv(output / "trials.csv", index=False)
    # Stable tie-break: candidate order declared in the design, never final-period results.
    winner = min(rows, key=lambda row: row["score"])
    selected = choices[winner["candidate"]]
    selected_experiment = with_parameters(experiment, selected)
    (output / "selected_model.yaml").write_text(selected_experiment.model_source_text)
    selection = dict(
        model=selected.model,
        criterion="equal mean of three tail-specific pinball losses",
        selected_parameters=yaml.safe_load(selected_experiment.model_source_text),
        validation_start=str(table.index[start].date()),
        validation_end=str(table.index[stop - 1].date()),
        evaluation_start=str(table.index[stop].date()),
        score=winner["score"],
        selected_validation_run=winner["run_path"],
        note="Limited prespecified grid. Final-period results were previously viewed; exploratory study.",
    )
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected configuration saved to {output}", flush=True)
    return output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Edit config/experiment.yaml and config/validation.yaml; no CLI flags."
        )
    main()
