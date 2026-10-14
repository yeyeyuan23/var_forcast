"""一次实验只读取一个模型；公共设置和模型超参数分别放在 YAML 中。"""

from dataclasses import dataclass
from pathlib import Path
import math
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().with_name("experiment.yaml")
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "spy_data.csv"
METHODS = ("historical", "linear_qr", "mlp_qr", "gru_qr")
LABELS = {
    "historical": "Historical",
    "linear_qr": "Linear QR",
    "mlp_qr": "MLP QR",
    "gru_qr": "GRU QR",
}
MODEL_FIELDS = {
    "gru_qr": (
        "window",
        "lookback",
        "epochs",
        "hidden",
        "learning_rate",
        "weight_decay",
        "seed",
    ),
    "historical": ("window",),
    "linear_qr": ("window", "l1_penalty"),
    "mlp_qr": (
        "window",
        "lookback",
        "epochs",
        "hidden",
        "learning_rate",
        "weight_decay",
        "seed",
    ),
}


def integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是至少为 {minimum} 的整数")


def number(value, name, minimum, strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} 超出允许范围")


@dataclass(frozen=True)
class Config:
    model: str = "historical"
    window: int = 1000
    alphas: tuple = (0.01, 0.05, 0.10)
    epochs: int = 100
    hidden: int = 16
    learning_rate: float = 0.01
    l1_penalty: float = 0.001
    weight_decay: float = 0.001
    seed: int = 42
    lookback: int = 1

    def __post_init__(self):
        if not isinstance(self.model, str) or self.model not in METHODS:
            raise ValueError(f"model 必须是一个模型名称：{METHODS}，不接受 all 或列表")
        integer(self.window, "window", 5)
        integer(self.epochs, "epochs", 1)
        integer(self.hidden, "hidden", 1)
        integer(self.seed, "seed", 0)
        integer(self.lookback, "lookback", 1)
        if not self.alphas:
            raise ValueError("alphas 不能为空")
        for alpha in self.alphas:
            number(alpha, "alpha", 0, strict=True)
            if alpha >= 1:
                raise ValueError("alpha 必须小于 1")
        if tuple(sorted(set(self.alphas))) != self.alphas:
            raise ValueError("alphas 必须递增且不重复")
        number(self.learning_rate, "learning_rate", 0, strict=True)
        number(self.l1_penalty, "l1_penalty", 0)
        number(self.weight_decay, "weight_decay", 0)


@dataclass(frozen=True)
class Experiment:
    config: Config
    test_size: int
    data: Path
    output: Path
    source: Path
    source_text: str
    model_source: Path
    model_source_text: str


def check_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} 必须包含且只包含：{', '.join(expected)}")


def load_experiment(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    source_text = path.read_text(encoding="utf-8")
    values = yaml.safe_load(source_text)
    check_keys(
        values, ("model", "test_size", "alphas", "data", "output"), "experiment.yaml"
    )
    model = values["model"]
    if not isinstance(model, str) or model not in METHODS:
        raise ValueError(f"model 必须只选择一个：{METHODS}，不接受 all 或列表")
    integer(values["test_size"], "test_size", 2)
    if not isinstance(values["alphas"], list):
        raise ValueError("alphas 必须是概率列表")
    for key in ("data", "output"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} 必须是非空路径")
    model_source = path.parent / f"{model}.yaml"
    model_source_text = model_source.read_text(encoding="utf-8")
    parameters = yaml.safe_load(model_source_text)
    check_keys(parameters, MODEL_FIELDS[model], f"{model}.yaml")
    return Experiment(
        config=Config(model=model, alphas=tuple(values["alphas"]), **parameters),
        test_size=values["test_size"],
        data=(path.parent / values["data"]).resolve(),
        output=(path.parent / values["output"]).resolve(),
        source=path,
        source_text=source_text,
        model_source=model_source,
        model_source_text=model_source_text,
    )
