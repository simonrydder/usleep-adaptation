import itertools
import os

import yaml
from pydantic import BaseModel

_YAML_FOLDER = os.path.join("src", "config", "yaml")


class Experiment(BaseModel):
    dataset: str
    method: str
    model: str
    trainer: str
    id: int | str = 0
    seed: int = 42


def get_experiment_name(experiment: Experiment) -> str:
    return "_".join([experiment.dataset, experiment.method])


def generate_experiments(
    datasets: list[str],
    methods: list[str] | None,
    id: int,
    seed: int = 42,
) -> list[Experiment]:
    if methods is None:
        methods = _get_yaml_methods()

    exps = []
    for dataset, method in itertools.product(datasets, methods):
        exp = Experiment(
            dataset=dataset,
            method=method,
            model="usleep",
            trainer="usleep",
            id=id,
            seed=seed,
        )

        exps.append(exp)

    return exps


def _get_yaml_methods() -> list[str]:
    global _YAML_FOLDER

    files = os.listdir(os.path.join(_YAML_FOLDER, "adapter_method"))
    methods = [f.split(".")[0] for f in files]

    methods.remove("_default")

    return methods


def save_experiment(experiment: Experiment) -> None:
    global _YAML_FOLDER
    name = get_experiment_name(experiment)
    filename = f"{name}.yaml"

    with open(os.path.join(_YAML_FOLDER, "experiments", filename), "w") as f:
        yaml.dump(experiment.model_dump(), f)
