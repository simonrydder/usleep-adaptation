import itertools
import os

import yaml
from pydantic import BaseModel

from src.config.utils import load_yaml_content
from src.utils.id_generation import generate_base62_id

_YAML_FOLDER = os.path.join("src", "config", "yaml")


class Experiment(BaseModel):
    key: str
    dataset: str
    method: str
    model: str
    trainer: str
    train_size: int | None = None
    fold: int
    seed: int


def get_experiment_name(experiment: Experiment) -> str:
    return "_".join(
        [
            experiment.dataset,
            experiment.method,
            str(experiment.fold),
            experiment.key,
        ]
    )


def generate_experiments(
    datasets: list[str] | None,
    methods: list[str] | None,
    folds: list[int] | None,
    train_size: int | None,
    seed: int = 42,
    key: str | None = None,
) -> list[Experiment]:
    if methods is None:
        methods = _get_yaml_methods()

    if datasets is None:
        datasets = _get_yaml_datasets()

    exps = []
    for dataset, method in itertools.product(datasets, methods):
        if key is None:
            key = generate_base62_id()

        if folds is None:
            dataset_content = load_yaml_content(os.path.join("dataset", dataset))
            num_fold = int(dataset_content["num_fold"])
            folds = list(range(num_fold))

        for fold in folds:
            exp = Experiment(
                key=key,
                dataset=dataset,
                method=method,
                model="usleep",
                trainer="usleep",
                train_size=train_size,
                fold=fold,
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


def _get_yaml_datasets() -> list[str]:
    global _YAML_FOLDER

    files = os.listdir(os.path.join(_YAML_FOLDER, "dataset"))
    datasets = [f.split(".")[0] for f in files]

    datasets.remove("_default")

    return datasets


def save_experiment(experiment: Experiment) -> None:
    global _YAML_FOLDER
    name = get_experiment_name(experiment)
    filename = f"{name}.yaml"

    with open(os.path.join(_YAML_FOLDER, "experiments", filename), "w") as f:
        yaml.dump(experiment.model_dump(), f)
