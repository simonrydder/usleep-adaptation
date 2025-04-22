from pydantic import BaseModel


class Experiment(BaseModel):
    dataset: str
    method: str
    model: str
    trainer: str
    id: int | str = 0
    seed: int = 42


def get_experiment_name(experiment: Experiment) -> str:
    return experiment.trainer


def generate_experiments(datasets: list[str]): ...
