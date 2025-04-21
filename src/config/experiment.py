from pydantic import BaseModel


class Experiment(BaseModel):
    dataset: str
    method: str
    model: str
    trainer: str
    id: str = "Id_0"


def get_experiment_name(experiment: Experiment) -> str:
    return experiment.trainer


def generate_experiments(datasets: list[str]): ...
