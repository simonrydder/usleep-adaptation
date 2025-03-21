from pydantic import BaseModel


class Experiment(BaseModel):
    dataset: str
    method: str
    model: str
    trainer: str


def get_experiment_name(experiment: Experiment) -> str:
    return "_".join(experiment.model_dump().values())


def generate_experiments(datasets: list[str]): ...
