from pydantic import BaseModel


class Experiment(BaseModel):
    dataset: str
    method: str
    model: str
    trainer: str


def generate_experiments(datasets: list[str]): ...
