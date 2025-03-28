import os
from typing import Any

from dotenv import load_dotenv
from neptune import Project, Run, init_project, init_run
from neptune.exceptions import MissingFieldException
from pydantic import BaseModel

from src.config.experiment import Experiment

load_dotenv()


def get_run(id: str) -> Run:
    return init_run(
        with_id=id,
        project="S4MODEL/Usleep-Adaptation",
        api_token=os.getenv("NEPTUNE_KEY"),
        mode="read-only",
    )


def get_project(project_name: str = "S4MODEL/Usleep-Adaptation") -> Project:
    return init_project(
        project=project_name,
        api_token=os.getenv("NEPTUNE_KEY"),
        mode="read-only",
    )


def get_data_series(run: Run, key: str) -> list[Any] | None:
    try:
        return run[key].fetch_values()["value"].tolist()
    except MissingFieldException:
        return None


def get_data(run: Run, key: str) -> Any:
    return run[key].fetch()


class TrainableParameterData(BaseModel):
    free: int = 0
    frozen: int = 0


class ParameterData(BaseModel):
    total: int
    classification: TrainableParameterData
    model: TrainableParameterData


class ConfigData(BaseModel):
    experiment: Experiment


class Kappa(BaseModel):
    epoch: list[float]
    step: list[float]
    records: list[str] | None


class ModelResults(BaseModel):
    val: Kappa
    test: Kappa
    train: Kappa | None = None


class ResultData(BaseModel):
    parameters: ParameterData
    config: ConfigData


def get_tag_data(tags: str | list[str]) -> ResultData:
    project = get_project()
    runs = project.fetch_runs_table(tag=tags).to_pandas()

    for run_id in runs["sys/id"]:
        run = get_run(run_id)
        run_data = get_run_data(run)
    pass


def get_run_data(run: Run) -> Any:
    param_data = ParameterData(**get_data(run, "model/parameter_count"))
    config_data = ConfigData(**get_data(run, "model/config"))

    for mode in ["org", "new"]:
        for type in ["train", "val", "test"]:
            epoch_data = get_data_series(run, f"training/{mode}/{type}/kappa_epoch")
            assert epoch_data is not None

            step_data = get_data_series(run, f"training/{mode}/{type}/kappa_step")
            assert step_data is not None

            records_data = get_data_series(run, f"training/{mode}/{type}/records")

            kappa = Kappa(
                epoch=epoch_data,
                step=step_data,
                records=records_data,
            )
    org_results = ModelResults()

    pass


if __name__ == "__main__":
    res = get_tag_data("1bv9KeTFq")
