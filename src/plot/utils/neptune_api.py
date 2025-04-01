import os
from datetime import datetime
from typing import Any, Sequence

import pandas as pd
import polars as pl
from dotenv import load_dotenv
from neptune import Project, Run, init_project, init_run
from pandas import DataFrame
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


def get_data_series(run: Run, key: str) -> DataFrame:
    return run[key].fetch_values()


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


class Measurement(BaseModel):
    record: str
    value: float


class Performance(BaseModel):
    kappa: list[Measurement]
    accuracy: list[Measurement]
    loss: list[Measurement]


class Record(BaseModel):
    step: int
    record: str


class Step(BaseModel):
    step: int
    value: float
    timestamp: datetime


class Epoch(BaseModel):
    epoch: int
    step: int
    value: float
    timestamp: datetime


class Training(BaseModel):
    kappa_step: list[Step]
    kappa_epoch: list[Epoch]
    accuracy_step: list[Step]
    accuracy_epoch: list[Epoch]
    loss_step: list[Step]
    loss_epoch: list[Epoch]


class Validation(BaseModel):
    val: Training
    records: list[Record]


class RunData(BaseModel):
    parameters: ParameterData
    config: ConfigData
    original_performance: Performance
    new_performance: Performance
    new_validation: Validation
    new_training: Training


def get_tag_data(tags: str | list[str]) -> dict[int, RunData]:
    project = get_project()
    runs = project.fetch_runs_table(tag=tags).to_pandas()

    data = {}
    for fold, run_id in enumerate(runs["sys/id"]):
        run = get_run(run_id)
        run_data = get_run_data(run)
        data[fold] = run_data

    return data


def _get_records(run: Run, mode: str, type: str) -> list[Record]:
    df = get_data_series(run, f"training/{mode}/{type}/records")

    return [
        Record(step=int(row["step"]), record=row["value"]) for _, row in df.iterrows()
    ]


def _get_steps(run: Run, mode: str, type: str, measurement: str) -> list[Step]:
    df = get_data_series(run, f"training/{mode}/{type}/{measurement}_step")

    return [
        Step(step=int(row["step"]), value=row["value"], timestamp=row["timestamp"])
        for _, row in df.iterrows()
    ]


def _get_epoch(run: Run, mode: str, type: str, measurement: str) -> list[Epoch]:
    df = get_data_series(run, f"training/{mode}/{type}/{measurement}_epoch")
    return [
        Epoch(
            epoch=i,
            step=int(row["step"]),
            value=row["value"],
            timestamp=row["timestamp"],
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]


def _get_measurements(
    run: Run, mode: str, type: str, measurement: str
) -> list[Measurement]:
    records = _get_records(run, mode, type)
    steps = _get_steps(run, mode, type, measurement)

    return [
        Measurement(record=rec.record, value=step.value)
        for rec, step in zip(records, steps)
    ]


def _get_performance(run: Run, mode: str) -> Performance:
    kappa = _get_measurements(run, mode, "test", "kappa")
    accuracy = _get_measurements(run, mode, "test", "accuracy")
    loss = _get_measurements(run, mode, "test", "loss")

    return Performance(kappa=kappa, accuracy=accuracy, loss=loss)


def _get_training(run: Run, mode: str, type: str) -> Training:
    kappa_step = _get_steps(run, mode, type, "kappa")
    kappa_epoch = _get_epoch(run, mode, type, "kappa")
    accuracy_step = _get_steps(run, mode, type, "accuracy")
    accuracy_epoch = _get_epoch(run, mode, type, "accuracy")
    loss_step = _get_steps(run, mode, type, "loss")
    loss_epoch = _get_epoch(run, mode, type, "loss")

    return Training(
        kappa_step=kappa_step,
        kappa_epoch=kappa_epoch,
        accuracy_step=accuracy_step,
        accuracy_epoch=accuracy_epoch,
        loss_step=loss_step,
        loss_epoch=loss_epoch,
    )


def get_run_data(run: Run) -> RunData:
    param_data = ParameterData(**get_data(run, "model/parameter_count"))
    config_data = ConfigData(**get_data(run, "model/config"))

    org_performance = _get_performance(run, "org")
    new_performance = _get_performance(run, "new")

    new_training = _get_training(run, "new", "train")
    new_validation = Validation(
        val=_get_training(run, "new", "val"),
        records=_get_records(run, "new", "val"),
    )

    return RunData(
        parameters=param_data,
        config=config_data,
        original_performance=org_performance,
        new_performance=new_performance,
        new_training=new_training,
        new_validation=new_validation,
    )


def get_original(tag_data: dict[int, RunData], tag: str) -> DataFrame:
    org_dfs = []
    for fold, run in tag_data.items():
        org = run.original_performance.kappa
        org_df = DataFrame([rec.model_dump() for rec in org])
        org_df["tag"] = tag
        org_df["dataset"] = run.config.experiment.dataset
        org_df["method"] = "original"
        org_df["fold"] = fold

        org_dfs.append(org_df)

    return pd.concat(org_dfs, ignore_index=True)


def get_test(tag_data: dict[int, RunData], tag: str) -> DataFrame:
    new_dfs = []
    for fold, run in tag_data.items():
        new = run.new_performance.kappa
        new_df = DataFrame([rec.model_dump() for rec in new])
        new_df["tag"] = tag
        new_df["dataset"] = run.config.experiment.dataset
        new_df["method"] = run.config.experiment.method
        new_df["fold"] = fold

        new_dfs.append(new_df)

    return pd.concat(new_dfs, ignore_index=True)


def convert_to_polars(values: Sequence[BaseModel]) -> pl.DataFrame:
    """
    Convert a list of Pydantic BaseModel instances to a Polars DataFrame.
    """
    data = [value.model_dump() for value in values]
    return pl.DataFrame(data)


if __name__ == "__main__":
    res = get_tag_data("1bv9KeTFq")
