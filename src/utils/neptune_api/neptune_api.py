import os
from typing import Any

from dotenv import load_dotenv
from neptune import Project, Run, init_project, init_run
from pandas import DataFrame

load_dotenv()


def get_project(project_name: str = "S4MODEL/Usleep-Adaptation") -> Project:
    return init_project(
        project=project_name,
        api_token=os.getenv("NEPTUNE_KEY"),
        mode="read-only",
    )


def get_run(id: str) -> Run:
    return init_run(
        with_id=id,
        project="S4MODEL/Usleep-Adaptation",
        api_token=os.getenv("NEPTUNE_KEY"),
        mode="read-only",
    )


def get_data_series(run: Run, key: str) -> DataFrame:
    return run[key].fetch_values()


def get_data_scalar(run: Run, key: str) -> Any:
    return run[key].fetch()
