import contextlib
import logging
import os
import sys
from io import StringIO
from typing import Any

from dotenv import load_dotenv
from neptune import Project, Run, init_project, init_run
from pandas import DataFrame

load_dotenv()
logging.getLogger("neptune").setLevel(logging.WARNING)


@contextlib.contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout


def get_project(project_name: str = "S4MODEL/Usleep-Adaptation") -> Project:
    with suppress_stdout():
        return init_project(
            project=project_name,
            api_token=os.getenv("NEPTUNE_KEY"),
            mode="read-only",
        )


def get_run(id: str) -> Run:
    with suppress_stdout():
        return init_run(
            with_id=id,
            project="S4MODEL/Usleep-Adaptation",
            api_token=os.getenv("NEPTUNE_KEY"),
            mode="read-only",
        )


def get_data_series(run: Run, key: str) -> DataFrame:
    return run[key].fetch_values(progress_bar=False)


def get_data_scalar(run: Run, key: str) -> Any:
    return run[key].fetch()
