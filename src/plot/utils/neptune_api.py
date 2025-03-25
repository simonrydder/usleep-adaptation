import os
from typing import Any

from dotenv import load_dotenv
from neptune import Run, init_run
from pandas import DataFrame

load_dotenv()


def get_run(id: str) -> Run:
    return init_run(
        with_id=id,
        project="",
        api_token=os.getenv("NEPTUNE_KEY"),
        mode="read-only",
    )


def get_data_series(run: Run, key: str) -> DataFrame:
    return run[key].fetch_values()


def get_data(run: Run, key: str) -> Any:
    return run[key].fetch()
