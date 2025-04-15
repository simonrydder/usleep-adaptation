import os
from typing import Iterator

from src.utils.neptune_api.method_data import (
    MethodData,
    get_method_data,
    load_method_data,
    save_method_data,
)
from src.utils.neptune_api.neptune_api import get_project


def method_ids_iterator(dataset: str) -> Iterator[tuple[str, list[str]]]:
    project = get_project()
    runs = project.fetch_runs_table(tag=dataset).to_pandas()
    method_info = "model/config/experiment/method"
    methods = runs[method_info].unique()

    for method in methods:
        method_runs = runs.loc[runs[method_info] == method]
        run_ids = method_runs["sys/id"].values.tolist()
        yield method, run_ids


def get_data(dataset: str) -> dict[str, MethodData]:
    data = {}
    for method, ids in method_ids_iterator(dataset):
        method_data = get_method_data(ids)
        save_method_data(method_data, dataset, method)

        data[method] = method_data

    return data


def load_data(dataset: str) -> dict[str, MethodData]:
    folder = os.path.join("results", dataset)
    data = {}
    for file in os.listdir(folder):
        method = file.split(".")[0]
        data[method] = load_method_data(dataset, method)

    return data


if __name__ == "__main__":
    # get_data("eesm19")
    x = load_data("eesm19")
    pass
