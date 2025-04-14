from typing import Iterator

from src.plot.utils.neptune_api_old import get_project
from src.utils.neptune_api.method_data import (
    MethodData,
    get_method_data,
    save_method_data,
)


def method_ids_iterator(dataset: str) -> Iterator[tuple[str, list[str]]]:
    project = get_project()
    runs = project.fetch_runs_table(tag=[dataset, "BitFit"]).to_pandas()
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


if __name__ == "__main__":
    get_data("eesm19")
