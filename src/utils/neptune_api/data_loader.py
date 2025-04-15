import os

from tqdm import tqdm

from src.utils.neptune_api.method_data import (
    MethodData,
    get_method_data,
    load_method_data,
    save_method_data,
)
from src.utils.neptune_api.neptune_api import get_project


class MethodIterator:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.project = get_project()
        runs = self.project.fetch_runs_table(
            tag=dataset, progress_bar=False
        ).to_pandas()
        self.runs = runs.loc[~runs["sys/failed"]]
        self.method_column = "model/config/experiment/method"
        self.methods = self.runs[self.method_column].unique().tolist()

    def __len__(self) -> int:
        return len(self.methods)

    def __iter__(self) -> "MethodIterator":
        self.index = 0
        return self

    def __next__(self) -> tuple[str, list[str]]:
        if self.index >= len(self):
            raise StopIteration

        method = self.methods[self.index]
        method_runs = self.runs.loc[self.runs[self.method_column] == method]
        runs_ids = method_runs["sys/id"].values.tolist()
        self.index += 1

        return method, runs_ids


def get_data(dataset: str) -> dict[str, MethodData]:
    data = {}
    for method, ids in tqdm(MethodIterator(dataset), desc=f"Downloading {dataset}"):
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
    get_data("eesm19")
    # x = load_data("eesm19")
    pass
