from pathlib import Path

import yaml


def create_experiments():
    datasets_path = Path("src/config/yaml/dataset")
    methods_path = Path("src/config/yaml/adapter_method")

    datasets = []
    methods = []

    for file in datasets_path.iterdir():
        datasets.append(file.name.split(".")[0])

    for file in methods_path.iterdir():
        methods.append(file.name.split(".")[0])

    for dataset in datasets:
        for method in methods:
            data = {
                "dataset": dataset,
                "model": "usleep",
                "method": method,
                "trainer": "usleep",
            }

            with open(
                f"src/config/yaml/experiments/{dataset}_{method}.yaml", "w"
            ) as file:
                yaml.dump(data, file, default_flow_style=False)


if __name__ == "__main__":
    create_experiments()
