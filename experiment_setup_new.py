import itertools
import os

import polars as pl
from filelock import FileLock

from src.config.experiment import (
    Experiment,
    _get_yaml_methods,
)
from src.config.utils import load_yaml_content
from src.utils.id_generation import generate_base62_id


def generate_experiments(
    dataset: str,
    methods: list[str] | None,
    folds: list[int] | None,
    train_sizes: list[int | None],
    seed: int = 42,
    key: str | None = None,
) -> list[Experiment]:
    if methods is None:
        methods = _get_yaml_methods()

    exps = []
    for method, train_size in itertools.product(methods, train_sizes):
        if key is None:
            fold_key = generate_base62_id()
        else:
            fold_key = key

        if folds is None:
            dataset_content = load_yaml_content(os.path.join("dataset", dataset))
            num_fold = int(dataset_content["num_fold"])
            folds = list(range(num_fold))

        for fold in folds:
            exp = Experiment(
                key=fold_key,
                dataset=dataset,
                method=method,
                model="usleep",
                trainer="usleep",
                train_size=train_size,
                fold=fold,
                seed=seed,
            )

            exps.append(exp)

    return exps


EXPERIMENT_SCHEMA = pl.Schema(
    {
        "index": pl.Int64(),
        "key": pl.String(),
        "dataset": pl.String(),
        "method": pl.String(),
        "model": pl.String(),
        "trainer": pl.String(),
        "train_size": pl.Int8(),
        "fold": pl.Int8(),
        "seed": pl.Int64(),
        "status": pl.String(),
    }
)


def get_existing_experiments(csv_path: str) -> pl.DataFrame:
    try:
        return pl.read_csv(csv_path, schema=EXPERIMENT_SCHEMA)
    except FileNotFoundError:
        return pl.DataFrame(schema=EXPERIMENT_SCHEMA)


def filter_new_experiments(org: pl.DataFrame, all: pl.DataFrame) -> pl.DataFrame:
    return all.join(
        org,
        on=["dataset", "method", "model", "trainer", "train_size", "fold", "seed"],
        how="anti",
        nulls_equal=True,
    )


def update_experiments(csv_path: str, experiments: list[Experiment]) -> None:
    org_experiments = get_existing_experiments(csv_path)
    all_experiments = pl.DataFrame(experiments, schema=EXPERIMENT_SCHEMA).with_columns(
        pl.lit("pending").alias("status")
    )
    new_experiments = filter_new_experiments(org_experiments, all_experiments)
    last_index: int = (
        org_experiments.get_column("index").max() if len(org_experiments) > 0 else -1
    )  # type: ignore
    new_start = last_index + 1
    new_experiments = new_experiments.with_columns(
        pl.int_range(new_start, last_index + 1 + new_experiments.height).alias("index")
    )

    updated = pl.concat([org_experiments, new_experiments])
    updated.write_csv(csv_path)
    return None


def update_experiments_with_lock(csv_path: str, experiments: list[Experiment]) -> None:
    lock_path = csv_path + ".lock"
    with FileLock(lock_path, timeout=10):
        update_experiments(csv_path=csv_path, experiments=experiments)


def main() -> None:
    experiments = [
        generate_experiments("dod_h", None, None, [4, 8, None]),
        generate_experiments("dod_o", None, None, [4, 8, 16, 32, None]),
        generate_experiments("eesm19", None, None, [1, 2, 4, 8, None]),
        generate_experiments("eesm23", None, None, [4, 8, None]),
        generate_experiments("isruc_sg1", None, None, [4, 8, 16, 32, None]),
        generate_experiments("isruc_sg2", None, None, [None]),
        generate_experiments("isruc_sg3", None, None, [4, None]),
        generate_experiments("mass_c1", None, None, [4, 8, 16, None]),
        generate_experiments("mass_c3", None, None, [4, 8, 16, 32, None]),
        generate_experiments("svuh", None, None, [4, 8, None]),
    ]

    experiment_folder = os.path.join("src", "config", "yaml", "experiments")
    for exp in experiments:
        dataset = exp[0].dataset
        csv_path = os.path.join(experiment_folder, f"{dataset}.csv")
        update_experiments_with_lock(csv_path, exp)


if __name__ == "__main__":
    main()
