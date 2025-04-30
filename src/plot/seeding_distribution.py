from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_validation_data

import polars as pl


def plot_seeding_distribution() -> None:
    data = _get_seeding_data()


def _get_seeding_data() -> pl.DataFrame:
    raw_data = load_data(ids=[42, 43, 44, 45, 46])

    dfs = []
    for data in raw_data:
        val = extract_validation_data(data)
        val = val.drop("accuracy", "loss", "f1", "id")
        dfs.append(val)

    df = pl.concat(dfs, how='vertical')

    return df


def _plot_seeding_distribution()