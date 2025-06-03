import polars as pl

METHODS = [
    "Full",
    "SegCls",
    "BitFit",
    "BatchNorm",
    "Fish10",
    "Fish20",
    "Fish50",
    "LoRA10",
    "LoRA20",
    "LoRA50",
    "PCA10",
    "PCA20",
    "PCA50",
    "PCL10",
    "PCL20",
    "PCL50",
    "SCA10",
    "SCA20",
    "SCA50",
    "SCL10",
    "SCL20",
    "SCL50",
]

ORDER_MAP = {name: i for i, name in enumerate(METHODS)}


def sort_dataframe_by_method_order(df: pl.DataFrame) -> pl.DataFrame:
    sorted = (
        df.with_columns(
            pl.col("method")
            .map_elements(lambda x: ORDER_MAP.get(x, None), pl.Int128())
            .alias("sort_key")
        )
        .sort("sort_key")
        .drop("sort_key")
    )

    return sorted
