import os

from dotenv import load_dotenv
from lightning.pytorch.loggers import Logger, NeptuneLogger

load_dotenv()


def neptune_logger(name: str) -> Logger:
    return NeptuneLogger(
        api_token=os.environ["NEPTUNE_KEY"],
        project="S4MODEL/Usleep-Adaptation",
        name=name,
        source_files=[
            "src/config/yaml/adapter_method/*.yaml",
            "src/config/yaml/dataset/*.yaml",
            "src/config/yaml/trainer/*.yaml",
            "src/config/yaml/model/*.yaml",
        ],
    )
