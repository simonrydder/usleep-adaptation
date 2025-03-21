import os

from dotenv import load_dotenv
from lightning.pytorch.loggers import Logger, NeptuneLogger

load_dotenv()


def neptune_logger() -> Logger:
    return NeptuneLogger(
        api_token=os.environ["NEPTUNE_KEY"],
        project="S4MODEL/Usleep-Adaptation",
        name="test",
    )
