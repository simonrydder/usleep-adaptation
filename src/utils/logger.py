import os

from dotenv import load_dotenv
from lightning import Trainer
from lightning.pytorch.loggers import Logger, NeptuneLogger
from torch.utils.data import DataLoader

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


def log_size_of_datasets(
    trainer: Trainer, train: DataLoader, val: DataLoader, test: DataLoader
) -> None:
    train_size = train.dataset.sampler.num_records[0]  # type: ignore
    val_size = val.dataset.sampler.num_samples  # type: ignore
    test_size = test.dataset.sampler.num_samples  # type: ignore

    if isinstance(trainer.logger, NeptuneLogger):
        trainer.logger.experiment["model/config/data/sizes"] = {
            "train": train_size,
            "validation": val_size,
            "test": test_size,
        }


def add_tags(trainer: Trainer, *tags: str) -> None:
    if isinstance(trainer.logger, NeptuneLogger):
        for tag in tags:
            trainer.logger.experiment["sys/tags"].add(tag)
