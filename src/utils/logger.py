import os

from dotenv import load_dotenv
from lightning import Trainer
from lightning.pytorch.loggers import Logger, NeptuneLogger
from neptune import Run
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
    trainer: Trainer,
    train: DataLoader,
    val: DataLoader,
    test: DataLoader,
    num_batches: int,
) -> None:
    train_size = len(train.dataset) / num_batches  # type: ignore
    val_size = len(val.dataset)  # type: ignore
    test_size = len(test.dataset)  # type: ignore

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


def add_fold(trainer: Trainer, fold: int) -> None:
    if isinstance(trainer.logger, NeptuneLogger):
        trainer.logger.experiment["fold"] = fold


def add_completed(trainer: Trainer) -> None:
    if isinstance(trainer.logger, NeptuneLogger):
        trainer.logger.experiment["completed"] = True


def stop_logger(trainer: Trainer) -> None:
    logger = trainer.logger
    if isinstance(logger, NeptuneLogger):
        assert isinstance(logger.experiment, Run)
        logger.experiment.stop()
