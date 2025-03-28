from typing import Literal

from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger


def define_trainer(
    max_epochs: int,
    accelerator: Literal["gpu", "cpu"] = "gpu",
    callbacks: list[Callback] | None = None,
    logger: Logger | bool = True,
) -> Trainer:
    return Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        log_every_n_steps=1,
        devices=accelerator,
        callbacks=callbacks,
        logger=logger,
    )
