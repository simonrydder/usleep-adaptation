from typing import Literal

from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, Timer


def early_stopping(
    monitor: str,
    patience: int,
    mode: Literal["min", "max"],
) -> Callback:
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=True,
        mode=mode,
        min_delta=0.0001,
    )


def learning_rate_monitor(logging_interval: Literal["epoch", "step"]) -> Callback:
    return LearningRateMonitor(logging_interval=logging_interval)


def timer() -> Callback:
    return Timer()
