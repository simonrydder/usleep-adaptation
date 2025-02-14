from abc import ABC, abstractmethod
from pathlib import Path

from lightning import LightningModule


class Preloader(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update_weights(
        self,
        model: LightningModule,
        ckpt: str | Path,
        strictly: bool,  # Maybe
    ) -> LightningModule:
        pass
