from abc import ABC, abstractmethod

from lightning import LightningModule


class ModelLoader(ABC):
    def __init__(self, model_class: type[LightningModule]) -> None:
        self.model_cls = model_class
        super().__init__()

    @abstractmethod
    def load_pretrained(self, ckpt_path: str) -> LightningModule:
        pass
