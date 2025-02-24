from lightning import LightningModule

from src.interfaces.model_loader import ModelLoader


class StandardModelLoader(ModelLoader):
    def __init__(self, model_cls: type[LightningModule]) -> None:
        super().__init__(model_cls)
        self.model_cls = model_cls

    def load_pretrained(self, ckpt_path: str) -> LightningModule:
        return self.model_cls.load_from_checkpoint(ckpt_path)
