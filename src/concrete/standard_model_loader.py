from lightning import LightningModule

from src.interfaces.model_loader import ModelLoader


class StandardModelLoader(ModelLoader):
    def __init__(self, model: LightningModule) -> None:
        super().__init__(model)
        self.model_cls = type(model)

    def load_pretrained(self, ckpt_path: str) -> LightningModule:
        return self.model_cls.load_from_checkpoint(ckpt_path)
