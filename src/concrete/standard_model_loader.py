from src.config.config import ModelConfig
from src.interfaces.framework_model import FrameworkModel
from src.interfaces.model_loader import ModelLoader


class StandardModelLoader(ModelLoader):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model_cls = config.model
        self.ckpt = config.ckpt

    def load_pretrained(self) -> FrameworkModel:
        return self.model_cls.load_from_checkpoint(self.ckpt)
