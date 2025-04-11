from src.config.config import Config, ModelConfig
from src.interfaces.framework_model import FrameworkModel
from src.interfaces.model_loader import ModelLoader
from src.utils.id_generation import generate_base62_id


class StandardModelLoader(ModelLoader):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model_cls = config.model
        self.ckpt = config.ckpt
        self.lr = config.lr
        self.lr_patience = config.lr_patience
        self.lr_minimum = config.lr_minimum
        self.lr_factor = config.lr_factor

    def load_pretrained(self, config: Config) -> FrameworkModel:
        model = self.model_cls.load_from_checkpoint(self.ckpt)
        setattr(model, "lr", self.lr)
        setattr(model, "lr_patience", self.lr_patience)
        setattr(model, "lr_minimum", self.lr_minimum)
        setattr(model, "lr_factor", self.lr_factor)

        setattr(model, "config", config.model_dump())
        setattr(model, "original_model", True)
        setattr(model, "experiment_id", generate_base62_id())
        return model
