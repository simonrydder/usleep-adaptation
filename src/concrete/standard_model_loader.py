from src.config.config import ModelConfig
from src.config.experiment import Experiment
from src.interfaces.framework_model import FrameworkModel
from src.interfaces.model_loader import ModelLoader


class StandardModelLoader(ModelLoader):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model_cls = config.model
        self.ckpt = config.ckpt

    def load_pretrained(self, experiment: Experiment) -> FrameworkModel:
        model = self.model_cls.load_from_checkpoint(self.ckpt)
        setattr(model, "exp_setting", experiment.model_dump())
        setattr(model, "original_model", True)
        return model
