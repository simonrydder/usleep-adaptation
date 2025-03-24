from lightning import LightningModule

from src.interfaces.strategies.parameter_count_method import ParameterCountMethod


class Hook(ParameterCountMethod):
    def __init__(self) -> None:
        super().__init__()

    def set_parameter_count(self, model: LightningModule) -> LightningModule:
        return super().set_parameter_count(model)
