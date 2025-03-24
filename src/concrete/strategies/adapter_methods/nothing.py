from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.adapter_method import AdapterMethod


class Nothing(AdapterMethod):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        return model
