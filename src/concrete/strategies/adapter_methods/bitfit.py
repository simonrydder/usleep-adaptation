from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.adapter_method import AdapterMethod


class BitFit(AdapterMethod):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        for name, param in model.named_parameters():
            if name.endswith("bias"):
                param.requires_grad = True

        return model
