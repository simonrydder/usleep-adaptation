import torch
from peft import LoraConfig, get_peft_model

from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.adapter_method import AdapterMethod
from src.utils.classification import is_classification_parameter


class DoRA(AdapterMethod):
    def __init__(self, rank: int = 1, alpha: float = 1.0, dropout: float = 0.0) -> None:
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        super().__init__()

    def apply(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        exclude_modules = []
        target_modules = []

        for name, module in model.named_modules():
            if "classifier" in name or "dense" in name:
                exclude_modules.append(name)
            elif isinstance(module, torch.nn.Conv1d):
                target_modules.append(name)

            elif isinstance(module, torch.nn.Linear):
                target_modules.append(name)

            elif isinstance(module, torch.nn.Conv2d):
                target_modules.append(name)

        lora_conf = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,  # type: ignore
            lora_dropout=self.dropout,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            bias="lora_only",
            fan_in_fan_out=True,
            use_dora=True,
            use_rslora=True,
        )
        peft_model = get_peft_model(model, lora_conf)  # type: ignore

        for name, param in peft_model.named_parameters():
            if is_classification_parameter(name, model):
                param.requires_grad = True

        return peft_model  # type: ignore
