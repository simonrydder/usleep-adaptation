import torch
from peft import LoraConfig, get_peft_model

from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.adapter_method import AdapterMethod


class LoRA(AdapterMethod):
    def __init__(self, rank: int = 1, alpha: float = 1.0, dropout: float = 0.0) -> None:
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        super().__init__()

    def apply(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
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
        )
        peft_model = get_peft_model(model, lora_conf)  # type: ignore

        return peft_model  # type: ignore
