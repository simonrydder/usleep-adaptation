from lightning import LightningModule
from peft import LoraConfig, get_peft_model

from src.interfaces.adapter import Adapter


class LoRA(Adapter):
    def __init__(self):  # rank: int, alpha: int, dropout: float) -> None:
        super().__init__()

    def adapt(self, model: LightningModule) -> LightningModule:
        config = LoraConfig(
            r=2,
            lora_alpha=16,
            target_modules=["conv1", "conv2"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        lora_model = get_peft_model(model.model, config)

        return lora_model
