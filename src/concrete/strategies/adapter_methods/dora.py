from lightning import LightningModule
from peft import LoraConfig, get_peft_model

from src.interfaces.strategies.adapter_method import AdapterMethod


class DoRA(AdapterMethod):
    def __init__(self, rank: int = 1, alpha: int = 1, dropout: float = 0.0) -> None:
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        super().__init__()

    def apply(self, model: LightningModule) -> LightningModule:
        config = LoraConfig(
            use_dora=True,
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=["conv1", "conv2"],
            lora_dropout=self.dropout,
            bias="none",
            modules_to_save=["classifier"],
        )

        model = get_peft_model(model, config)

        return model
