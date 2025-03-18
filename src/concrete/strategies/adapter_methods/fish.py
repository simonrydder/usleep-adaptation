from typing import Literal

from lightning import LightningModule

from src.concrete.strategies.adapter_methods.utils.fisher import FisherMask
from src.interfaces.strategies.adapter_method import AdapterMethod


class FISH(AdapterMethod):
    def __init__(
        self,
        keep_ratio: float,
        num_samples: int = 1,
        grad_type: Literal["absolute", "square"] = "square",
    ) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.grad_type = grad_type
        self.keep_ratio = keep_ratio

    def apply(
        self,
        model: LightningModule,
        **kwargs,
    ) -> LightningModule:
        dataloader = kwargs.get("dataloader")
        assert dataloader is not None, "Dataloader is required for FISH adapter method"

        for param in model.parameters():
            param.requires_grad = True

        fish = FisherMask(
            model,
            dataloader,
            self.num_samples,
            self.keep_ratio,
            self.grad_type,
        )
        masks = fish.calculate_mask()

        for name, param in model.named_parameters():
            param.register_hook(lambda grad, mask=masks[name]: grad * mask)

        return model
