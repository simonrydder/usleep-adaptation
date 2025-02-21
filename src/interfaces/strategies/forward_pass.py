from abc import ABC, abstractmethod

import torch


class ForwardPass(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
