import torch.nn as nn
from torch import Tensor

from src.interfaces.strategies.forward_pass import ForwardPass


class ParallelForwardPass(ForwardPass):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        original = kwargs.get("original")
        assert isinstance(original, nn.Module)

        adapter = kwargs.get("adapter")
        assert isinstance(adapter, nn.Module)

        h = original(x)
        h_delta = adapter(x)

        return h + h_delta
