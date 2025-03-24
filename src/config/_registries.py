from typing import Generic, KeysView, TypeVar

import torch.nn as nn

from src.concrete.strategies.adapter_methods.batch_norm import BatchNorm
from src.concrete.strategies.adapter_methods.bitfit import BitFit
from src.concrete.strategies.adapter_methods.conv_adapter import ConvAdapter
from src.concrete.strategies.adapter_methods.fish import FISH
from src.concrete.strategies.adapter_methods.fully_finetune import FullyFinetune
from src.concrete.strategies.adapter_methods.lora import LoRA
from src.concrete.strategies.adapter_methods.nothing import Nothing
from src.concrete.strategies.adapter_methods.residual_adapter import ResidualAdapter
from src.concrete.strategies.forward_passes.parallel_forward_pass import (
    ParallelForwardPass,
)
from src.concrete.strategies.forward_passes.sequential_forward_pass import (
    SequentialForwardPass,
)
from src.concrete.strategies.parameter_count_methods.grad import Grad
from src.concrete.strategies.parameter_count_methods.hook import Hook
from src.concrete.strategies.parameter_count_methods.skip import Skip
from src.models.conv_simple import ConvSimple
from src.models.resnet import Resnet
from src.models.simple import Simple
from src.models.usleep import UsleepLightning

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, registry: dict[str, T]) -> None:
        super().__init__()
        self.reg = registry

    def lookup(self, key: str) -> T:
        if key not in self.reg:
            raise NotImplementedError(
                f"'{key}' not defined in {self.__str__()}. Available keys: {list(self.reg.keys())}"
            )

        return self.reg[key]

    def keys(self) -> KeysView[str]:
        return self.reg.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.reg


MODEL_REG = Registry(
    {
        "simple": Simple,
        "resnet": Resnet,
        "conv_simple": ConvSimple,
        "usleep": UsleepLightning,
    }
)


ADAPTER_METHOD_REG = Registry(
    {
        "bitfit": BitFit,
        "conv-adapter": ConvAdapter,
        "batch-norm": BatchNorm,
        "lora": LoRA,
        "nothing": Nothing,
        "fully-finetune": FullyFinetune,
        "fish": FISH,
        "residual-adapter": ResidualAdapter,
    }
)


FORWARD_PASS_REG = Registry(
    {
        "parallel": ParallelForwardPass(),
        "sequential": SequentialForwardPass(),
    }
)


ACTIVATION_REG = Registry(
    {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }
)


PARAMETER_COUNT_REG = Registry(
    {
        "skip": Skip,
        "hook": Hook,
        "grad": Grad,
    }
)

if __name__ == "__main__":
    MODEL_REG.lookup("test")
    pass
