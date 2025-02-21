from lightning import LightningModule

from src.concrete.adapters.bitfit import BitFit
from src.concrete.adapters.conv_adapter import ConvAdapter
from src.concrete.strategies.forward_passes.parallel_forward_pass import (
    ParallelForwardPass,
)
from src.concrete.strategies.forward_passes.sequential_forward_pass import (
    SequentialForwardPass,
)
from src.interfaces.adapter import Adapter
from src.interfaces.strategies.forward_pass import ForwardPass
from src.models.resnet import Resnet
from src.models.simple import Simple

MODEL_REGISTRY: dict[str, type[LightningModule]] = {
    "simple": Simple,
    "resnet": Resnet,
}

ADAPTER_REGISTRY: dict[str, type[Adapter]] = {
    "bitfit": BitFit,
    "conv-adapter": ConvAdapter,
}


FORWARD_PASS_REGISTRY: dict[str, ForwardPass] = {
    "parallel": ParallelForwardPass(),
    "sequential": SequentialForwardPass(),
}
