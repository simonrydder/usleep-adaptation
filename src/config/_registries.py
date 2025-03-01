import pytorch_lightning as pl
import torch.nn as nn
from csdp_training.lightning_models.usleep import USleep_Lightning
from lightning import LightningModule

from src.concrete.strategies.adapter_methods.batch_norm import BatchNorm
from src.concrete.strategies.adapter_methods.bitfit import BitFit
from src.concrete.strategies.adapter_methods.conv_adapter import ConvAdapter
from src.concrete.strategies.adapter_methods.fully_finetune import FullyFinetune
from src.concrete.strategies.adapter_methods.lora import LoRA
from src.concrete.strategies.adapter_methods.nothing import Nothing
from src.concrete.strategies.forward_passes.parallel_forward_pass import (
    ParallelForwardPass,
)
from src.concrete.strategies.forward_passes.sequential_forward_pass import (
    SequentialForwardPass,
)
from src.interfaces.strategies.adapter_method import AdapterMethod
from src.interfaces.strategies.forward_pass import ForwardPass
from src.models.conv_simple import ConvSimple
from src.models.resnet import Resnet
from src.models.simple import Simple

MODEL_REGISTRY: dict[str, type[LightningModule] | type[pl.LightningModule]] = {
    "simple": Simple,
    "resnet": Resnet,
    "conv_simple": ConvSimple,
    "usleep": USleep_Lightning,
}

ADAPTER_METHODS_REGISTRY: dict[str, type[AdapterMethod]] = {
    "bitfit": BitFit,
    "conv-adapter": ConvAdapter,
    "batch-norm": BatchNorm,
    "lora": LoRA,
    "nothing": Nothing,
    "fully-finetune": FullyFinetune,
}


FORWARD_PASS_REGISTRY: dict[str, ForwardPass] = {
    "parallel": ParallelForwardPass(),
    "sequential": SequentialForwardPass(),
}


ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
}
