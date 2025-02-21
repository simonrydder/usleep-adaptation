from lightning import LightningModule

from src.concrete.adapters.batch_norm_adapter import BatchNormAdapter
from src.concrete.adapters.bitfit import BitFit
from src.concrete.adapters.conv_adapter import ConvAdapter
from src.concrete.adapters.lora import LoRA
from src.interfaces.adapter import Adapter
from src.models.conv_simple import ConvSimple
from src.models.resnet import Resnet
from src.models.simple import Simple

MODEL_REGISTRY: dict[str, type[LightningModule]] = {
    "simple": Simple,
    "resnet": Resnet,
    "conv_simple": ConvSimple,
}

ADAPTER_REGISTRY: dict[str, type[Adapter]] = {
    "bitfit": BitFit,
    "conv-adapter": ConvAdapter,
    "batch_norm_adapter": BatchNormAdapter,
    "lora": LoRA,
}
