from lightning import LightningModule

from src.concrete.adapters.bitfit import BitFit
from src.interfaces.adapter import Adapter
from src.models.resnet import Resnet
from src.models.simple import Simple

MODEL_REGISTRY: dict[str, type[LightningModule]] = {
    "simple": Simple,
    "resnet": Resnet,
}
ADAPTER_REGISTRY: dict[str, type[Adapter]] = {
    "bitfit": BitFit,
}
