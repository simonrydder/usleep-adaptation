from src.concrete.adapters.bitfit import BitFit
from src.models.resnet import Resnet
from src.models.simple import Simple

MODEL_REGISTRY = {
    "simple": Simple,
    "resnet": Resnet,
}
ADAPTER_REGISTRY = {
    "bitfit": BitFit,
}
