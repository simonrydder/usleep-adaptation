from src.models.resnet import Resnet
from src.models.simple import Simple

MODEL_REGISTRY = {
    "simple": Simple,
    "resnet": Resnet,
}
