from lightning import LightningModule

from src.models.simple import Simple
from src.models.usleep import UsleepLightning


def is_classification_parameter(name: str, model: LightningModule) -> bool:
    if isinstance(model, UsleepLightning):
        return usleep_classification(name)

    if isinstance(model, Simple):
        return simple_classification(name)

    raise NotImplementedError(
        f"No implementation for classification check for {type(model)}"
    )


def usleep_classification(name: str) -> bool:
    return "dense" in name or "classifier" in name


def simple_classification(name: str) -> bool:
    return False
