from lightning import LightningModule

from src.concrete.strategies.models.linear import SimpleLinearModel
from src.concrete.strategies.models.usleep import UsleepModel


def is_classification_parameter(name: str, model: LightningModule) -> bool:
    if isinstance(model, UsleepModel):
        return usleep_classification(name)

    if isinstance(model, SimpleLinearModel):
        return simple_classification(name)

    raise NotImplementedError(
        f"No implementation for classification check for {type(model)}"
    )


def usleep_classification(name: str) -> bool:
    return "dense" in name or "classifier" in name


def simple_classification(name: str) -> bool:
    return False
