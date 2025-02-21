import os
from typing import Annotated

import yaml
from lightning import LightningModule
from pydantic import AfterValidator, BaseModel

from src.config._adapter_config import AdapterConfig
from src.config._registries import MODEL_REGISTRY
from src.config._validators import (
    validate_file_existence,
    validate_model_name,
)


class Config(BaseModel):
    model: Annotated[str, AfterValidator(validate_model_name)]
    ckpt: Annotated[str, AfterValidator(validate_file_existence)]
    adapter: AdapterConfig

    def get_model_class(self) -> type[LightningModule]:
        """Retrieve the LightningModule class based on the model name."""
        return MODEL_REGISTRY[self.model]


def load_config(yaml_filename: str) -> Config:
    yaml_file = os.path.join("src", "config", "yaml", yaml_filename)

    if not yaml_file.endswith(".yaml"):
        yaml_file += ".yaml"

    with open(yaml_file, "r") as f:
        raw_config = yaml.safe_load(f)

    return Config(**raw_config)
