import os
from typing import Annotated, Any

import yaml
from lightning import LightningModule
from pydantic import AfterValidator, BaseModel

from src.config._adapter_config import AdapterConfig
from src.config._data_config import DataConfig
from src.config._registries import MODEL_REGISTRY
from src.config._validators import (
    validate_file_existence,
    validate_model_name,
)


class Config(BaseModel):
    model: Annotated[str, AfterValidator(validate_model_name)]
    ckpt: Annotated[str, AfterValidator(validate_file_existence)]
    data: DataConfig
    adapter: AdapterConfig
    # trainer: TrainerConfig

    def get_model_class(self) -> type[LightningModule]:
        """Retrieve the LightningModule based on the model name."""
        model_cls = MODEL_REGISTRY[self.model]

        return model_cls  # type: ignore


def include_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    """Custom constructor to handle !include directives in YAML files."""
    filename = node.value
    base_path = os.path.dirname(loader.name)
    file_path = os.path.join(base_path, filename)

    with open(file_path, "r") as f:
        return yaml.safe_load(f)


yaml.SafeLoader.add_constructor("!include", include_constructor)


def load_config(yaml_filename: str) -> Config:
    yaml_file = os.path.join("src", "config", "yaml", yaml_filename)

    if not yaml_file.endswith(".yaml"):
        yaml_file += ".yaml"

    with open(yaml_file, "r") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)

    default = raw_config.get("default", {})
    del raw_config["default"]
    configs = update_default_configs(default, raw_config)

    return Config(**configs)


def update_default_configs(
    default: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    config = default.copy()

    for key, value in updates.items():
        if key in config and isinstance(value, dict) and isinstance(config[key], dict):
            config[key] = update_default_configs(config[key], value)

        else:
            config[key] = value

    return config


if __name__ == "__main__":
    load_config("resnet/bitfit")
