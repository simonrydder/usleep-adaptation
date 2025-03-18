import os

from pydantic import BaseModel, Field

from src.config._adapter_setting import AdapterSetting
from src.config._registries import ADAPTER_METHODS_REGISTRY
from src.config.utils import load_yaml_content
from src.interfaces.strategies.adapter_method import AdapterMethod


class AdapterMethodConfig(BaseModel):
    method: type[AdapterMethod]
    settings: AdapterSetting = Field(default_factory=AdapterSetting)


def get_adapter_method_config(file: str) -> AdapterMethodConfig:
    config_file = os.path.join("adapter_method", file)

    content = load_yaml_content(config_file)

    method_str = content.get("method", "")
    method = ADAPTER_METHODS_REGISTRY.get(method_str)
    if method is None:
        raise NotImplementedError(
            f"{method_str} not defined in ADAPTER_METHODS_REGISTRY"
        )

    content["method"] = method

    return AdapterMethodConfig(**content)


if __name__ == "__main__":
    res = get_adapter_method_config("bitfit")
    pass
