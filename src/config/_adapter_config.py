import os

from pydantic import BaseModel, Field

from src.config._adapter_setting import AdapterSetting
from src.config._registries import ADAPTER_METHOD_REG, PARAMETER_COUNT_REG
from src.config.utils import load_yaml_content
from src.interfaces.strategies.adapter_method import AdapterMethod
from src.interfaces.strategies.parameter_count_method import ParameterCountMethod


class AdapterMethodConfig(BaseModel):
    method: type[AdapterMethod]
    settings: AdapterSetting = Field(default_factory=AdapterSetting)
    parameter_count: type[ParameterCountMethod]


def get_adapter_method_config(file: str) -> AdapterMethodConfig:
    config_file = os.path.join("adapter_method", file)

    content = load_yaml_content(config_file)

    method = content.get("method")
    assert method is not None, (
        f"yaml-file: {config_file} does not include `method` key."
    )

    content["method"] = ADAPTER_METHOD_REG.lookup(method)

    param_count = content.get("parameter_count")
    if param_count is None:
        param_count = "skip"

    content["parameter_count"] = PARAMETER_COUNT_REG.lookup(param_count)

    return AdapterMethodConfig(**content)


if __name__ == "__main__":
    res = get_adapter_method_config("bitfit")
    pass
