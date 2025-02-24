from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from src.config._adapter_setting import AdapterSetting
from src.config._registries import ADAPTER_METHODS_REGISTRY
from src.config._validators import validate_adapter_name
from src.interfaces.strategies.adapter_method import AdapterMethod


class AdapterConfig(BaseModel):
    type: Annotated[str, AfterValidator(validate_adapter_name)]
    settings: AdapterSetting = Field(default_factory=AdapterSetting)

    def get_adapter_method(self) -> AdapterMethod:
        adapter_class = ADAPTER_METHODS_REGISTRY[self.type]
        return adapter_class(**self.settings.get_settings())
