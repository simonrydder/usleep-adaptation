from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from src.config._adapter_setting import AdapterSetting
from src.config._registries import ADAPTER_REGISTRY
from src.config._validators import validate_adapter_name
from src.interfaces.adapter import Adapter


class AdapterConfig(BaseModel):
    type: Annotated[str, AfterValidator(validate_adapter_name)]
    settings: AdapterSetting = Field(default_factory=AdapterSetting)

    def get_adapter(self) -> Adapter:
        adapter_class = ADAPTER_REGISTRY[self.type]
        return adapter_class(**self.settings.get_settings())
