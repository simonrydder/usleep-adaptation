from typing import Annotated, Any

from pydantic import BaseModel

from src.config._registries import FORWARD_PASS_REGISTRY
from src.config._validators import validate_forward_pass


class AdapterSetting(BaseModel):
    forward_pass: Annotated[str, validate_forward_pass] | None = None

    def get_settings(self) -> dict[str, Any]:
        settings = {}

        if self.forward_pass is not None:
            settings["forward_pass"] = FORWARD_PASS_REGISTRY[self.forward_pass]

        return settings
