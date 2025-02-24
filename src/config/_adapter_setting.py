from typing import Annotated, Any

from pydantic import BaseModel

from src.config._registries import ACTIVATION_REGISTRY, FORWARD_PASS_REGISTRY
from src.config._validators import validate_activation, validate_forward_pass


class AdapterSetting(BaseModel):
    forward_pass: Annotated[str, validate_forward_pass] | None = None
    reduction: int | None = None
    activation: Annotated[str, validate_activation] | None = None
    kernel: int | tuple[int] | tuple[int, int] | None = None

    def get_settings(self) -> dict[str, Any]:
        settings = {}

        if self.forward_pass is not None:
            settings["forward_pass"] = FORWARD_PASS_REGISTRY[self.forward_pass]

        if self.reduction is not None:
            settings["reduction"] = self.reduction

        if self.activation is not None:
            settings["activation"] = ACTIVATION_REGISTRY[self.activation]

        if self.kernel is not None:
            settings["kernel"] = self.kernel

        return settings
