from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict

from src.config._registries import FORWARD_PASS_REGISTRY
from src.config._validators import validate_forward_pass


class AdapterSetting(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank: int = 0
    alpha: int = 1
    dropout: float = 0
    forward_pass: Annotated[str, validate_forward_pass] | None = None
    residual: bool | None = None

    def get_settings(self) -> dict[str, Any]:
        settings = {}

        if self.forward_pass is not None:
            settings["forward_pass"] = FORWARD_PASS_REGISTRY[self.forward_pass]

        if self.residual is not None:
            settings["residual"] = self.residual

        if self.rank is not None:
            settings["rank"] = self.rank
        if self.alpha is not None:
            settings["alpha"] = self.alpha

        if self.dropout is not None:
            settings["dropout"] = self.dropout
        return settings
