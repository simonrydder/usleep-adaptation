from typing import Any

from pydantic import BaseModel


class OptimizerSettings(BaseModel):
    lr: float = 1e-6
    lr_patience: int = 10
    lr_minimum: float = 1e-7
    lr_factor: float = 0.5

    def get_settings(self) -> dict[str, Any]:
        settings = {}

        if self.lr is not None:
            settings["lr"] = self.lr

        if self.lr_patience is not None:
            settings["lr_patience"] = self.lr_patience

        if self.lr_minimum is not None:
            settings["lr_minimum"] = self.lr_minimum

        if self.lr_factor is not None:
            settings["lr_factor"] = self.lr_factor

        return settings
