from typing import Any

from pydantic import BaseModel


class OptimizerSettings(BaseModel):
    lr: float
    lr_patience: int
    lr_minimum: float
    lr_factor: float

    def get_settings(self) -> dict[str, Any]:
        return self.model_dump()
