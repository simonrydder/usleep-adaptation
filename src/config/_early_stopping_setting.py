from typing import Literal

from pydantic import BaseModel


class EarlyStoppingSetting(BaseModel):
    monitor: str
    patience: int = 25
    mode: Literal["min", "max"]
    min_delta: float = 0.0001
