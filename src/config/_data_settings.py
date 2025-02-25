from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel

from src.config._validators import validate_split_percentages


class DataSetting(BaseModel):
    num_samples: int | None = None
    num_classes: int | None = None
    distribution: Literal[1, 2] | Literal["normal", "shifted", "uniform"] | None = None
    batch_size: int | None = None
    split_percentages: (
        Annotated[list[float], AfterValidator(validate_split_percentages)] | None
    ) = None
    num_workers: int | None = None

    def get_settings(self) -> dict[str, Any]:
        settings = {}

        if self.num_samples is not None:
            settings["num_samples"] = self.num_samples

        if self.num_classes is not None:
            settings["num_classes"] = self.num_classes

        if self.distribution is not None:
            settings["distribution"] = self.distribution

        if self.batch_size is not None:
            settings["batch_size"] = self.batch_size

        if self.split_percentages is not None:
            settings["split_percentages"] = self.split_percentages

        if self.num_workers is not None:
            settings["num_workers"] = self.num_workers

        return settings
