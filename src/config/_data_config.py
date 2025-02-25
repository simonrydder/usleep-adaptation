from pydantic import BaseModel, Field

from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.usleep_data_creater import UsleepDataCreater
from src.config._data_settings import DataSetting
from src.dataset.resnet.simple_images import SimpleImages
from src.dataset.simple.simple_linear import SimpleLinear
from src.interfaces.data_creater import DataCreater


class DataConfig(BaseModel):
    dataset: str
    settings: DataSetting = Field(default_factory=DataSetting)

    def get_data_creater(self) -> DataCreater:
        if self.dataset == "images":
            image_settings = self.settings.get_settings()
            return StandardDataCreater(SimpleImages(**image_settings))

        if self.dataset == "linear":
            linear_setting = self.settings.get_settings()
            return StandardDataCreater(SimpleLinear(**linear_setting))

        usleep_setting = self.settings.get_settings()

        if not self.dataset.endswith(".hdf5"):
            self.dataset = self.dataset + ".hdf5"

        return UsleepDataCreater(dataset=self.dataset, **usleep_setting)
