# TODO: Maybe something to use.
import os
from typing import Annotated

from pydantic import AfterValidator, BaseModel

from src.config._validators import validate_folder_existence


class CheckpointConfig(BaseModel):
    folder: Annotated[str, AfterValidator(validate_folder_existence)]
    file: str

    def get_file(self) -> str:
        return os.path.join(self.folder, self.file)
