import os

from pydantic import BaseModel, field_serializer

from src.config._registries import MODEL_REG
from src.config.utils import load_yaml_content
from src.interfaces.framework_model import FrameworkModel


class ModelConfig(BaseModel):
    model: type[FrameworkModel]
    ckpt: str

    @field_serializer("model")
    def serialize_class(self, v):
        return str(v)


def get_model_config(file: str) -> ModelConfig:
    model_config_file = os.path.join("model", file)
    content = load_yaml_content(model_config_file)

    model_str = content.get("model", "")
    content["model"] = MODEL_REG.lookup(model_str)

    return ModelConfig(**content)


if __name__ == "__main__":
    res = get_model_config("usleep")
    pass
