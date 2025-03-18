import os

from lightning import LightningModule
from pydantic import BaseModel

from src.config._registries import MODEL_REGISTRY
from src.config.utils import load_yaml_content


class ModelConfig(BaseModel):
    model: type[LightningModule]
    ckpt: str


def get_model_config(file: str) -> ModelConfig:
    model_config_file = os.path.join("model", file)
    content = load_yaml_content(model_config_file)

    model_str = content.get("model", "")
    model = MODEL_REGISTRY.get(model_str)
    if model is None:
        raise NotImplementedError(f"{model_str} not defined in MODEL_REGISRTY")

    content["model"] = model
    return ModelConfig(**content)


if __name__ == "__main__":
    res = get_model_config("usleep")
    pass
