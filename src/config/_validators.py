import os

from src.config.config import MODEL_REGISTRY


def validate_model_name(name: str) -> str:
    """Ensures the model name is valid and exists in the registry."""
    if name in MODEL_REGISTRY:
        return name

    raise ValueError(f"{name} not a model in MODEL_REGISTRY")


def validate_file_existence(file: str) -> str:
    if os.path.exists(file):
        return file

    raise ValueError(f"{file = } does not exists.")
