import os

from src.config._registries import (
    ADAPTER_METHODS_REGISTRY,
    FORWARD_PASS_REGISTRY,
    MODEL_REGISTRY,
)
from src.interfaces.strategies.forward_pass import ForwardPass


def validate_model_name(name: str) -> str:
    """Ensures the model name is valid and exists in the registry."""
    if name in MODEL_REGISTRY:
        return name

    raise ValueError(f"{name} not a model in MODEL_REGISTRY")


def validate_file_existence(file: str) -> str:
    if os.path.exists(file):
        return file

    raise ValueError(f"{file = } does not exists.")


def validate_folder_existence(folder: str | list[str]) -> str:
    if isinstance(folder, list):
        folder = os.path.join(*folder)

    if os.path.exists(folder):
        return folder

    raise ValueError(f"{folder = } does not exists.")


def validate_adapter_name(name: str) -> str:
    if name in ADAPTER_METHODS_REGISTRY:
        return name

    raise ValueError(f"{name} not a model in ADAPTER_REGISTRY")


def validate_forward_pass(name: str) -> ForwardPass:
    if name in FORWARD_PASS_REGISTRY:
        return FORWARD_PASS_REGISTRY[name]

    raise ValueError(f"{name} is not a valid ForwardPass type")
