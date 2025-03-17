import os

from src.config._registries import (
    ACTIVATION_REGISTRY,
    ADAPTER_METHODS_REGISTRY,
    FORWARD_PASS_REGISTRY,
    MODEL_REGISTRY,
)


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


def validate_forward_pass(name: str) -> str:
    if name in FORWARD_PASS_REGISTRY:
        return name

    raise ValueError(f"{name} is not a valid ForwardPass type")


def validate_activation(name: str) -> str:
    if name in ACTIVATION_REGISTRY:
        return name

    raise ValueError(f"{name} is not a activation in ACTIVATION_REGISTRY")


def validate_split_percentages(split: list[float]) -> list[float]:
    if not len(split) == 3:
        raise ValueError(f"{split} do not have lenght 3.")

    if sum(split) == 1:
        return split

    raise ValueError(f"{split} is not valid split_percentages")


def validate_location(name: str) -> str:
    if name in ["root", "layer", "skip", "contract", "expand", "concat"]:
        return name

    raise ValueError(f"{name} is not a valid location")
