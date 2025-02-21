import json

from lightning import LightningModule
from lightning.pytorch.utilities.model_summary.model_summary import summarize


def save_model_structure(
    model: LightningModule, path: str, include_parameter_count: bool = True
) -> None:
    """
    Saves the structure of a Lightning model's parameters to a JSON file.

    Args:
        model (LightningModule): The PyTorch Lightning model.
        path (str): The file path where the JSON structure will be saved.

    Example:
        save_model_structure(my_model, "model_structure.json")
    """
    if include_parameter_count:
        model_structure = {
            name: param.numel() for name, param in model.named_parameters()
        }
    else:
        model_structure = {name for name, _ in model.named_parameters()}

    with open(path, "w") as f:
        json.dump(model_structure, f, indent=4)

    print(f"Model structure saved to {path}")


def summarize_lightning(model: LightningModule, depth: int = 1) -> None:
    print(summarize(model, depth))
