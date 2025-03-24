import json
import os
from typing import Any

import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from src.concrete.standard_adapter import StandardAdapter
from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.config.config import load_config
from src.config.experiment import Experiment, get_experiment_name
from src.models.usleep import UsleepLightning


def run_experiment(experiment: Experiment, debug: bool = False):
    config = load_config(experiment)

    model_loader = StandardModelLoader(config.model)
    org_model = model_loader.load_pretrained()

    dataload_generator = StandardDataCreater(config.data)

    org_pred_error = {}
    new_pred_error = {}
    for train, val, test in dataload_generator:
        adapter = StandardAdapter(config.adapter)
        new_model = adapter.adapt(org_model, dataloader=train)

        trainer = StandardModelTrainer(config.trainer, config.experiment).get()

        org_pred_error = update_predictions(org_model, test, trainer, org_pred_error)
        trainer.fit(new_model, train, val)
        new_pred_error = update_predictions(new_model, test, trainer, new_pred_error)

        if debug:
            break

    if not debug:
        save_predictions(experiment, org_pred_error, new_pred_error)


def update_predictions(
    model: LightningModule, test: DataLoader, trainer: Trainer, results: dict
) -> dict:
    if model is not UsleepLightning:
        trainer.test(model, test)
        return {}

    prediction = trainer.predict(model, test)
    assert prediction is not None, f"predict_step not implemented for {model}"
    for pred, y_true, index in prediction:
        results[index[0]] = model.compute_metrics(pred, y_true)


def save_predictions(
    experiment: Experiment, original: dict[str, Any], new: dict[str, Any]
) -> None:
    base_filename = get_experiment_name(experiment) + ".json"
    directory = "results"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(directory, base_filename)

    # Increment filename if it already exists
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(directory, f"{base_filename[:-5]}_{counter}.json")
        counter += 1

    # Save original and new predictions in JSON format
    original_safe = convert_json_serializable(original)
    new_safe = convert_json_serializable(new)

    # Save to JSON
    with open(file_path, "w") as f:
        json.dump({"original": original_safe, "new": new_safe}, f, indent=4)

    print(f"Predictions saved to {file_path}")


def convert_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable formats."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensor to a list
    elif isinstance(obj, dict):
        return {
            str(k): convert_json_serializable(v) for k, v in obj.items()
        }  # Convert dict keys and values
    elif isinstance(obj, list):
        return [convert_json_serializable(v) for v in obj]  # Convert list elements
    elif isinstance(obj, tuple):
        return tuple(
            convert_json_serializable(v) for v in obj
        )  # Convert tuple elements
    return obj  # Return as is if already serializable


if __name__ == "__main__":
    exp = Experiment(
        dataset="eesm19",
        method="fish",
        model="usleep",
        trainer="usleep_debug",  # _neptune",
    )
    run_experiment(exp, True)
    pass
