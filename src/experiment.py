import json
import os
from typing import Any

import torch

from src.concrete.standard_adapter import StandardAdapter
from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.config.config import load_config
from src.config.experiment import Experiment


def run_experiment(experiment: Experiment):
    config = load_config(experiment)

    model_loader = StandardModelLoader(config.model)
    original_model = model_loader.load_pretrained()

    dataload_generator = StandardDataCreater(config.data)

    original_results = {}
    fine_tuned_results = {}
    for train, val, test in dataload_generator:
        adapter = StandardAdapter(config.adapter)
        new_model = adapter.adapt(original_model, dataloader=train)

        trainer = StandardModelTrainer(config.trainer).get()

        original_predictions = trainer.predict(original_model, test)
        assert original_predictions is not None, (
            "Original model needs an predict_step function"
        )

        for pred, y_true, index in original_predictions:
            original_results[index[0]] = original_model._compute_metrics(pred, y_true)

        trainer.fit(new_model, train, val)

        new_predictions = trainer.predict(new_model, test)
        assert new_predictions is not None, "New model needs an predict_step function"

        for pred, y_true, index in new_predictions:
            fine_tuned_results[index[0]] = new_model._compute_metrics(pred, y_true)

    save_predictions(experiment, original_results, fine_tuned_results)


def save_predictions(
    experiment: Experiment, original: dict[str, Any], new: dict[str, Any]
) -> None:
    base_filename = "_".join(experiment.model_dump().values()) + ".json"
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
        method="bitfit",
        model="usleep",
        trainer="usleep_debug",
    )
    run_experiment(exp)
    pass
