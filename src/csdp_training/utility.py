import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score
from torchmetrics.functional import (
    accuracy,  # pytorch_lightning.metrics.Accuracy is deprecated
)


def log_test_step(
    base: str, run_id: str, dataset: str, subject: str, record: str, **kwargs: Any
) -> None:
    """
    Log raw predictions and true labels for a single step.

    Args:
        base (str): Base directory path where logs will be saved.
        run_id (str): Identifier for the current run.
        dataset (str): Name of the dataset.
        subject (str): Subject identifier.
        record (str): Record identifier.
        **kwargs (Any): Additional data to log (predictions, labels, etc.).
    """
    identifier = f"{dataset}.{subject}.{record}"
    print(f"logging for: {dataset}/{identifier}")
    print(f"kwargs: {kwargs}")

    path = f"{base}/{run_id}/{dataset}"
    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"{path}/{identifier}"
    print(f"log preds and labels to file: {filename}")

    with open(filename, "ab") as f:
        pickle.dump(kwargs, f)


def filter_unknowns(
    predictions: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out any 'unknown' labels (assumed to have value == 5).

    Args:
        predictions (torch.Tensor): Predicted labels.
        labels (torch.Tensor): True labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Filtered predictions and labels.
    """
    mask = labels != 5
    labels_filtered = torch.masked_select(labels, mask)
    preds_filtered = torch.masked_select(predictions, mask)

    assert len(labels_filtered) == len(preds_filtered)

    return preds_filtered, labels_filtered


def plot_confusionmatrix(
    conf: np.ndarray,
    title: str,
    percentages: bool = True,
    formatting: str = ".2f",
    save_figure: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix using Seaborn heatmap.

    Args:
        conf (np.ndarray): Confusion matrix data.
        title (str): Title of the plot.
        percentages (bool, optional): Whether to convert counts to percentages. Defaults to True.
        formatting (str, optional): Format for annot text. Defaults to '.2f'.
        save_figure (bool, optional): Whether to save the figure. Defaults to False.
        save_path (Optional[str], optional): If save_figure=True, path to save the figure. Defaults to None.

    Returns:
        plt.Figure: Matplotlib Figure object for the confusion matrix.
    """
    if percentages:
        conf = conf.astype("float") / conf.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(
        conf,
        index=["Wake", "N1", "N2", "N3", "REM"],
        columns=["Wake", "N1", "N2", "N3", "REM"],
    )

    plt.title(title)
    plt.figure(figsize=(10, 7))

    f = sn.heatmap(df_cm, annot=True, fmt=formatting)
    f.set(xlabel="Predicted", ylabel="Truth")

    # Optional saving
    if save_figure and save_path is not None:
        plt.savefig(save_path)

    return f.figure


def kappa(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int = 5
) -> torch.Tensor:
    """
    Compute Cohen's Kappa ignoring unknown class (label == 5).

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground-truth labels.
        num_classes (int, optional): Number of classes (excluding unknown). Defaults to 5.

    Returns:
        torch.Tensor: Cohen's Kappa score.
    """
    preds, labs = filter_unknowns(predictions, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = preds.to(device)
    labs = labs.to(device)

    metric = MulticlassCohenKappa(num_classes=num_classes).to(device)
    score = metric(preds, labs)
    return score


def acc(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy ignoring unknown class (label == 5).

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground-truth labels.

    Returns:
        torch.Tensor: Accuracy score.
    """
    preds, labs = filter_unknowns(predictions, labels)
    return accuracy(task="multiclass", num_classes=5, preds=preds, target=labs)


def f1(
    predictions: torch.Tensor, labels: torch.Tensor, average: bool = True
) -> torch.Tensor:
    """
    Compute F1 score ignoring unknown class (label == 5).

    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground-truth labels.
        average (bool, optional): If True, return the average F1; otherwise, return per-class F1. Defaults to True.

    Returns:
        torch.Tensor: F1 score (scalar if average=True, else per-class scores).
    """
    preds, labs = filter_unknowns(predictions, labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = preds.to(device)
    labs = labs.to(device)

    if average:
        metric = MulticlassF1Score(num_classes=5).to(device)
    else:
        metric = MulticlassF1Score(num_classes=5, average=None).to(device)

    score = metric(preds, labs)
    return score


def create_split_file(hdf5_basepath: str) -> str:
    """
    Create a random train/val/test split file for the HDF5 dataset.

    Args:
        hdf5_basepath (str): Directory containing .hdf5 files.

    Returns:
        str: The name of the output JSON file containing splits.
    """
    output_name = "random_split.json"
    hdf5_paths = os.listdir(hdf5_basepath)
    output_dic: Dict[str, Dict[str, List[str]]] = dict()

    for path in hdf5_paths:
        with h5py.File(f"{hdf5_basepath}/{path}", "r") as hdf5_file:
            subs = list(hdf5_file.keys())
            dataset_name = path.replace(".hdf5", "")

            train, test = train_test_split(subs, train_size=0.80, test_size=0.20)
            val, test = train_test_split(test, train_size=0.5, test_size=0.5)

            output_dic[dataset_name] = {"train": train, "val": val, "test": test}

    json_object = json.dumps(output_dic, indent=4)
    print(json_object)

    with open(output_name, "w") as fp:
        fp.write(json_object)

    return output_name
