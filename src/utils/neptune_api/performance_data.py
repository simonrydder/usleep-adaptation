from neptune import Run
from pydantic import BaseModel

from src.utils.neptune_api.neptune_api import get_data_series, get_run


class PerformanceData(BaseModel):
    record: str
    kappa: float
    accuracy: float
    loss: float
    f1: float


def get_performance_data(run: Run, mode: str, type: str) -> list[PerformanceData]:
    folder = f"training/{mode}/{type}"
    records = get_data_series(run, f"{folder}/records")["value"].tolist()
    kappas = get_data_series(run, f"{folder}/kappa_step")["value"].tolist()
    accuracies = get_data_series(run, f"{folder}/accuracy_step")["value"].tolist()
    losses = get_data_series(run, f"{folder}/loss_step")["value"].tolist()
    f1s = get_data_series(run, f"{folder}/f1_step")["value"].tolist()

    return [
        PerformanceData(record=rec, kappa=kappa, accuracy=acc, loss=loss, f1=f1)
        for rec, kappa, acc, loss, f1 in zip(records, kappas, accuracies, losses, f1s)
    ]


if __name__ == "__main__":
    run = get_run("US-523")
    x = get_performance_data(run, "new", "val")
    y = get_performance_data(run, "org", "test")
    pass
