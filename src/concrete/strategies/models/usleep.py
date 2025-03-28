from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from csdp.csdp_training.utility import acc, f1, kappa
from csdp.ml_architectures.usleep.usleep import USleep
from src.interfaces.framework_model import FrameworkModel


class UsleepModel(FrameworkModel):
    def __init__(
        self,
        lr: float,
        batch_size: int,
        initial_filters: int = 5,
        complexity_factor: float = 1.67,
        progression_factor: int = 2,
        depth: int = 12,
        lr_patience: int = 50,
        lr_factor: float = 0.5,
        lr_minimum: float = 0.0000001,
        loss_weights=None,
        include_eog: bool = True,
    ):
        super().__init__()

        num_channels = 2 if include_eog is True else 1

        self.model = USleep(
            num_channels=num_channels,
            initial_filters=initial_filters,
            complexity_factor=complexity_factor,
            progression_factor=progression_factor,
            depth=depth,
        )

        weights = torch.tensor(loss_weights) if loss_weights is not None else None
        self.loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=5)

        self.lr = lr
        self.batch_size = batch_size
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_minimum = lr_minimum
        self.loss_weights = loss_weights

        self.save_hyperparameters(ignore=["model"])

        self.log_to_progress_bar = False

        self.prediction_resolution = 3840
        self.initial_filters = initial_filters
        self.complexity_factor = complexity_factor
        self.progression_factor = progression_factor
        self.depth = depth
        self.include_eog = include_eog
        self.num_channels = num_channels

    def forward(self, x: Tensor | dict[str, Any]) -> Tensor:
        if isinstance(x, dict):
            x = self._prep_batch(x["eeg"], x["eog"])

        y = self.model(x.float())
        return y

    def training_step(self, batch: dict[str, Any], batch_index: int) -> Tensor:
        return self._step(batch, batch_index, "train")

    def validation_step(self, batch: dict[str, Any], batch_index: int) -> Tensor:
        return self._step(batch, batch_index, "val")

    def test_step(self, batch: dict[str, Any], batch_index: int) -> Tensor:
        return self._step(batch, batch_index, "test")

    def predict_step(
        self, batch: Any, batch_index: int
    ) -> tuple[Tensor, Tensor, Sequence[Any]]:
        x_eeg, x_eog = batch["eeg"], batch["eog"]
        x = self._prep_batch(x_eeg, x_eog)
        y_true = torch.flatten(batch["labels"])

        subjects = batch["tag"]["subject"]
        records = batch["tag"]["record"]

        records = list(zip(subjects, records))

        y_pred = self(x)
        y_pred = torch.swapdims(y_pred, 1, 2)
        y_pred = torch.reshape(y_pred, (-1, 5))

        return y_pred, y_true, records

    def _step(
        self,
        batch: dict[str, Any],
        batch_index: int,
        type: Literal["test", "val", "train"],
    ) -> Tensor:
        pred, y, _ = self.predict_step(batch, batch_index)
        model = "org" if getattr(self, "original_model") else "new"
        measurements = self.measurements(pred, y)
        for key, val in measurements.items():
            self.log(
                f"{model}/{type}/{key}",
                val,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )

        return self.loss(pred, y)

    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.lr_factor,
            patience=self.lr_patience,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=self.lr_minimum,
            eps=1e-8,
        )

        return {
            "optimizer": optimizer,
            "monitor": "new/val/kappa",
            "lr_scheduler": scheduler,
        }  # type: ignore

    def _prep_batch(self, x_eeg: Tensor, x_eog: Tensor) -> Tensor:
        assert len(x_eeg.shape) == 3, (
            "EEG shape must be on the form (batch_size, num_channels, data)"
        )
        assert x_eeg.shape[1] == 1, "Only one EEG channel allowed"

        if self.include_eog:
            assert len(x_eog.shape) == 3, (
                "EOG shape must be on the form (batch_size, num_channels, data)"
            )
            assert x_eog.shape[1] == 1, "Only one EOG channel allowed"
            xbatch = torch.cat((x_eeg, x_eog), dim=1)
        else:
            xbatch = x_eeg

        return xbatch

    def loss(self, prediction: Tensor, true: Tensor) -> Tensor:
        return self.loss_fn(prediction, true.long())

    def measurements(self, prediction: Tensor, true: Tensor) -> dict[str, Tensor]:
        argmax_pred = torch.argmax(prediction, dim=1)
        return {
            "loss": self.loss(prediction, true),
            "accuracy": acc(argmax_pred, true),
            "kappa": kappa(argmax_pred, true, 5),
            "f1": f1(argmax_pred, true, average=True),
        }

    def is_classification_parameter(self, parameter_name: str) -> bool:
        if "dense" in parameter_name:
            return True

        if "classifier" in parameter_name:
            return True

        return False


if __name__ == "__main__":
    UsleepModel(lr=0.0001, batch_size=4)
