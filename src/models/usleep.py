from typing import Any, Literal

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from csdp.csdp_training.utility import acc, f1, kappa
from csdp.ml_architectures.usleep.usleep import USleep


class UsleepLightning(LightningModule):
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
    ) -> None:
        super().__init__()

        num_channels = 2 if include_eog is True else 1

        self.model = USleep(
            num_channels=num_channels,
            initial_filters=initial_filters,
            complexity_factor=complexity_factor,
            progression_factor=progression_factor,
            depth=depth,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_minimum = lr_minimum
        self.loss_weights = loss_weights
        self.training_step_outputs = []
        self.validation_step_loss = []
        self.validation_step_acc = []
        self.validation_step_kap = []
        self.validation_step_f1 = []
        self.validation_preds = []
        self.validation_labels = []

        weights = torch.tensor(loss_weights) if loss_weights is not None else None
        self.loss = nn.CrossEntropyLoss(weight=weights, ignore_index=5)

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

    def _step(
        self,
        batch: dict[str, Any],
        batch_index: int,
        type: Literal["test", "val", "train"],
    ) -> Tensor:
        x_eeg, x_eog = batch["eeg"], batch["eog"]
        x = self._prep_batch(x_eeg, x_eog)
        y = batch["labels"]

        pred = self(x)

        loss, acc, kappa, f1 = self._compute_metrics(pred, y)
        # self.training_step_outputs.append(loss)  # Maybe remove

        self.log(f"{type}_loss", loss, prog_bar=True)
        self.log(f"{type}_acc", acc, prog_bar=True)
        self.log(f"{type}_kappa", kappa, prog_bar=True)
        self.log(f"{type}_f1", f1, prog_bar=True)

        return loss

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
            "monitor": "val_kappa",
            "lr_scheduler": scheduler,
        }  # type: ignore

    def _prep_batch(self, x_eeg: Tensor, x_eog: Tensor) -> Tensor:
        assert (
            len(x_eeg.shape) == 3
        ), "EEG shape must be on the form (batch_size, num_channels, data)"
        assert x_eeg.shape[1] == 1, "Only one EEG channel allowed"

        if self.include_eog:
            assert (
                len(x_eog.shape) == 3
            ), "EOG shape must be on the form (batch_size, num_channels, data)"
            assert x_eog.shape[1] == 1, "Only one EOG channel allowed"
            xbatch = torch.cat((x_eeg, x_eog), dim=1)
        else:
            xbatch = x_eeg

        return xbatch

    def _compute_metrics(
        self, y_pred: Tensor, y_true: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y_pred = torch.swapdims(y_pred, 1, 2)
        y_pred = torch.reshape(y_pred, (-1, 5))
        y_true = torch.flatten(y_true)

        loss = self.loss(y_pred, y_true.long())

        y_pred = torch.argmax(y_pred, dim=1)

        # try:
        accu = acc(y_pred, y_true)
        kap = kappa(y_pred, y_true, 5)
        f1_score = f1(y_pred, y_true, average=True)
        # except:
        #     accu = None
        #     kap = None
        #     f1_score = None

        return loss, accu, kap, f1_score
