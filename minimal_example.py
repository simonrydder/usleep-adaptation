import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)

from src.csdp_pipeline.factories.dataloader_factory import USleep_Dataloader_Factory
from src.csdp_training.lightning_models.factories.lightning_model_factory import (
    USleep_Factory,
)


def main(): ...


def minimal_test_dataloader() -> torch.utils.data.DataLoader:
    dl_fac = USleep_Dataloader_Factory(
        gradient_steps=5,
        batch_size=64,
        hdf5_base_path=os.path.join("data", "hdf5"),
        trainsets=["eesm19"],
        testsets=["eesm19"],
        valsets=["eesm19"],
        data_split_path=os.path.join("data", "splits", "test.json"),
        create_random_split=False,
    )

    return dl_fac.create_testing_loader(num_workers=1)


def minimal_pretrained_model() -> pl.LightningModule:
    usleep_factory = USleep_Factory(
        lr=0.0001,
        batch_size=64,
        initial_filters=5,
        complexity_factor=0.5,
        progression_factor=2,
    )
    # ckpt = os.path.join("src", "usleep", "weights", "Depth10_CF05.ckpt")
    ckpt = os.path.join("src", "usleep", "weights", "EESM19_finetuned.ckpt")
    pretrained = usleep_factory.create_pretrained_net(ckpt)

    return pretrained


def minimal_trainer(accelerator: str = "gpu") -> pl.Trainer:
    early_stopping = EarlyStopping(
        monitor="valKap", min_delta=0.00, patience=50, verbose=True, mode="max"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(monitor="valKap", mode="max")
    timer = Timer()

    callbacks = [early_stopping, timer, lr_monitor, checkpoint_callback]

    trainer = pl.Trainer(
        logger=True,
        max_epochs=2,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        num_nodes=1,
    )

    return trainer


if __name__ == "__main__":
    dl = minimal_test_dataloader()
    model = minimal_pretrained_model()
    trainer = minimal_trainer("gpu")
    with torch.no_grad():
        model.eval()
        _ = trainer.test(model, dl)

    import torch

    checkpoint_path = "src/usleep/weights/EESM19_finetuned.ckpt"
    checkpoint_path = "src/usleep/weights/Depth10_CF05.ckpt"
    checkpoint_path = "src/usleep/weights/m1-m2_finetuned.ckpt"
    checkpoint_path = "src/usleep/weights/usleep_onechannel.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Inspect the keys to understand how the model state is stored
    print(checkpoint.keys())

    checkpoint["state_dict"]
    # main()
