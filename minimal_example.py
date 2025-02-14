import os

import pytorch_lightning as pl
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


def main():
    usleep_factory = USleep_Factory(
        lr=0.0001,
        batch_size=64,
        initial_filters=5,
        complexity_factor=0.5,
        progression_factor=2,
        depth=5,
    )
    # ckpt = os.path.join("src", "usleep", "weights", "Depth10_CF05.ckpt")
    ckpt = os.path.join("src", "usleep", "weights", "Depth10_CF05.ckpt")
    new_net = usleep_factory.create_new_net()

    dataloader = USleep_Dataloader_Factory(
        gradient_steps=5,
        batch_size=64,
        hdf5_base_path=os.path.join("data", "hdf5"),
        trainsets=["sedf_st"],
        testsets=["sedf_st"],
        valsets=["sedf_st"],
        data_split_path=os.path.join("data", "splits", "usleep_split.json"),
        create_random_split=False,
    )
    early_stopping = EarlyStopping(
        monitor="valKap", min_delta=0.00, patience=50, verbose=True, mode="max"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(monitor="valKap", mode="max")
    timer = Timer()

    callbacks = [early_stopping, timer, lr_monitor, checkpoint_callback]

    trainer = pl.Trainer(
        logger=True,
        max_epochs=200,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        num_nodes=1,
    )

    train_loader = dataloader.create_training_loader(num_workers=5)
    val_loader = dataloader.create_validation_loader(num_workers=5)

    trainer.fit(new_net, train_loader, val_loader)


if __name__ == "__main__":
    main()
