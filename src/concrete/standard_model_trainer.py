from lightning import Trainer

from src.interfaces.model_trainer import ModelTrainer
from src.utils.callbacks import early_stopping, timer
from src.utils.trainer import define_trainer


class StandardModelTrainer(ModelTrainer):
    def __init__(self) -> None:
        super().__init__()

    def get(self) -> Trainer:
        return define_trainer(
            max_epochs=1,
            accelerator="gpu",
            callbacks=[early_stopping("train_loss", 5, "min"), timer()],
            logger=True,
        )
