from lightning import Trainer

from src.config._trainer_config import TrainerConfig
from src.interfaces.model_trainer import ModelTrainer
from src.utils.trainer import define_trainer


class StandardModelTrainer(ModelTrainer):
    def __init__(self, trainer: TrainerConfig) -> None:
        super().__init__(trainer)
        self.trainer = trainer

    def get(self) -> Trainer:
        return define_trainer(
            max_epochs=self.trainer.max_epochs,
            accelerator=self.trainer.accelerator,
            callbacks=self.trainer.get_callbacks(),
            logger=True,
        )
