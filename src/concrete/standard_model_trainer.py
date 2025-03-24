from lightning import Trainer

from src.config._trainer_config import TrainerConfig
from src.interfaces.model_trainer import ModelTrainer
from src.utils.logger import neptune_logger
from src.utils.trainer import define_trainer


class StandardModelTrainer(ModelTrainer):
    def __init__(self, trainer: TrainerConfig, experiment_name: str) -> None:
        super().__init__(trainer)
        self.trainer = trainer

        self.logger = (
            neptune_logger(name=experiment_name)
            if trainer.logger == "neptune"
            else True
        )

    def get(self) -> Trainer:
        return define_trainer(
            max_epochs=self.trainer.max_epochs,
            accelerator=self.trainer.accelerator,
            callbacks=self.trainer.get_callbacks(),
            logger=self.logger,
        )
