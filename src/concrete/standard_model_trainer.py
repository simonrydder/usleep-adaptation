from lightning import Trainer

from src.config._trainer_config import TrainerConfig
from src.config.experiment import Experiment, get_experiment_name
from src.interfaces.model_trainer import ModelTrainer
from src.utils.logger import neptune_logger
from src.utils.trainer import define_trainer


class StandardModelTrainer(ModelTrainer):
    def __init__(self, trainer: TrainerConfig, experiment: Experiment) -> None:
        super().__init__(trainer, experiment)
        self.trainer = trainer

        self.logger = (
            neptune_logger(name=get_experiment_name(experiment))
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
