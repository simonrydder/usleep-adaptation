from time import sleep

import torch
from lightning import seed_everything

from src.concrete.standard_adapter import StandardAdapter
from src.concrete.standard_data_creater import ImprovedDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.config.config import load_config
from src.config.experiment import Experiment
from src.utils.logger import add_completed, add_fold, log_size_of_datasets, stop_logger

torch.set_float32_matmul_precision("medium")


def run_experiment(experiment: Experiment):
    config = load_config(experiment)
    seed_everything(experiment.seed)

    model_loader = StandardModelLoader(config.model)
    org_model = model_loader.load_pretrained(config)

    dataload_generator = ImprovedDataCreater(config.data)
    train, val, test = dataload_generator.get_dataloaders(
        experiment.fold, experiment.train_size
    )

    seed_everything(experiment.seed)
    adapter = StandardAdapter(config.adapter)
    new_model = adapter.adapt(org_model, dataloader=train)

    trainer = StandardModelTrainer(config.trainer, config.experiment).get()
    log_size_of_datasets(trainer, train, val, test, config.data.num_batches)
    add_fold(trainer, experiment.fold)

    seed_everything(config.experiment.seed)
    trainer.test(org_model, test)
    trainer.fit(new_model, train, val)
    trainer.test(new_model, test, ckpt_path="best")
    add_completed(trainer)
    stop_logger(trainer)
    print("Experiment: {get_experiment_name(experiment)} is done.")
    sleep(0.5)


if __name__ == "__main__":
    exp = Experiment(
        key="test",
        dataset="eesm19",
        method="LoRA10",
        model="usleep",
        trainer="usleep_debug_neptune",
        train_size=None,
        fold=1,
        seed=42,
    )
    run_experiment(exp)
    pass
