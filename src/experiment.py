import torch

from src.concrete.standard_adapter import StandardAdapter
from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.config.config import load_config
from src.config.experiment import Experiment
from src.utils.logger import add_fold, add_tags, log_size_of_datasets

torch.set_float32_matmul_precision("medium")


def run_experiment(experiment: Experiment, debug: bool = False):
    config = load_config(experiment)

    model_loader = StandardModelLoader(config.model)
    org_model = model_loader.load_pretrained(config)

    dataload_generator = StandardDataCreater(config.data)

    for fold, (train, val, test) in enumerate(dataload_generator):
        adapter = StandardAdapter(config.adapter)
        new_model = adapter.adapt(org_model, dataloader=train)

        trainer = StandardModelTrainer(config.trainer, config.experiment).get()
        log_size_of_datasets(trainer, train, val, test)
        add_tags(
            trainer,
            config.experiment.dataset,
            config.experiment.method,
            config.experiment.model,
            str(config.experiment.id),
        )
        add_fold(trainer, fold)

        trainer.test(org_model, test)
        trainer.fit(new_model, train, val)
        trainer.test(new_model, test, ckpt_path="best")

        del trainer
        if debug:
            break


if __name__ == "__main__":
    exp = Experiment(dataset="eesm19", method="SCA10", model="usleep", trainer="usleep")
    run_experiment(exp, False)
    pass
