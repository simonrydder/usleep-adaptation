import torch

from src.concrete.standard_adapter import StandardAdapter
from src.concrete.standard_data_creater import StandardDataCreater
from src.concrete.standard_model_loader import StandardModelLoader
from src.concrete.standard_model_trainer import StandardModelTrainer
from src.config.config import load_config
from src.config.experiment import Experiment

torch.set_float32_matmul_precision("medium")


def run_experiment(experiment: Experiment, debug: bool = False):
    config = load_config(experiment)

    model_loader = StandardModelLoader(config.model)
    org_model = model_loader.load_pretrained(config)

    dataload_generator = StandardDataCreater(config.data)

    for train, val, test in dataload_generator:
        adapter = StandardAdapter(config.adapter)
        new_model = adapter.adapt(org_model, dataloader=train)

        trainer = StandardModelTrainer(config.trainer, config.experiment).get()

        trainer.validate(org_model, val)
        trainer.test(org_model, test)
        trainer.fit(new_model, train, val)
        trainer.test(new_model, test)

        if debug:
            break


if __name__ == "__main__":
    exp = Experiment(
        dataset="eesm19",
        method="fish",
        model="usleep",
        trainer="usleep_debug_neptune",
    )
    run_experiment(exp, True)
    pass
