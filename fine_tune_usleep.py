import os

import pytorch_lightning as pl
import torch
from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_pipeline.pipeline_elements.models import Split
from csdp_pipeline.pipeline_elements.samplers import (
    Determ_sampler,
    Random_Sampler,
    SamplerConfiguration,
)
from csdp_training.lightning_models.usleep import USleep_Lightning

from src.concrete.adapters.bitfit import BitFit
from src.concrete.standard_model_updater import StandardModelUpdater


def main():
    results_folder = f"{os.getcwd()}/lighting_logs"  # Where to save test results
    base_hdf5_path = "data/hdf5"  # Base-path to the HDF5 files. Should contain at least one HDF5 file
    split_file_path = os.getcwd()  # Where to save the generated dataloading split file

    # Path to existing pre-trained weights. Can be used for finetuning or testing.
    pretrained_path = "data/ckpt/alternative_big_sleep.ckpt"

    batch_size = 64
    learning_rate = 0.0001
    complexity_factor = 0.5
    depth = 11
    max_training_epochs = 2

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    sleep_epochs_per_sample = 35  # Number of sleep epochs per sample
    num_batches = 3  # Number of batches per training epoch
    training_iterations = (
        batch_size * num_batches
    )  # The resulting number of sampling iterations in the data
    dataloader_workers = (
        1  # Number of dataloader workers. Set to number of CPU cores available.
    )

    # Create a random subject-based split in the data. Can be dumped to a file, which can later be used with Split.File(....)
    split = Split.random(
        base_hdf5_path=base_hdf5_path,
        split_name="demo",
        split_percentages=(0.4, 0.3, 0.3),
    )
    split.dump_file(path=split_file_path)

    train_sampler = Random_Sampler(
        split, num_epochs=sleep_epochs_per_sample, num_iterations=training_iterations
    )

    val_sampler = Determ_sampler(split, split_type="val")

    test_sampler = Determ_sampler(split, split_type="test")

    samplers = SamplerConfiguration(train_sampler, val_sampler, test_sampler)

    data_fac = Dataloader_Factory(training_batch_size=batch_size, samplers=samplers)

    # Load a clean model or from a checkpoint
    if pretrained_path == None:
        net: USleep_Lightning = USleep_Lightning(
            lr=learning_rate,
            batch_size=batch_size,
            complexity_factor=complexity_factor,
            depth=depth,
        )
    else:
        net: USleep_Lightning = USleep_Lightning.load_from_checkpoint(pretrained_path)

    updater = StandardModelUpdater(BitFit)
    new_model = updater.adapt(net)

    trainer = pl.Trainer(
        max_epochs=max_training_epochs, accelerator=accelerator, devices=1, num_nodes=1
    )

    train_loader = data_fac.training_loader(num_workers=dataloader_workers)
    val_loader = data_fac.validation_loader(num_workers=dataloader_workers)
    test_loader = data_fac.testing_loader(num_workers=dataloader_workers)

    net.run_test(
        trainer, test_loader, output_folder_prefix=results_folder, load_best_model=False
    )
    trainer.fit(new_model, train_loader, val_loader)
    # Start test and output results to a specific folder
    new_model.run_test(trainer, test_loader, output_folder_prefix=results_folder)


if __name__ == "__main__":
    main()
