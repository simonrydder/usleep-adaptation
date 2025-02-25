import os

import h5py
from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_pipeline.pipeline_elements.models import Dataset_Split, Split
from csdp_pipeline.pipeline_elements.samplers import (
    Determ_sampler,
    Random_Sampler,
    SamplerConfiguration,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.interfaces.data_creater import DataCreater


class UsleepDataCreater(DataCreater):
    def __init__(
        self,
        dataset: str,
        split_percentages: tuple[float, float, float],
        batch_size: int = 64,
        num_workers: int = 1,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.base_hdf5_path = os.path.join("data", "hdf5")
        self.hdf5_file = os.path.join(self.base_hdf5_path, self.dataset)
        split_file_path = os.path.join("data", "splits")

        batch_size = batch_size
        self.dataloader_workers = num_workers
        sleep_epochs_per_sample = 35
        num_batches = 3
        training_iterations = batch_size * num_batches

        self.split_percentages = split_percentages
        split = self._create_splits()
        split.dump_file(path=split_file_path)

        train_sampler = Random_Sampler(
            split, sleep_epochs_per_sample, training_iterations
        )
        val_sampler = Determ_sampler(split, split_type="val")
        test_sampler = Determ_sampler(split, split_type="test")

        samplers = SamplerConfiguration(train_sampler, val_sampler, test_sampler)

        self.data_fac = Dataloader_Factory(
            training_batch_size=batch_size, samplers=samplers
        )

    def _create_splits(self) -> Split:
        with h5py.File(self.hdf5_file, "r") as hdf5:
            subs = list(hdf5["data"].keys())  # type: ignore

            train, test = train_test_split(
                subs, test_size=1 - self.split_percentages[0]
            )
            val, test = train_test_split(
                test,
                test_size=self.split_percentages[2]
                / (self.split_percentages[2] + self.split_percentages[1]),
            )

            split = Dataset_Split(self.hdf5_file, train, val, test)

        return Split(self.dataset, [split], self.base_hdf5_path)

    def create_training_loader(self, **kwargs) -> DataLoader:
        return self.data_fac.training_loader(num_workers=self.dataloader_workers)

    def create_validation_loader(self, **kwargs) -> DataLoader:
        return self.data_fac.validation_loader(num_workers=self.dataloader_workers)

    def create_test_loader(self, **kwargs) -> DataLoader:
        return self.data_fac.testing_loader(num_workers=self.dataloader_workers)
