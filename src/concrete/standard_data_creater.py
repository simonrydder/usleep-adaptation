import os
from typing import Any, Iterator

import h5py
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from csdp.csdp_pipeline.pipeline_elements.models import Dataset_Split, Split
from csdp.csdp_pipeline.pipeline_elements.pipeline import PipelineDataset
from csdp.csdp_pipeline.pipeline_elements.samplers import Determ_sampler, Random_Sampler
from src.config._data_config import DataConfig
from src.dataset.resnet.simple_images import SimpleImages
from src.dataset.simple.simple_linear import SimpleLinear
from src.interfaces.data_creater import DataCreater


class StandardDataCreater(DataCreater):
    def __init__(self, config: DataConfig) -> None:
        super().__init__()

        self.batch_size = config.batch_size
        self._define_number_of_workers(config.num_workers)
        self._define_splits(config.split_percentages)
        self.seed = config.random_state

        self.kfold = KFold(
            n_splits=int(self.test_size * 100),
            shuffle=True,
            random_state=self.seed,
        )

        splitter = HDF5Splitter if config.type == "hdf5" else CustomSplitter
        self.splitter = splitter(config)

    def __iter__(self) -> Iterator[tuple[DataLoader, DataLoader, DataLoader]]:
        splits = self.splitter.get_splits()
        for fold, (other, test) in enumerate(self.kfold.split(splits)):
            other = [splits[i] for i in other]
            test = [splits[i] for i in test]
            train, val = train_test_split(
                other, test_size=self.val_size, random_state=self.seed
            )
            self.train, self.validation, self.test = self.splitter.get_datasets(
                train,
                val,
                test,  # type: ignore
            )

            yield (
                self.create_training_loader(),
                self.create_validation_loader(),
                self.create_test_loader(),
            )

    def create_training_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.train_workers,
            pin_memory=True,
        )

    def create_validation_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.validation,
            batch_size=1,
            shuffle=False,
            num_workers=self.val_workers,
            persistent_workers=True,
        )

    def create_test_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=1,
            shuffle=False,
            num_workers=self.test_workers,
            persistent_workers=True,
        )

    def _define_number_of_workers(
        self, num_workers: int | tuple[int, int, int]
    ) -> None:
        """Define number of workers for train, val and test."""
        if isinstance(num_workers, tuple):
            self.train_workers = num_workers[0]
            self.val_workers = num_workers[1]
            self.test_workers = num_workers[2]

        else:
            self.train_workers = num_workers
            self.val_workers = num_workers
            self.test_workers = num_workers

    def _define_splits(self, splits: tuple[float, float, float]) -> None:
        self.train_size = splits[0]
        self.val_size = splits[1]
        self.test_size = splits[2]


class HDF5Splitter:
    def __init__(self, config: DataConfig) -> None:
        dataset = config.dataset
        if not dataset.endswith(".hdf5"):
            dataset = dataset + ".hdf5"

        self.dataset = dataset
        self.base_path = os.path.join("data", "hdf5")
        self.hdf5_file = os.path.join(self.base_path, self.dataset)

        with h5py.File(self.hdf5_file, "r") as hdf5:
            self.subjects = list(hdf5["data"].keys())  # type: ignore

        self.sleep_epochs_pr_sample = config.sleep_epochs
        self.num_batches = config.num_batches
        self.training_iterations = config.batch_size * self.num_batches

    def get_splits(self) -> list[Any]:
        return self.subjects

    def get_datasets(
        self, train: list[Any], val: list[Any], test: list[Any]
    ) -> tuple[Dataset, Dataset, Dataset]:
        data_split = Dataset_Split(self.hdf5_file, train, val, test)
        split = Split(self.dataset, [data_split], self.base_path)
        train_sampler = Random_Sampler(
            split, self.sleep_epochs_pr_sample, self.training_iterations
        )
        val_sampler = Determ_sampler(split, split_type="val")
        test_sampler = Determ_sampler(split, split_type="test")

        return (
            PipelineDataset(train_sampler, []),
            PipelineDataset(val_sampler, []),
            PipelineDataset(test_sampler, []),
        )


class CustomSplitter:
    def __init__(self, config: DataConfig) -> None:
        assert config.num_samples is not None, (
            "CustomSplitter must have number of samples defined."
        )
        self.num_samples = config.num_samples
        match config.dataset:
            case "linear":
                self.dataset = SimpleLinear(self.num_samples, distribution=2)
            case "images":
                self.dataset = SimpleImages(
                    self.num_samples, num_classes=3, distribution="shifted"
                )
            case _:
                NotImplementedError(
                    f"Dataset: {config.dataset} is not a custom dataset."
                )

    def get_splits(self) -> list[Any]:
        return list(range(self.num_samples))

    def get_datasets(
        self, train: list[Any], val: list[Any], test: list[Any]
    ) -> tuple[Dataset, Dataset, Dataset]:
        return (
            Subset(self.dataset, train),
            Subset(self.dataset, val),
            Subset(self.dataset, test),
        )
