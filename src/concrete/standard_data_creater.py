import os
from abc import ABC, abstractmethod
from itertools import chain
from time import sleep
from typing import Any, Dict, Literal

import h5py
import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from csdp.csdp_pipeline.pipeline_elements.models import Dataset_Split, Split
from csdp.csdp_pipeline.pipeline_elements.pipeline import PipelineDataset
from csdp.csdp_pipeline.pipeline_elements.samplers import Determ_sampler, Random_Sampler
from src.config._data_config import DataConfig
from src.interfaces.data_creater import DataCreater


class StandardDataCreater(DataCreater):
    def __init__(self, config: DataConfig) -> None:
        super().__init__()

        self.batch_size = config.batch_size
        self._define_number_of_workers(config.num_workers)
        self.val_size = config.validation_size

        splitter = HDF5Splitter
        self.splitter = splitter(config)
        self.subjects = self.splitter.get_splits()

        np.random.shuffle(self.subjects)
        subject_split = np.array_split(self.subjects, config.num_fold)
        self.folds = {fold: sub.tolist() for fold, sub in enumerate(subject_split)}

    def get_dataloaders(
        self, fold: int, train_size: int | None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        test = self.folds[fold]

        rest = list(
            chain(*[subjects for i, subjects in self.folds.items() if i != fold])  # type: ignore
        )
        np.random.shuffle(rest)  # type: ignore
        val = rest[: self.val_size]
        end_idx = train_size if train_size is None else self.val_size + train_size
        train = rest[self.val_size : end_idx]

        self.train, self.validation, self.test = self.splitter.get_datasets(
            train,
            val,
            test,  # type: ignore
        )

        return (
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
        return self.subjects.copy()

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


class SessionData(BaseModel):
    hypnogram: np.ndarray  # Keep as NumPy array
    psg: Dict[str, np.ndarray]  # Dict of channel name to array

    model_config = {"arbitrary_types_allowed": True}


class SubjectData(BaseModel):
    sessions: Dict[str, SessionData]


class HDF5Data(BaseModel):
    subjects: Dict[str, SubjectData]
    dataset: str

    def get_subject_names(self) -> list[str]:
        return list(self.subjects.keys())

    def get_record_names(self) -> list[tuple[str, str]]:
        records = []
        for subject, subject_data in self.subjects.items():
            for session in subject_data.sessions.keys():
                records.append((subject, session))

        return records


def load_hdf5_data(dataset: str) -> HDF5Data:
    if not dataset.endswith(".hdf5"):
        dataset += ".hdf5"

    base_path = os.path.join("data", "hdf5")
    hdf5_file = os.path.join(base_path, dataset)

    subjects: Dict[str, SubjectData] = {}

    with h5py.File(hdf5_file, "r") as hdf5:
        data_group = hdf5["data"] if "data" in hdf5 else hdf5

        for subject_id, subject_group in data_group.items():  # type: ignore
            session_dict: Dict[str, SessionData] = {}
            for session_id, session_group in subject_group.items():
                hypnogram = session_group["hypnogram"][:]  # np.ndarray
                psg = {
                    channel: session_group["psg"][channel][:]
                    for channel in session_group["psg"]
                }
                session_dict[session_id] = SessionData(hypnogram=hypnogram, psg=psg)
            subjects[subject_id] = SubjectData(sessions=session_dict)

    return HDF5Data(subjects=subjects, dataset=dataset)


def split_hdf5_data(
    data: HDF5Data,
    train_subjects: list[str],
    validation_subjects: list[str],
    test_subjects: list[str],
) -> tuple[HDF5Data, HDF5Data, HDF5Data]:
    def build_subset(subject_ids: list[str]) -> HDF5Data:
        subset_data = {
            subject_id: data.subjects[subject_id] for subject_id in subject_ids
        }

        return HDF5Data(subjects=subset_data, dataset=data.dataset)

    return (
        build_subset(train_subjects),
        build_subset(validation_subjects),
        build_subset(test_subjects),
    )


class TagData(BaseModel):
    dataset: str
    subject: str
    record: str
    eeg: str
    eog: str


class SampleData(BaseModel):
    eeg: Tensor
    eog: Tensor
    labels: Tensor
    tag: TagData

    model_config = {"arbitrary_types_allowed": True}


class Sampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_sample(self, data: HDF5Data, index: int) -> SampleData: ...

    @abstractmethod
    def select_signal(self, channels: list[str]) -> str | None: ...

    def get_record_names(self, data: HDF5Data) -> list[tuple[str, str]]:
        records = []
        for subject, subject_data in data.subjects.items():
            for session in subject_data.sessions.keys():
                records.append((subject, session))

        return records

    def get_session(self, data: HDF5Data, record: tuple[str, str]) -> SessionData:
        subject_id, session_id = record
        return data.subjects[subject_id].sessions[session_id]

    def split_channels(self, channels: list[str]) -> tuple[list[str], list[str]]:
        eog_channels = [x for x in channels if x.startswith("EOG_")]
        eeg_channels = [x for x in channels if not x.startswith("EOG_")]

        return eog_channels, eeg_channels


class RandomSampler(Sampler):
    def __init__(self, num_sleep_epochs: int, num_batches: int):
        self.num_sleep_epochs = num_sleep_epochs
        self.points_pr_sleep_epoch = 30 * 128
        self.num_batches = num_batches
        super().__init__()

    def __len__(self) -> int:
        return self.num_batches

    def select_signal(self, channels: list[str]) -> str | None:
        try:
            return np.random.choice(channels, 1)[0]
        except ValueError:
            return None

    def get_sample(self, data: HDF5Data, index: int) -> SampleData:
        records = data.get_record_names()
        record_index = np.random.choice(range(len(records)), 1)[0]
        record = records[record_index]
        session = self.get_session(data, record)

        hyp = session.hypnogram
        assert len(hyp) > self.num_sleep_epochs
        start_index = np.random.randint(0, len(hyp) - self.num_sleep_epochs)
        end_index = start_index + self.num_sleep_epochs

        psg_channels = list(session.psg.keys())
        eog_channels, eeg_channels = self.split_channels(psg_channels)
        eeg_channel = self.select_signal(eeg_channels)
        assert eeg_channel is not None
        eog_channel = self.select_signal(eog_channels)
        if eog_channel is None:
            eog_channel = eeg_channel

        x_start = start_index * self.points_pr_sleep_epoch
        x_end = end_index * self.points_pr_sleep_epoch

        return SampleData(
            eeg=torch.tensor(session.psg[eeg_channel][x_start:x_end]).unsqueeze(0),
            eog=torch.tensor(session.psg[eog_channel][x_start:x_end]).unsqueeze(0),
            labels=torch.tensor(hyp[start_index:end_index]),
            tag=TagData(
                dataset=data.dataset,
                subject=record[0],
                record=record[1],
                eeg=eeg_channel,
                eog=eog_channel,
            ),
        )


class DetermSampler(Sampler):
    def __init__(self):
        super().__init__()

    def __len__(self) -> int:
        return 1

    def select_signal(self, channels: list[str]) -> str | None:
        try:
            return channels[0]
        except IndexError:
            return None

    def get_sample(self, data: HDF5Data, index: int) -> SampleData:
        record = data.get_record_names()[index]
        session = self.get_session(data, record)
        psg_channels = list(session.psg.keys())

        eog_channels, eeg_channels = self.split_channels(psg_channels)
        eeg_channel = self.select_signal(eeg_channels)
        assert eeg_channel is not None

        eog_channel = self.select_signal(eog_channels)
        if eog_channel is None:
            eog_channel = eeg_channel

        eeg = session.psg[eeg_channel]
        eog = session.psg[eog_channel]

        return SampleData(
            eeg=torch.tensor(eeg).unsqueeze(0),
            eog=torch.tensor(eog).unsqueeze(0),
            labels=torch.tensor(session.hypnogram),
            tag=TagData(
                dataset=data.dataset,
                subject=record[0],
                record=record[1],
                eeg=eeg_channel,
                eog=eog_channel,
            ),
        )


class UsleepDataset(Dataset):
    def __init__(
        self, data: HDF5Data, sampler: Sampler, mode: Literal["train", "test", "val"]
    ):
        self.data = data
        self.sampler = sampler
        self.mode = mode
        self.print_report()

    def __len__(self) -> int:
        return len(self.sampler) * sum(
            len(subject_data.sessions) for subject_data in self.data.subjects.values()
        )

    def __getitem__(self, idx: int) -> dict:
        return self.sampler.get_sample(self.data, idx).model_dump()

    def print_report(self) -> None:
        print(
            f"Records in {self.mode} - {self.__len__()}: {self.data.get_record_names()}"
        )


class ImprovedDataCreater(DataCreater):
    def __init__(self, config: DataConfig) -> None:
        for i in range(100):
            try:
                print(f"Reading data try number: {i + 1}")
                self.data = load_hdf5_data(config.dataset)
                break
            except Exception as e:
                print(e)
                sleep(0.5)
                continue

        self._define_number_of_workers(config.num_workers)
        self.val_size = config.validation_size
        self.batch_size = config.batch_size

        self.subjects = self.data.get_subject_names()

        np.random.shuffle(self.subjects)
        subject_split = np.array_split(self.subjects, config.num_fold)
        self.folds = {fold: sub.tolist() for fold, sub in enumerate(subject_split)}

        self.random_sampler = RandomSampler(config.sleep_epochs, config.num_batches)

    def get_dataloaders(
        self, fold: int, train_size: int | None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        test_subjects = self.folds[fold]

        rest = list(
            chain(*[subjects for i, subjects in self.folds.items() if i != fold])  # type: ignore
        )

        val_subjects = rest[: self.val_size]
        end_idx = train_size if train_size is None else self.val_size + train_size
        train_subjects = rest[self.val_size : end_idx]

        self.train, self.validation, self.test = split_hdf5_data(
            self.data, train_subjects, val_subjects, test_subjects
        )

        return (
            self.create_training_loader(),
            self.create_validation_loader(),
            self.create_test_loader(),
        )

    def create_training_loader(self) -> DataLoader:
        return DataLoader(
            dataset=UsleepDataset(self.train, self.random_sampler, "train"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.train_workers,
            pin_memory=True,
        )

    def create_validation_loader(self) -> DataLoader:
        return DataLoader(
            dataset=UsleepDataset(self.validation, DetermSampler(), "val"),
            batch_size=1,
            shuffle=False,
            num_workers=self.val_workers,
            persistent_workers=True,
        )

    def create_test_loader(self) -> DataLoader:
        return DataLoader(
            dataset=UsleepDataset(self.test, DetermSampler(), "test"),
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


if __name__ == "__main__":
    from lightning import seed_everything

    dcnf = DataConfig(
        dataset="eesm19",
        type="hdf5",
        batch_size=64,
        sleep_epochs=35,
        num_batches=3,
        num_workers=1,
        validation_size=2,
        num_fold=10,
    )

    seed_everything(42)
    improved = ImprovedDataCreater(dcnf)
    train, val, test = improved.get_dataloaders(1, None)

    for loader in train, val, test:
        fist = next(iter(loader))

    seed_everything(42)
    standard = StandardDataCreater(dcnf)
    _, val2, test2 = standard.get_dataloaders(1, None)

    seed_everything(42)
    first_standard = next(iter(test2))

    seed_everything(42)
    first_improved = next(iter(test))
    pass
