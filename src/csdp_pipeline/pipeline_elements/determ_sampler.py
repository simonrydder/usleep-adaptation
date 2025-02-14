# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import json
import math
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor

from src.csdp_pipeline.pipeline_elements.pipe import IPipe


class Determ_sampler(IPipe):
    def __init__(
        self,
        base_file_path: str,
        datasets: list[str],
        split_type: str,
        num_epochs: int,
        split_file: Optional[str] = None,
        subject_percentage: float = 1.0,
        get_all_channels: bool = False,
    ) -> None:
        """
        A deterministic sampler that fetches EEG and EOG data (plus labels/tags)
        from a given list of HDF5 files.

        Args:
            base_file_path: Directory path where the .hdf5 files reside.
            datasets: List of dataset names (without the ".hdf5" extension).
            split_type: The split identifier (e.g. "train", "val", "test").
            num_epochs: The number of epochs (time segments) in your dataset (not used here directly, but stored).
            split_file: Optional path to a JSON file that defines train/val/test subjects.
            subject_percentage: Fraction of subjects to load from each dataset (e.g. 0.5 loads half).
            get_all_channels: Whether to load all EEG/EOG channels or just one.
        """
        self.base_file_path = base_file_path
        self.datasets = datasets
        self.split_type = split_type
        self.split_file = split_file
        self.subject_percentage = subject_percentage
        self.records = self.list_records()
        print(f"Number of {split_type} records: {len(self.records)}")
        self.epoch_length = num_epochs
        self.get_all_channels = get_all_channels

    def process(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        IPipe-compatible method. Fetches an EEG, EOG, label, and tag
        at the specified index.

        Args:
            index: Index of the record to sample.

        Returns:
            A tuple:
                - EEG data (Tensor)
                - EOG data (Tensor)
                - Labels (Tensor)
                - Tag dictionary containing metadata
        """
        x_eeg, x_eog, label, tag = self.__get_sample(index)

        # If no EOG channel was found, replicate EEG as EOG
        if any(dim == 0 for dim in x_eog.shape):
            print("Found no EOG channel, duplicating EEG instead")
            x_eog = x_eeg

        return x_eeg, x_eog, label, tag

    def list_records(self) -> list[tuple[str, str, str]]:
        """
        Build a list of (dataset, subject, record) tuples from HDF5 files,
        optionally filtered by a split file.

        Returns:
            A list of 3-tuples (dataset_name, subject_key, record_key).
        """
        list_of_records: list[tuple[str, str, str]] = []

        for f in self.datasets:
            with h5py.File(f"{self.base_file_path}/{f}.hdf5", "r") as hdf5:
                if self.split_file is not None:
                    with open(self.split_file, "r") as splitfile:
                        splitdata = json.load(splitfile)
                        try:
                            sets = splitdata[f]
                            subjects = sets[self.split_type]
                        except KeyError:
                            print(
                                f"Could not find configured split for dataset {f} "
                                f"and split type {self.split_type}. All subjects are sampled."
                            )
                            subjects = list(hdf5.keys())
                else:
                    subjects = list(hdf5.keys())

                num_subjects = len(subjects)
                num_subjects_to_use = math.ceil(num_subjects * self.subject_percentage)
                subjects = subjects[:num_subjects_to_use]

                if not subjects:
                    raise ValueError(f"No subjects in split type: {self.split_type}")

                for s in subjects:
                    try:
                        records = list(hdf5[s])
                    except KeyError:
                        print(
                            f"Did not find subject {s} in dataset {f} "
                            f"for split type {self.split_type}"
                        )
                        continue

                    for r in records:
                        list_of_records.append((f, s, r))

        return list_of_records

    def __get_sample(self, index: int) -> tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Retrieve a sample (EEG, EOG, labels, tags) from the internally stored
        list of records at the given index.

        Args:
            index: Index into self.records.

        Returns:
            A tuple:
                - EEG data (Tensor)
                - EOG data (Tensor)
                - Label array (Tensor)
                - Tag dictionary with metadata
        """
        dataset, subject, rec = self.records[index]

        with h5py.File(f"{self.base_file_path}/{dataset}.hdf5", "r") as hdf5:
            y_data = hdf5[subject][rec]["hypnogram"][()]
            psg_channels = list(hdf5[subject][rec]["psg"].keys())

            eeg_data, eog_data, eeg_tag, eog_tag = self.__load_data(
                hdf5, subject, rec, psg_channels
            )

        tag: Dict[str, Any] = {
            "dataset": dataset,
            "subject": subject,
            "record": rec,
            "eeg": eeg_tag,
            "eog": eog_tag,
        }

        y = torch.tensor(y_data)
        return eeg_data, eog_data, y, tag

    def determine_single_key(self, keys: list[str]) -> tuple[list[str], str]:
        """
        From a list of channel keys, pick the first one if available.

        Args:
            keys: Channel key names.

        Returns:
            A 2-tuple:
                - A list containing either one key or an empty list.
                - The chosen key (or "none" if none was available).
        """
        if keys:
            key = keys[0]
            tag = key
            return [key], tag
        else:
            return [], "none"

    def __load_data(
        self, hdf5: h5py.File, subject: str, rec: str, psg_channels: list[str]
    ) -> tuple[Tensor, Tensor, str, str]:
        """
        Load EEG/EOG data from the given HDF5 handle, subject, record,
        and list of PSG channels.

        Args:
            hdf5: An open h5py.File object.
            subject: Subject key.
            rec: Record key.
            psg_channels: Available channels under "psg".

        Returns:
            A 4-tuple:
                - EEG data (Tensor) of shape (#EEG_channels, #samples)
                - EOG data (Tensor) of shape (#EOG_channels, #samples)
                - A string tag for the EEG channel(s)
                - A string tag for the EOG channel(s)
        """
        eeg_data_list = []
        eog_data_list = []

        available_eeg_keys = [x for x in psg_channels if x.startswith("EEG")]
        available_eog_keys = [x for x in psg_channels if x.startswith("EOG")]

        if not self.get_all_channels:
            eeg_keys, eeg_tag = self.determine_single_key(available_eeg_keys)
            eog_keys, eog_tag = self.determine_single_key(available_eog_keys)
        else:
            eeg_keys = available_eeg_keys
            eog_keys = available_eog_keys
            eeg_tag = "all"
            eog_tag = "all"

        for ch in eeg_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eeg_data_list.append(data)

        for ch in eog_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eog_data_list.append(data)

        eeg_data_array = np.array(eeg_data_list)
        eog_data_array = np.array(eog_data_list)

        eeg_tensor = torch.tensor(eeg_data_array)
        eog_tensor = torch.tensor(eog_data_array)

        return eeg_tensor, eog_tensor, eeg_tag, eog_tag
