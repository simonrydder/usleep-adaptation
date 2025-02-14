# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import json
import math
from typing import Any, Callable, Dict, Optional, Tuple

import h5py
import numpy as np
import torch

from src.csdp_pipeline.pipeline_elements.pipe import IPipe


class Sampler(IPipe):
    def __init__(
        self,
        base_file_path: str,
        datasets: list[str],
        split_type: str,
        num_epochs: int,
        split_file_path: Optional[str] = None,
        subject_percentage: float = 1.0,
        eeg_picker_func: Optional[Callable[[list[str]], str]] = None,
        eog_picker_func: Optional[Callable[[list[str]], str]] = None,
    ) -> None:
        """
        Args:
            base_file_path: Base directory path where .hdf5 files reside.
            datasets: List of dataset names (without file extension).
            split_type: Identifier for the data split (e.g., "train", "val", "test").
            num_epochs: Number of epochs to sample in a segment.
            split_file_path: Optional path to a JSON file dictating data splits.
            subject_percentage: Fraction of subjects to load from each dataset [0.0, 1.0].
            eeg_picker_func: Optional function to pick an EEG channel from a list of channel names.
            eog_picker_func: Optional function to pick an EOG channel from a list of channel names.
        """
        self.base_file_path = base_file_path
        self.datasets = datasets
        self.split_type = split_type
        self.split_file = split_file_path
        self.subject_percentage = subject_percentage
        self.subjects, self.num_records = self.__list_files()
        print(
            f"Number of {split_type} subjects: {len(self.subjects)} "
            f"records: {self.num_records} - subject percentage: {subject_percentage}"
        )

        self.probs = self.calc_probs()
        self.epoch_length = num_epochs

        self.eeg_picker_func = eeg_picker_func
        self.eog_picker_func = eog_picker_func

    def process(
        self, x: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Main interface from IPipe. Continuously attempts to retrieve a valid sample.

        Args:
            x: Unused placeholder (pipeline interface). Could contain prior pipeline data.

        Returns:
            A 4-tuple of (EEG tensor, EOG tensor, label tensor, tag dict).
        """
        success = False
        sample: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]
        ] = None

        while not success:
            sample = self.__get_sample()
            if sample is not None:
                success = True

        return sample  # type: ignore  # sample is guaranteed to be non-None if success=True

    def calc_probs(self) -> list[float]:
        """
        Calculate the sampling probability for each dataset based on
        a combination of:
        - Stratified probability (relative to number of records)
        - Uniform probability across datasets

        Returns:
            A list of probabilities for each dataset in self.datasets.
        """
        total_num_datasets = len(self.datasets)
        total_num_records = sum(self.num_records)

        probs = []
        for i, _ in enumerate(self.datasets):
            num_records = self.num_records[i]
            strat_prob = num_records / total_num_records
            dis_prob = 1.0 / total_num_datasets

            # Weighted average
            prob_d = 0.5 * strat_prob + 0.5 * dis_prob
            probs.append(prob_d)

        return probs

    def __get_sample(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Randomly pick a dataset, subject, record, channel(s), and label segment,
        then return the sampled EEG/EOG data and labels.

        Returns:
            A 4-tuple (EEG tensor, EOG tensor, label tensor, tag dict) or None
            if sampling fails (e.g., channel not found).
        """
        possible_sets = self.datasets
        probs = self.probs

        # Choose random dataset
        r_dataset = np.random.choice(possible_sets, 1, p=probs)[0]
        index = possible_sets.index(r_dataset)

        subjects = self.subjects[index]
        if len(subjects) == 0:
            raise ValueError(
                f"No subjects in split type: {self.split_type} for dataset {r_dataset}"
            )

        r_subject = np.random.choice(subjects, 1)[0]

        with h5py.File(f"{self.base_file_path}/{r_dataset}.hdf5", "r") as hdf5:
            hdf5 = hdf5["data"]
            records = list(hdf5[r_subject].keys())
            # Choose random record
            r_record = np.random.choice(records, 1)[0]

            hyp = hdf5[r_subject][r_record]["hypnogram"][()]
            psg = list(hdf5[r_subject][r_record]["psg"].keys())

            # Pick EEG/EOG channels
            try:
                eeg_channel = (
                    self.eeg_picker_func(psg)
                    if self.eeg_picker_func is not None
                    else self.__pick_random_channel(psg, "EEG")
                )
                eog_channel = (
                    self.eog_picker_func(psg)
                    if self.eog_picker_func is not None
                    else self.__pick_random_channel(psg, "EOG")
                )
            except Exception:
                # Could not pick EEG or EOG channel
                return None

            # Choose a random label in hyp, then pick one of its indices
            label_set = np.unique(hyp)
            r_label = np.random.choice(label_set, 1)[0]
            indexes = [i for i in range(len(hyp)) if hyp[i] == r_label]
            r_index = np.random.choice(indexes, 1)[0]

            # Shift the position
            r_shift = np.random.choice(list(range(0, self.epoch_length)), 1)[0]
            assert r_shift <= 200  # Safety check if desired

            start_index = r_index - r_shift
            if start_index < 0:
                start_index = 0
            elif (start_index + self.epoch_length) >= len(hyp):
                start_index = len(hyp) - self.epoch_length

            y_data = hyp[start_index : start_index + self.epoch_length]
            y = torch.tensor(y_data)

            x_start_index = start_index * 128 * 30

            try:
                eeg_segment = hdf5[r_subject][r_record]["psg"][eeg_channel][
                    x_start_index : x_start_index + (self.epoch_length * 30 * 128)
                ]
            except Exception:
                eeg_segment = []

            try:
                eog_segment = hdf5[r_subject][r_record]["psg"][eog_channel][
                    x_start_index : x_start_index + (self.epoch_length * 30 * 128)
                ]
            except Exception:
                eog_segment = []

        x_eeg = torch.tensor(eeg_segment).unsqueeze(0)
        x_eog = torch.tensor(eog_segment).unsqueeze(0)

        tag: Dict[str, Any] = {
            "dataset": r_dataset,
            "subject": r_subject,
            "record": r_record,
            "eeg": eeg_channel,
            "eog": eog_channel,
            "start_idx": x_start_index,
            "end_idx": x_start_index + (self.epoch_length * 30 * 128),
        }

        return (x_eeg, x_eog, y, tag)

    def __pick_random_channel(self, channel_list: list[str], ch_type: str) -> str:
        """
        Filter channels in `channel_list` for those matching `ch_type`,
        then pick one at random.

        Args:
            channel_list: List of all available channels in a record.
            ch_type: Prefix to match (e.g., "EEG" or "EOG").

        Returns:
            A randomly chosen channel name.
        """
        channels = [ch for ch in channel_list if ch.startswith(ch_type)]
        return np.random.choice(channels, 1)[0]

    def __list_files(self) -> tuple[list[list[str]], list[int]]:
        """
        Build a list of subjects and number of records for each dataset
        based on the defined split (if split_file is provided).

        Returns:
            A tuple of (subjects, num_records):
                - subjects: List of list of subject IDs for each dataset.
                - num_records: Number of records for each dataset (indexed correspondingly).
        """
        subjects: list[list[str]] = []
        num_records: list[int] = []
        base_path = self.base_file_path

        for f in self.datasets:
            with h5py.File(base_path + "/" + f"{f}.hdf5", "r") as hdf5:
                hdf5 = hdf5["data"]
                if self.split_file is not None:
                    with open(self.split_file, "r") as splitfile:
                        splitdata = json.load(splitfile)
                        try:
                            # Try finding the correct split
                            sets = splitdata[f]
                            subs = sets[self.split_type]
                        except KeyError:
                            print("Could not find configured split in split_file")
                            exit()
                else:
                    subs = list(hdf5.keys())

                num_subjects = len(subs)
                num_subjects_to_use = math.ceil(num_subjects * self.subject_percentage)
                subs = subs[0:num_subjects_to_use]

                tot_records = 0
                subjects_to_add: list[str] = []

                for subj_key in subs:
                    try:
                        _ = hdf5[subj_key]
                    except KeyError:
                        print(
                            f"Did not find subject {subj_key} in dataset {f} "
                            f"for split_type {self.split_type}"
                        )
                        continue

                    records = len(hdf5[subj_key].keys())
                    tot_records += records
                    subjects_to_add.append(subj_key)

                num_records.append(tot_records)
                subjects.append(subjects_to_add)

        return subjects, num_records
