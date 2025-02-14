import os

import h5py
import torch
from torch.utils.data import Dataset


class HDF5SleepDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the directory containing .hdf5 files.
            transform (callable, optional): A callable (or pipeline) that processes
                                            a (data, label) pair before returning.
        """
        self.folder_path = folder_path
        self.transform = transform

        # Gather all .hdf5 files in the folder
        self.filepaths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".hdf5")
        ]
        self.filepaths.sort()  # optional, but keeps things orderly

        # Build an index mapping for [start, end) in each file
        self.file_info = []
        current_index = 0
        for hdf5_file in self.filepaths:
            with h5py.File(hdf5_file, "r") as f:
                length = len(f["X"])  # or whatever dataset name you have
            # This file spans [current_index, current_index + length)
            self.file_info.append((hdf5_file, current_index, current_index + length))
            current_index += length

        self.total_length = current_index

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which file the index belongs to
        for hdf5_file, start_idx, end_idx in self.file_info:
            if start_idx <= idx < end_idx:
                local_idx = idx - start_idx
                with h5py.File(hdf5_file, "r") as f:
                    # Read your data
                    data = f["X"][local_idx]  # shape depends on how it's stored
                    label = f["y"][local_idx]

                # Optional pipeline/transform step
                if self.transform is not None:
                    data, label = self.transform((data, label))

                # Convert to torch tensors if not already
                data = torch.as_tensor(data, dtype=torch.float)
                label = torch.as_tensor(label, dtype=torch.long)

                return data, label

        # Fallback if idx somehow wasn’t found
        raise IndexError(f"Index {idx} out of range")
