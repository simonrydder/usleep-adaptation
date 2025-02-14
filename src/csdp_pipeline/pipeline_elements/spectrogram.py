# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

from typing import Tuple

import numpy as np
import torch

from src.csdp_pipeline.pipeline_elements.pipe import IPipe
from src.csdp_pipeline.preprocessing.spectrogram import create_spectrogram_images


class Spectrogram(IPipe):
    def __init__(
        self,
        win_size: float = 2.0,
        fs_fourier: int = 100,
        overlap: float = 1.0,
        sample_rate: int = 100,
    ) -> None:
        """
        Args:
            win_size: Window size (in seconds) for the spectrogram.
            fs_fourier: Sampling frequency used for Fourier transforms.
            overlap: Overlap fraction or overlap size (if you define it differently).
            sample_rate: Original signal sampling rate.
        """
        self.win_size = win_size
        self.fs_fourier = fs_fourier
        self.sample_rate = sample_rate
        self.overlap = overlap

    def spectrograms_for_collection(self, coll: np.ndarray) -> torch.Tensor:
        """
        Convert a collection of signals into a batch of spectrogram tensors.

        Args:
            coll: A 2D NumPy array where each row corresponds to one signal.

        Returns:
            A tensor containing stacked spectrograms for all signals in `coll`.
        """
        specs = []
        for i in range(len(coll)):
            _, _, spectrograms = create_spectrogram_images(
                coll[i], self.sample_rate, self.win_size, self.fs_fourier, self.overlap
            )
            # Convert list of spectrograms to NumPy array, then to Tensor
            spectrograms_tensor = torch.tensor(np.array(spectrograms))
            specs.append(spectrograms_tensor)

        return torch.stack(specs)

    def process(
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a tuple of (EEGs, EOGs, labels, tags) by creating spectrograms
        from the EEG and EOG signals.

        Args:
            x: A 4-tuple where:
                - x[0] = eegs (torch.Tensor), shape (num_eegs, num_samples)
                - x[1] = eogs (torch.Tensor), shape (num_eogs, num_samples)
                - x[2] = labels (torch.Tensor)
                - x[3] = tags (torch.Tensor)

        Returns:
            A 4-tuple with transformed EEGs, EOGs, labels, and tags. The EEGs
            and EOGs are converted into spectrogram tensors.
        """
        eegs, eogs, labels, tags = x

        # Channels, samples
        assert (
            eegs.dim() == 2 and eogs.dim() == 2
        ), "EEG and EOG inputs must be 2D: (num_signals, num_samples)."

        # Convert from torch.Tensor to NumPy for spectrogram creation
        eegs_np = eegs.numpy()
        eogs_np = eogs.numpy()

        # Convert to spectrograms
        eegs_spectrograms = self.spectrograms_for_collection(eegs_np)
        eogs_spectrograms = self.spectrograms_for_collection(eogs_np)

        return eegs_spectrograms, eogs_spectrograms, labels, tags
