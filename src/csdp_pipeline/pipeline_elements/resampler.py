# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

from typing import Any

import torch
from scipy.signal import resample_poly
from torch import Tensor

from src.csdp_pipeline.pipeline_elements.pipe import IPipe


class Resampler(IPipe):
    def __init__(self, source_sample: int, target_sample: int) -> None:
        """
        Args:
            source_sample: Original sampling rate (e.g., 128 Hz).
            target_sample: Desired sampling rate (e.g., 100 Hz).
        """
        self.source_sample = source_sample
        self.target_sample = target_sample

    def resample_collection(self, coll: Tensor) -> Tensor:
        """
        Resample each channel in a 2D tensor from `source_sample` to `target_sample`.

        Args:
            coll: 2D tensor of shape (num_channels, num_samples).

        Returns:
            A 2D tensor of shape (num_channels, new_num_samples) after resampling.
        """
        resampled = []
        for i in range(len(coll)):
            chnl = coll[i]  # 1D slice for channel i
            if self.source_sample != self.target_sample:
                chnl = resample_poly(
                    chnl.numpy(), self.target_sample, self.source_sample, axis=0
                )
            chnl = torch.tensor(chnl)
            resampled.append(chnl)

        return torch.stack(resampled, dim=0)

    def process(
        self, x: tuple[Tensor, Tensor, Tensor, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Any]:
        """
        IPipe-compatible process method. Resamples EEG and EOG data if necessary.

        Args:
            x: A 4-tuple containing:
                - eegs (Tensor): Shape (num_channels, num_samples).
                - eogs (Tensor): Shape (num_channels, num_samples).
                - labels (Tensor).
                - tags (Any): Additional metadata.

        Returns:
            A 4-tuple of:
                - eeg_resampled (Tensor): Resampled EEG data.
                - eog_resampled (Tensor): Resampled EOG data.
                - labels (Tensor): Unchanged.
                - tags (Any): Unchanged metadata.
        """
        eegs, eogs, labels, tags = x

        # Check dimensionality
        assert (
            eegs.dim() == 2 and eogs.dim() == 2
        ), f"eegs and eogs must be 2D (got shapes {eegs.shape}, {eogs.shape})"

        eeg_resampled = self.resample_collection(eegs)
        eog_resampled = self.resample_collection(eogs)

        return eeg_resampled, eog_resampled, labels, tags
