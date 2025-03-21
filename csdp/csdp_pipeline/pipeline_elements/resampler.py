# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import torch
from scipy.signal import resample_poly

from csdp.csdp_pipeline.pipeline_elements.pipeline import IPipe


class Resampler(IPipe):
    def __init__(self,
                source_sample,
                target_sample):
        self.source_sample = source_sample
        self.target_sample = target_sample
    
    def resample_collection(self, coll):
        resampled = []
        
        for i in range(len(coll)):
            chnl = coll[i]
        
            if self.source_sample != self.target_sample:
                chnl = resample_poly(chnl, self.target_sample, self.source_sample, axis=0)
            
            chnl = torch.tensor(chnl)
            resampled.append(chnl)
            
        resampled = torch.stack(resampled, dim=0)    
        return resampled
    
    def process(self, x):
        eegs = x[0]
        eogs = x[1]
        labels = x[2]
        tags = x[3]

        # Channels, samples
        assert eegs.dim() == 2, eogs.dim() == 2
        
        eeg_resampled = self.resample_collection(eegs)
        eog_resampled = self.resample_collection(eogs)

        return eeg_resampled, eog_resampled, labels, tags        return eeg_resampled, eog_resampled, labels, tags