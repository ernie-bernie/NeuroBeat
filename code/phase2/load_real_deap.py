# load_real_deap.py
# NeuroBeat — Phase 2
# Loads real DEAP data from BDF files using MNE-Python
# Written by Evyn Ernest

import mne
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load one participant's BDF file
# BDF is a standard EEG format that MNE reads natively
# -------------------------------

# Load participant 1
raw = mne.io.read_raw_bdf("NeuroBeat\\code\\phase2\\data\\s01.bdf", preload=True)

# Print basic info about the recording
print(raw.info)
print(f"\nSampling frequency: {raw.info['sfreq']} Hz")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names}")
print(f"Duration: {raw.times[-1]:.1f} seconds")