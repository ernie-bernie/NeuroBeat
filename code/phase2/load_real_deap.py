# load_real_deap.py
# NeuroBeat — Phase 2
# Loads real DEAP data from BDF files using MNE-Python
# Written by Evyn Ernest

import mne
import numpy as np
import matplotlib.pyplot as plt
mne.set_log_level('WARNING')
# -------------------------------
# Load one participant's BDF file
# BDF is a standard EEG format that MNE reads natively
# -------------------------------

# Load participant 1
raw = mne.io.read_raw_bdf(r"C:\\Users\\evyne\\Documents\\NeuroBeat\\NeuroBeat\\code\\phase2\\data\\s01.bdf", preload=True)

# Print basic info about the recording
print(raw.info)
print(f"\nSampling frequency: {raw.info['sfreq']} Hz")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names}")
print(f"Duration: {raw.times[-1]:.1f} seconds")


# -------------------------------
# Filter the EEG signal
# We only want the EEG channels, not GSR, respiration, etc.
# The frontal channels are most relevant for anxiety research
# -------------------------------

# Pick only EEG channels — drops GSR, respiration, temperature etc.
raw.pick_types(eeg=True)

# Apply a bandpass filter between 0.5 and 100 Hz
# This removes very slow drifts (below 0.5 Hz) and high frequency noise (above 100 Hz)
# that are not brain activity, they are just things like movement artifacts and electrical interference
raw.filter(0.5, 100, fir_window='hamming')

# Apply a notch filter at 50 Hz to remove electrical interference
# In Europe power lines run at 50 Hz which bleeds into EEG recordings
raw.notch_filter(50)

print(f"\nAfter filtering:")
print(f"Number of EEG channels: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names}")

# -------------------------------
# Plot a real EEG signal
# Pick the Fz channel — frontal midline, most relevant for anxiety
# -------------------------------

# Get the data as a numpy array
# Returns shape (n_channels, n_samples)
data, times = raw.get_data(return_times=True)

# Find the Fz channel index
fz_index = raw.ch_names.index('Fz')

# Plot 10 seconds of Fz channel
ten_seconds = int(10 * raw.info['sfreq'])

plt.figure(figsize=(12, 4))
plt.plot(times[:ten_seconds], 
         data[fz_index, :ten_seconds], 
         linewidth=0.8, color="steelblue")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.title("Real EEG signal — participant 1, channel Fz (frontal midline)")
plt.tight_layout()


from scipy.signal import welch

# -------------------------------
# Compute power spectrum on real EEG
# Using Fz channel — frontal midline, most relevant for anxiety
# -------------------------------

# Get 60 seconds of Fz data for a stable frequency estimate
sixty_seconds = int(60 * raw.info['sfreq'])
fz_data = data[fz_index, :sixty_seconds]

#Compute the power spectrum using Welch's method
frequencies, power = welch(fz_data, fs=raw.info['sfreq'], nperseg=512)

#Set up the plot with shaded frequency bands
plt.figure(figsize=(12, 4))
plt.plot(frequencies, power, color="steelblue", linewidth=0.8)
plt.axvspan(0.5, 4,   alpha=0.15, color="purple", label="Delta")
plt.axvspan(4,   8,   alpha=0.15, color="blue",   label="Theta")
plt.axvspan(8,   13,  alpha=0.15, color="green",  label="Alpha")
plt.axvspan(13,  30,  alpha=0.15, color="orange", label="Beta")
plt.axvspan(30,  100, alpha=0.15, color="red",    label="Gamma")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (µV²/Hz)")
plt.title("Real EEG power spectrum — participant 1, channel Fz")
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()



#------------------------------------
# Find the alpha/beta ratio for the Fz channel
#------------------------------------

def get_alpha_beta_ratio(frequencies, power):
    #Find the indices corresponding to the alpha and beta bands
    alpha_indices = np.where((frequencies >= 8) & (frequencies <= 13))[0]
    beta_indices = np.where((frequencies >= 13) & (frequencies <= 30))[0]

    #Calculate the average power in the alpha and beta bands
    alpha_power = np.mean(power[alpha_indices])
    beta_power = np.mean(power[beta_indices])

    return alpha_power / beta_power

print(f"\nAlpha/Beta ratio for Fz channel: {get_alpha_beta_ratio(frequencies, power):.3f}")

#--------------------------------
# Loop through all participants and compute alpha/beta ratio for Fz channel
#--------------------------------

# Create list rations to store results
ratios = {}

#This will take a few minutes to run since we are loading and processing 32 BDF files
#Kept to only processing 6 participants for testing purposes, but can be easily changed to 32 by changing the range below
for participant in range(1, 7):
    #Checks if the participant's BDF file exists and can be loaded, if not it will skip and print a message
    try:
        print(f"\nProcessing participant {participant:02d}...")
        
        raw = mne.io.read_raw_bdf(f"C:\\Users\\evyne\\Documents\\NeuroBeat\\NeuroBeat\\code\\phase2\\data\\s{participant:02d}.bdf", preload=True)
        raw.pick_types(eeg=True)
        raw.filter(0.5, 100, fir_window='hamming')
        raw.notch_filter(50)
        
        data, times = raw.get_data(return_times=True)
        fz_index = raw.ch_names.index('Fz')
        sixty_seconds = int(60 * raw.info['sfreq'])
        fz_data = data[fz_index, :sixty_seconds]
        frequencies, power = welch(fz_data, fs=raw.info['sfreq'], nperseg=512)
        
        ratio = get_alpha_beta_ratio(frequencies, power)
        ratios[participant] = ratio
        print(f"Participant {participant:02d} — Alpha/Beta ratio: {ratio:.3f}")
    
    #If not, it prints this
    except Exception as e:
        print(f"Participant {participant:02d} — SKIPPED: {e}")
        continue

# After processing all participants, print summary statistics
print(f"\nCompleted {len(ratios)} participants")
print(f"Average alpha/beta ratio: {np.mean(list(ratios.values())):.3f}")


#Shows the plots
plt.show()

