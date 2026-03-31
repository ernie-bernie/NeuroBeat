# load_deap.py
# NeuroBeat — Phase 2
# Loads synthetic EEG data in DEAP format, visualizes raw signals,
# computes power spectral density, and extracts alpha/beta ratio features.
# Written by Evyn Ernest

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Synthetic EEG data generator written by Claude
# Simulates DEAP-like structure
# 32 participants, 40 trials, 32 channels, 8064 samples
# -------------------------------

def generate_synthetic_deap(n_participants=32, n_trials=40, 
                             n_channels=32, n_samples=8064, 
                             sfreq=128):

    #np.random.seed(42)
    data = {}

    for p in range(n_participants):

        # EEG signals — random noise simulating raw EEG
        signals = np.random.randn(n_trials, n_channels, n_samples)

        # Labels — valence and arousal scores 1-9
        valence = np.random.uniform(1, 9, n_trials)
        arousal = np.random.uniform(1, 9, n_trials)
        labels = np.column_stack([valence, arousal])

        data[p] = {
            "signals": signals,
            "labels": labels
        }

    return data, sfreq

# -------------------------------
# Load the data
# -------------------------------

data, sfreq = generate_synthetic_deap()

print(f"Participants: {len(data)}")
print(f"Trials per participant: {data[0]['signals'].shape[0]}")
print(f"Channels per trial: {data[0]['signals'].shape[1]}")
print(f"Samples per trial: {data[0]['signals'].shape[2]}")
print(f"Sampling frequency: {sfreq} Hz")
print(f"Label shape: {data[0]['labels'].shape}")

# -------------------------------
# Plot a raw EEG signal
# One trial, one channel
# -------------------------------

trial = data[0]["signals"][0]   # first participant, first trial
channel = trial[0]              # first channel (Fp1)

time = np.linspace(0, len(channel) / sfreq, len(channel))

plt.figure(figsize=(12, 4))
plt.plot(time, channel, linewidth=0.8, color="steelblue")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.title("Raw EEG signal — participant 1, trial 1, channel Fp1")
plt.tight_layout()
plt.show()



from scipy.signal import welch

# -------------------------------
# Compute power spectral density
# Using Welch's FFT — same method as Paper 7
# -------------------------------



# Do the Fourier transform on one trial, one channel
frequencies, power = welch(channel, fs=sfreq, nperseg=256)
# fs=sfreq sets the sampling frequency, nperseg=256 defines the amount of samples to analyze at a time (window size)

# This will help us identify the power in different frequency bands (delta, theta, alpha, beta, gamma) and compute an average power for each band.
def get_band_power(frequencies, power, low, high):
    band = (frequencies >= low) & (frequencies <= high)
    return power[band].mean()

#Gets the power for each frequency band using set frequency ranges
#The frequency ranges are based on standard definitions of EEG bands:
delta = get_band_power(frequencies, power, 0.5, 4)
theta = get_band_power(frequencies, power, 4, 8)
alpha = get_band_power(frequencies, power, 8, 13)
beta  = get_band_power(frequencies, power, 13, 30)
gamma = get_band_power(frequencies, power, 30, 100)

#Print the average power for each band
print(f"\nBand powers:")
#The ".4f" is there so the output is kept to 4 decimal places
print(f"Delta: {delta:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Alpha: {alpha:.4f}")
print(f"Beta:  {beta:.4f}")
print(f"Gamma: {gamma:.4f}")

#Prints the ratio of alpha to beta power, which is often used as an indicator of relaxation (higher alpha) vs. alertness (higher beta)
print(f"\nAlpha/Beta ratio: {alpha/beta:.4f}")


#-------------------------------
# Plot the power spectral density
#-------------------------------


#Setting up the plot
plt.figure(figsize=(12, 4))
#Plots the EEG reading as a line 
plt.plot(frequencies, power, color="steelblue", linewidth=0.8)
#Labels the axes as frequency and power
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Power spectral density")
#Sets a limit on the x value to focus on the relevant frequency range for EEG analysis (0-100 Hz)
plt.xlim(0, 100)

#Plotting the frequency bands
#The alpha=0.15 is just to set the shading to 15% opacity
#The first two numbers in each axvspan are the start and end frequencies for that band
plt.axvspan(0.5, 4,   alpha=0.15, color="purple", label="Delta")
plt.axvspan(4,   8,   alpha=0.15, color="blue",   label="Theta")
plt.axvspan(8,   13,  alpha=0.15, color="green",  label="Alpha")
plt.axvspan(13,  30,  alpha=0.15, color="orange", label="Beta")
plt.axvspan(30,  100, alpha=0.15, color="red",    label="Gamma")

#Makes a legend to label the areas
plt.legend()
#Automatically adjusts the spacing and padding of everything in your plot so nothing gets cut off or overlaps
plt.tight_layout()
#Shows the plot on the screen
plt.show()


# -------------------------------
# Extract alpha/beta ratio for all trials using synthetic data
# Participant 1 only for now
# -------------------------------

#Takes in the signal data for participant 1
def extract_features(signals, sfreq):
    features = []
    for trial in signals:
        channel = trial[0]  # just channel 1 for now, like one electrode on scalp
        freqs, psd = welch(channel, fs=sfreq, nperseg=256)
        #gets the power for alpha and beta bands using the get_band_power function defined earlier
        a = get_band_power(freqs, psd, 8, 13)
        b = get_band_power(freqs, psd, 13, 30)
        #Computes the alpha/beta ratio and appends it to the features list
        features.append(a / b)
    #Returns the features as an array
    return np.array(features)

#Extracts the features for participant 1
p1_features = extract_features(data[0]["signals"], sfreq)
p1_labels   = data[0]["labels"]

#Prints the shape of the features array, which should be (40,) since there are 40 trials
print(f"\nFeature shape: {p1_features.shape}")
#Prints the first 5 alpha/beta ratios and the corresponding valence and arousal scores for participant 1
#Most will be around 1.0 because the data is random noise, but there will be some variation due to the randomness
print(f"First 5 alpha/beta ratios: {p1_features[:5].round(3)}")
#Prints the first 5 valence and arousal scores for participant 1, which are random values between 1 and 9
#For now, it is randomly generated, but in a real dataset, you would expect to see some correlation
print(f"First 5 valence scores: {p1_labels[:5, 0].round(2)}")
print(f"First 5 arousal scores: {p1_labels[:5, 1].round(2)}")