# neurobeat.py
# NeuroBeat — Phase 2
# Main integration script — load, calibrate, predict, recommend
# Written by Evyn Ernest

import pickle
import numpy as np
import os
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier

from load_preprocessed import get_alpha_beta_ratio
from calibration import extract_features_from_channels, FEATURE_SETS

DATA_DIR = r"C:\Users\evyne\Documents\NeuroBeat\NeuroBeat\code\phase2\data"
SFREQ = 128

MUSIC_MAP = {
    "binaural":   "Binaural beats — entraining toward alpha",
    "classical":  "Classical music — shifting toward relaxed EEG",
    "calm":       "Ambient music — maintaining calm alpha state",
    "nature":     "Nature sounds — gentle alpha promotion",
    "sad":        "Melancholic music — emotional validation",
    "upbeat":     "Upbeat music — gentle stimulation from understimulation",
}

def get_brain_state(ratio, valence):
    if ratio < 1.0:
        return "anxious", "binaural"
    elif ratio < 2.0:
        return "neutral", "classical"
    elif ratio < 3.0:
        return "calm", "calm"
    elif ratio < 4.0:
        return "calm", "sad" if valence < 3.0 else "nature"
    else:
        return "very calm", "upbeat"

def run(participant_id=17):
    print(f"\n{'='*50}")
    print(f"NeuroBeat — Participant {participant_id:02d}")
    print(f"{'='*50}")

    # Load data
    file_path = os.path.join(DATA_DIR, f"s{participant_id:02d}.dat")
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    eeg_data = data["data"]
    labels = data["labels"]
    print(f"Loaded {eeg_data.shape[0]} trials")

    # Calibrate
    print("\nRunning calibration...")
    indices = np.random.RandomState(42).permutation(40)
    eeg_shuffled = eeg_data[indices]
    labels_shuffled = labels[indices]
    y_all = labels_shuffled[:, 0]
    y_binary = np.where(y_all >= np.median(y_all), 1, 0)

    best_score = 0
    best_name = None
    for name, channels in FEATURE_SETS.items():
        X = extract_features_from_channels(eeg_shuffled, SFREQ, channels)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y_binary, cv=4
        )
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name

    print(f"Best feature set: {best_name} ({best_score:.1%} accuracy)")

    # Train personal model
    best_channels = FEATURE_SETS[best_name]
    X_all = extract_features_from_channels(eeg_data, SFREQ, best_channels)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_all, np.where(labels[:, 0] >= np.median(labels[:, 0]), 1, 0))

    # Loop through trials
    print(f"\n{'Trial':>5} | {'Valence':>7} | {'Arousal':>7} | {'State':>10} | {'Predicted':>10} | Music")
    print("-" * 75)

    for i in range(40):
        channel = eeg_data[i, 18, :]  # Fz
        freqs, power = welch(channel, fs=SFREQ, nperseg=256)
        alpha = np.mean(power[(freqs >= 8) & (freqs <= 13)])
        beta = np.mean(power[(freqs >= 13) & (freqs <= 30)])
        ratio = alpha / beta

        actual_valence = labels[i][0]
        actual_arousal = labels[i][1]
        X_trial = extract_features_from_channels(
            eeg_data[i:i+1], SFREQ, best_channels)
        predicted = model.predict(X_trial)[0]
        if predicted == 1:  # positive valence
            if ratio > 4.0:
                music_category = "upbeat"
            else:
                music_category = "calm"
            state = "positive"
        else:  # negative valence
            if ratio < 1.0:
                music_category = "binaural"
            elif ratio < 3.0:
                music_category = "classical"
            else:
                music_category = "sad"
            state = "negative"
        predicted_label = "positive" if predicted == 1 else "negative"

        print(f"  {i+1:3d}  | {actual_valence:7.2f} | {actual_arousal:7.2f} | "
              f"{state:>10} | {predicted_label:>10} | {MUSIC_MAP[music_category]}")

    print(f"\nCalibration accuracy: {best_score:.1%}")
    print(f"Feature set used: {best_name}")
    print("\nNeuroBeat session complete.")

if __name__ == "__main__":
    run(participant_id=17)

#------------------------
# Run using python code/phase2/neurobeat.py
#------------------------