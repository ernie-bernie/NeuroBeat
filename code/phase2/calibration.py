# calibration.py
# NeuroBeat — Phase 2
# Adaptive per-user calibration with feature set selection
# Written by Evyn Ernest

import pickle
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def get_alpha_beta_ratio(frequencies, power):
    alpha_indices = np.where((frequencies >= 8) & (frequencies <= 13))[0]
    beta_indices = np.where((frequencies >= 13) & (frequencies <= 30))[0]
    alpha_power = np.mean(power[alpha_indices])
    beta_power = np.mean(power[beta_indices])
    return alpha_power / beta_power

def extract_features_from_channels(eeg_data, sfreq, channels):
    features = []
    for i in range(eeg_data.shape[0]):
        trial_features = []
        for p in channels:
            channel = eeg_data[i, p, :]
            freqs, power = welch(channel, fs=sfreq, nperseg=256)
            ratio = get_alpha_beta_ratio(freqs, power)
            trial_features.append(ratio)
            theta = np.mean(power[(freqs >= 4) & (freqs <= 8)])
            beta = np.mean(power[(freqs >= 13) & (freqs <= 30)])
            trial_features.append(theta / beta)
            alpha = np.mean(power[(freqs >= 8) & (freqs <= 13)])
            trial_features.append(alpha)
            engagement = (alpha + theta) / (beta + 1e-8)
            trial_features.append(engagement)
        features.append(trial_features)
    return np.array(features)

def calibrate(eeg_data, labels, sfreq):
    feature_sets = {
        "Frontal (F3, Fz, F4)":   [2, 18, 19],
        "Temporal (T7, T8)":       [7, 25],
        "Occipital (O1, Oz, O2)":  [13, 14, 31],
        "All combined":            [2, 7, 13, 14, 18, 19, 25, 31],
    }

    indices = np.random.RandomState(42).permutation(40)
    eeg_data = eeg_data[indices]
    labels = labels[indices]

    y_all = labels[:, 0]
    median_val = np.median(y_all)
    y_binary = np.where(y_all >= median_val, 1, 0)

    best_score = 0
    best_name = None
    best_features = None
    results = {}

    for name, channels in feature_sets.items():
        X = extract_features_from_channels(eeg_data, sfreq, channels)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X, y_binary, cv=4)
        mean_score = scores.mean()
        results[name] = mean_score
        print(f"  {name}: {mean_score:.3f}")

        if mean_score >= 0.70:
            print(f"  Found feature set above 70% threshold.")
            best_score = mean_score
            best_name = name
            best_features = X
            break

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_features = X

    return best_name, best_score, best_features, y_binary

def main():
    sfreq = 128
    all_scores = []
    high_responders = 0
    low_responders = 0

    for i in range(1, 33):
        file_path = (r"C:\Users\evyne\Documents\NeuroBeat\NeuroBeat"
                     rf"\code\phase2\data\s{i:02d}.dat")
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        eeg_data = data["data"]
        labels = data["labels"]

        print(f"\nParticipant {i:02d}:")
        best_name, best_score, _, _ = calibrate(eeg_data, labels, sfreq)
        all_scores.append(best_score)

        if best_score >= 0.70:
            high_responders += 1
            print(f"  Best: {best_name} at {best_score:.3f} — calibrated successfully")
        elif best_score >= 0.50:
            print(f"  Best: {best_name} at {best_score:.3f} — moderate confidence")
        else:
            low_responders += 1
            print(f"  Best: {best_name} at {best_score:.3f} — needs improvement")

    print(f"\nOverall mean accuracy: {np.mean(all_scores):.3f}")
    print(f"Successfully calibrated (above 70%): {high_responders} participants")
    print(f"Low responders (below 50%): {low_responders} participants")

if __name__ == "__main__":
    main()



#------------------------
# Run using python code/phase2/calibration.py
#------------------------

