import pickle
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def get_alpha_beta_ratio(frequencies, power):
    alpha_indices = np.where((frequencies >= 8) & (frequencies <= 13))[0]
    beta_indices = np.where((frequencies >= 13) & (frequencies <= 30))[0]
    alpha_power = np.mean(power[alpha_indices])
    beta_power = np.mean(power[beta_indices])
    return alpha_power / beta_power

def extract_features(eeg_data, sfreq):
    # Returns a list of 5-feature vectors, one per trial
    features = []
    for i in range(eeg_data.shape[0]):
        trial_features = []

        # Alpha/beta ratio for F3 (2), Fz (18), F4 (19)
        for p in [2, 18, 19]:
            channel = eeg_data[i, p, :]
            freqs, power = welch(channel, fs=sfreq, nperseg=256)
            ratio = get_alpha_beta_ratio(freqs, power)
            trial_features.append(ratio)

        # Theta/beta ratio for Fz
        channel_fz = eeg_data[i, 18, :]
        freqs, power = welch(channel_fz, fs=sfreq, nperseg=256)
        theta = np.mean(power[(freqs >= 4) & (freqs <= 8)])
        beta = np.mean(power[(freqs >= 13) & (freqs <= 30)])
        trial_features.append(theta / beta)

        # Alpha asymmetry F3 minus F4
        ch_f3 = eeg_data[i, 2, :]
        ch_f4 = eeg_data[i, 19, :]
        freqs3, pow3 = welch(ch_f3, fs=sfreq, nperseg=256)
        freqs4, pow4 = welch(ch_f4, fs=sfreq, nperseg=256)
        alpha_f3 = np.mean(pow3[(freqs3 >= 8) & (freqs3 <= 13)])
        alpha_f4 = np.mean(pow4[(freqs4 >= 8) & (freqs4 <= 13)])
        trial_features.append(alpha_f3 - alpha_f4)

        features.append(trial_features)
    return features

def main():
    sfreq = 128
    all_features = []
    all_labels = []

    # Loop through all 32 participants
    for i in range(1, 33):
        file_path = (r"C:\Users\evyne\Documents\NeuroBeat\NeuroBeat"
                     rf"\code\phase2\data\s{i:02d}.dat")
        print(f"Loading participant {i:02d}...")

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        eeg_data = data["data"]   # (40, 40, 8064)
        labels = data["labels"]   # (40, 4)

        # Extract 5 features per trial
        features = extract_features(eeg_data, sfreq)
        all_features.extend(features)

        # Arousal label for each trial
        for trial_idx in range(len(features)):
            all_labels.append(labels[trial_idx][0])

    X = np.array(all_features, dtype=float).reshape(1280, 5)
    y = np.array(all_labels)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Median split for binary classification
    y_binary = np.where(y >= np.median(y), 1, 0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Train SVM
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)



    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # -------------------------------
    # Per-subject classification
    # Train and test on same participant
    # -------------------------------
    accuracies = []
    for i in range(1, 33):
        file_path = (r"C:\Users\evyne\Documents\NeuroBeat\NeuroBeat" rf"\code\phase2\data\s{i:02d}.dat")
    
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        eeg_data = data["data"]
        labels = data["labels"]
        
        X = np.array(extract_features(eeg_data, sfreq))
        y = labels[:, 0]
        
        y_binary = np.where(y >= np.median(y), 1, 0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Participant {i:02d}: {acc:.3f}")

    print(f"\nAverage Accuracy: {np.mean(accuracies):.3f}")
    print(f"Min: {np.min(accuracies):.3f}")
    print(f"Max: {np.max(accuracies):.3f}")
    
    best = np.argmax(accuracies) + 1
    worst = np.argmin(accuracies) + 1
    print(f"Best participant: {best:02d} ({np.max(accuracies):.3f})")
    print(f"Worst participant: {worst:02d} ({np.min(accuracies):.3f})")
if __name__ == "__main__":
    main()


#------------------------
# Run using python code/phase2/load_preprocessed.py
#------------------------


